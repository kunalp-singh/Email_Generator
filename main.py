from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal
import google.generativeai as genai
from google.api_core import retry
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO, BytesIO
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from gtts import gTTS
from collections import defaultdict
import time
import os
from dotenv import load_dotenv
import json
import re
import csv
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    job_title: str = Field(..., min_length=1, max_length=100)
    group: Literal["A", "B"] = "A"

class Account(BaseModel):
    account_name: str = Field(..., min_length=1, max_length=200)
    industry: str = Field(..., min_length=1, max_length=100)
    pain_points: List[str] = Field(..., min_items=1, max_items=5)
    contacts: List[Contact] = Field(..., min_items=1)
    campaign_objective: Literal["awareness", "nurturing", "upselling"]
    interest: str = Field(..., min_length=1, max_length=100)
    tone: Literal["formal", "casual", "enthusiastic", "neutral"] = "neutral"
    language: str = Field(..., min_length=1, max_length=200)

class EmailVariant(BaseModel):
    subject: str
    body: str
    call_to_action: str
    sub_variants: List[str] = []
    suggested_send_time: str

class Email(BaseModel):
    variants: List[EmailVariant]

class Campaign(BaseModel):
    account_name: str
    emails: List[Email]

class CampaignRequest(BaseModel):
    accounts: List[Account] = Field(..., min_items=1, max_items=10)
    number_of_emails: int = Field(..., gt=0, le=10)

class CampaignResponse(BaseModel):
    campaigns: List[Campaign]

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GEMINI_API_KEY:
        raise ValueError("API key for Gemini is required")
    yield

app = FastAPI(
    title="Email Drip Campaign API by Error Pointers",
    description="Generate personalized email campaigns using Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 10, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        self.requests[client_ip] = [req_time for req_time in self.requests[client_ip] 
                                  if now - req_time < self.period]
        
        if len(self.requests[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
            
        self.requests[client_ip].append(now)
        response = await call_next(request)
        return response

app.add_middleware(RateLimitMiddleware, calls=10, period=60)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Test")
    if not response:
        raise ValueError("Unable to generate test content")
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise HTTPException(
        status_code=500,
        detail="Failed to initialize AI model. Please check your API key."
    )

def get_gemini_client():
    return genai

def get_default_interests(job_title: str) -> List[str]:
    interests_map = {
        "developer": ["technology", "software development", "coding"],
        "manager": ["leadership", "team management", "business strategy"],
        "marketing": ["digital marketing", "brand strategy", "social media"],
        "sales": ["business development", "client relationships", "sales strategy"],
    }
    
    default_interests = ["professional development", "industry trends", "business growth"]
    for key, interests in interests_map.items():
        if key.lower() in job_title.lower():
            return interests
            
    return default_interests

def get_client_interests(name: str, job_title: str) -> List[str]:
    return get_default_interests(job_title)

def search_client_interests(name: str, job_title: str) -> List[str]:
    try:
        logger.info(f"Using fallback interests for {name}")
        return get_default_interests(job_title)
                                              
    except Exception as e:
        logger.error(f"Error in search_client_interests: {str(e)}")
        return ["professional development", "industry trends", "business growth"]

@retry.Retry(predicate=retry.if_exception_type(Exception))
def generate_email_content(client: genai, account: Account, email_number: int, total_emails: int, tone: str) -> List[EmailVariant]:
    try:
        client_interests = get_client_interests(account.contacts[0].name, account.contacts[0].job_title)

        prompt = f"""
        Create a personalized email for the following business account:
        Company: {account.account_name}
        Industry: {account.industry}
        Pain Points: {', '.join(account.pain_points)}
        Campaign Stage: Email {email_number} of {total_emails}
        Campaign Objective: {account.campaign_objective}
        Recipient Job Title: {account.contacts[0].job_title}
        Interest: {', '.join(client_interests)}  # Use interests from SerpAPI
        Tone: {tone}
        Language: {account.language}

        Generate a catchy and engaging subject line, personalized for the account and campaign objective. Please generate **three distinct subject lines**.

        Then, write the email body content with the following structure:
        1. An engaging email body personalized to the pain points and interest of the account
        2. A clear call-to-action encouraging the recipient to take the next step.
        3. Ensure the body is cohesive and flows well with the subject.

        Format the response as valid JSON with keys: "subject", "body", "call_to_action"
        """

        model = client.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )
        
        if not response or not response.text:
            raise ValueError("Empty response from Gemini API")
            
        response_text = response.text.strip()
    except Exception as e:
        logger.error(f"Error in generate_email_content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate email content: {str(e)}"
        )

    try:
        json_match = re.search(r"```json(.+?)```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()  # Extract the JSON string
            email_data = json.loads(json_str)  # Parse the JSON string
        else:
            email_data = json.loads(response_text)
    except json.JSONDecodeError:
        email_data = {
            "subject": ["Subject Line Here"],  # Default subject as a list
            "body": response_text,
            "call_to_action": "Call to Action Here",
            "sub_variants": ["Subject Line Here"]
        }

    subject = email_data.get("subject", ["Subject Line Here"])
    if isinstance(subject, list):
        subject = subject[0]  # Use the first subject line

    sub_variants = email_data.get("sub_variants", [subject])
    if isinstance(sub_variants, str):
        sub_variants = [sub_variants]  # Convert to list if it's a single string

    salutation = f"Best regards, The {account.account_name} Team"

    send_times = {
        "morning": "8 AM - 10 AM",
        "afternoon": "1 PM - 3 PM",
        "evening": "6 PM - 8 PM",
    }

    if account.industry.lower() in ["technology", "software"]:
        recommended_send_time = send_times["morning"]
    elif account.industry.lower() in ["retail", "e-commerce"]:
        recommended_send_time = send_times["afternoon"]
    else:
        recommended_send_time = send_times["evening"]

    return [
        EmailVariant(
            subject=subject,
            body=email_data["body"].replace("\n", ""),
            call_to_action=email_data["call_to_action"].replace("\n", "") + salutation.replace("\n", ""),
            sub_variants=sub_variants,
            suggested_send_time=recommended_send_time
        )
    ]

def generate_campaign(client: genai, account: Account, number_of_emails: int) -> Campaign:
    emails = []
    for contact in account.contacts:
        contact.group = random.choice(["A", "B"])

    for i in range(number_of_emails):
        tone = account.tone if account.tone else "neutral"
        email_variants = generate_email_content(client, account, i + 1, number_of_emails, tone)
        emails.append(Email(variants=email_variants))

    return Campaign(account_name=account.account_name, emails=emails)

@app.post(
    "/generate-campaigns/",
    response_model=CampaignResponse,
    summary="Generate email campaigns",
    response_description="Generated email campaigns for the provided accounts"
)
def generate_campaigns(
    request: CampaignRequest,
    client: genai = Depends(get_gemini_client)
) -> CampaignResponse:
    try:
        campaigns = []
        for account in request.accounts:
            try:
                campaign = generate_campaign(client, account, request.number_of_emails)
                campaigns.append(campaign)
            except Exception as e:
                logger.error(f"Error generating campaign for {account.account_name}: {str(e)}")
                continue
                
        if not campaigns:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any campaigns"
            )
            
        return CampaignResponse(campaigns=campaigns)
        
    except Exception as e:
        logger.error(f"Error in generate_campaigns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Campaign generation failed: {str(e)}"
        )

@app.post(
    "/export-campaigns-csv/",
    summary="Export campaigns as CSV",
    response_description="CSV file containing all generated campaigns"
)
def export_campaigns_csv(
    request: CampaignRequest,
    client: genai = Depends(get_gemini_client)
):
    campaigns_response = generate_campaigns(request, client)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Account Name', 'Email Number', 'Variant', 'Subject', 'Sub-Variants', 'Body', 'Call to Action', 'Recommended Send Time'])

    for campaign in campaigns_response.campaigns:
        for email_idx, email in enumerate(campaign.emails, 1):
            for variant_idx, variant in enumerate(email.variants, 1):
                writer.writerow([
                    campaign.account_name,
                    f"Email {email_idx}",
                    f"Variant {variant_idx}",
                    variant.subject,
                    "; ".join(variant.sub_variants),
                    variant.body,
                    variant.call_to_action,
                    variant.suggested_send_time  # Use the correct attribute
                ])

    output.seek(0)
    filename = f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

def generate_tts_from_email(email_body: str, language: str = "en", speed: float = 1.5) -> StreamingResponse:
    try:
        email_body = email_body.replace("\n", "")
        tts = gTTS(text=email_body, lang=language, slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)  # Use `write_to_fp` for BytesIO
        audio_file.seek(0)

        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=email_audio.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de']  # Add more as needed

@app.post("/generate-email-audio/")
def generate_email_audio(email_body: str, language: str = "en"):
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language. Supported languages are: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    return generate_tts_from_email(email_body=email_body, language=language)

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)