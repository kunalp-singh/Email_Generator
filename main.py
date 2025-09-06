from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal, Optional
import google.generativeai as genai
from google.api_core import retry
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

# ---------------- Models ----------------

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
    sub_variants: List[str] = Field(default_factory=list)  # prefer default_factory
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

class AudioRequest(BaseModel):
    email_body: str
    language: str = "en"

# ---------------- Lifespan ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GEMINI_API_KEY:
        raise ValueError("API key for Gemini is required")
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # quick sanity check
        _model = genai.GenerativeModel("gemini-1.5-flash")
        _ = _model.generate_content("Ping")
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        raise
    yield

app = FastAPI(
    title="Email Drip Campaign API by Error Pointers",
    description="Generate personalized email campaigns using Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Simple rate limiter ----------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 10, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # obtain client IP robustly
        ip = request.headers.get("x-forwarded-for", "").split(",").strip() or (request.client.host if request.client else "unknown")
        now = time.time()
        window = [t for t in self.requests[ip] if now - t < self.period]
        self.requests[ip] = window
        if len(window) >= self.calls:
            raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
        self.requests[ip].append(now)
        return await call_next(request)

app.add_middleware(RateLimitMiddleware, calls=10, period=60)

# ---------------- Gemini helpers ----------------

def get_gemini_client():
    return genai

def get_default_interests(job_title: str) -> List[str]:
    interests_map = {
        "developer": ["technology", "software development", "coding"],
        "manager": ["leadership", "team management", "business strategy"],
        "marketing": ["digital marketing", "brand strategy", "social media"],
        "sales": ["business development", "client relationships", "sales strategy"],
    }
    for key, interests in interests_map.items():
        if key.lower() in job_title.lower():
            return interests
    return ["professional development", "industry trends", "business growth"]

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
        client_interests = get_client_interests(account.contacts.name, account.contacts.job_title)
        prompt = f"""
Create a personalized email for the following business account:
Company: {account.account_name}
Industry: {account.industry}
Pain Points: {', '.join(account.pain_points)}
Campaign Stage: Email {email_number} of {total_emails}
Campaign Objective: {account.campaign_objective}
Recipient Job Title: {account.contacts.job_title}
Interest: {', '.join(client_interests)}
Tone: {tone}
Language: {account.language}

Generate a catchy and engaging subject line personalized for the account and campaign objective. Generate three distinct subject lines.

Then, write the email body with:
1) Personalized body aligned to pain points and interests
2) Clear call-to-action
3) Cohesive flow with subject

Return valid JSON with keys: "subject", "body", "call_to_action", optionally "sub_variants".
        """.strip()

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
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )
        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini API")
        response_text = response.text.strip()
    except Exception as e:
        logger.error(f"Error in generate_email_content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate email content: {str(e)}")

    try:
        json_match = re.search(r"``````", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            email_data = json.loads(json_str)
        else:
            email_data = json.loads(response_text)
    except json.JSONDecodeError:
        email_data = {
            "subject": ["Subject Line Here"],
            "body": response_text,
            "call_to_action": "Call to Action Here",
            "sub_variants": ["Subject Line Here"],
        }

    subject = email_data.get("subject", ["Subject Line Here"])
    if isinstance(subject, list):
        subject = subject

    sub_variants = email_data.get("sub_variants", [subject])
    if isinstance(sub_variants, str):
        sub_variants = [sub_variants]

    salutation = f" Best regards, The {account.account_name} Team"

    send_times = {
        "technology": "8 AM - 10 AM",
        "software": "8 AM - 10 AM",
        "retail": "1 PM - 3 PM",
        "e-commerce": "1 PM - 3 PM",
        "_default": "6 PM - 8 PM",
    }
    key = account.industry.lower()
    recommended_send_time = send_times.get(key, send_times["_default"])

    return [
        EmailVariant(
            subject=subject,
            body=(email_data.get("body", "") or "").replace("\n", ""),
            call_to_action=(email_data.get("call_to_action", "") or "").replace("\n", "") + salutation.replace("\n", ""),
            sub_variants=sub_variants,
            suggested_send_time=recommended_send_time,
        )
    ]

def generate_campaign(client: genai, account: Account, number_of_emails: int) -> Campaign:
    emails: List[Email] = []
    for contact in account.contacts:
        contact.group = random.choice(["A", "B"])
    for i in range(number_of_emails):
        tone = account.tone or "neutral"
        email_variants = generate_email_content(client, account, i + 1, number_of_emails, tone)
        emails.append(Email(variants=email_variants))
    return Campaign(account_name=account.account_name, emails=emails)

# ---------------- Routes ----------------

@app.post(
    "/generate-campaigns/",
    response_model=CampaignResponse,
    summary="Generate email campaigns",
    response_description="Generated email campaigns for the provided accounts",
)
def generate_campaigns(request: CampaignRequest, client: genai = Depends(get_gemini_client)) -> CampaignResponse:
    try:
        campaigns: List[Campaign] = []
        for account in request.accounts:
            try:
                campaign = generate_campaign(client, account, request.number_of_emails)
                campaigns.append(campaign)
            except Exception as e:
                logger.error(f"Error generating campaign for {account.account_name}: {str(e)}")
                continue
        if not campaigns:
            raise HTTPException(status_code=500, detail="Failed to generate any campaigns")
        return CampaignResponse(campaigns=campaigns)
    except Exception as e:
        logger.error(f"Error in generate_campaigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Campaign generation failed: {str(e)}")

@app.post(
    "/export-campaigns-csv/",
    summary="Export campaigns as CSV",
    response_description="CSV file containing all generated campaigns",
)
def export_campaigns_csv(request: CampaignRequest, client: genai = Depends(get_gemini_client)):
    campaigns_response = generate_campaigns(request, client)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Account Name", "Email Number", "Variant", "Subject", "Sub-Variants", "Body", "Call to Action", "Recommended Send Time"])
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
                    variant.suggested_send_time,
                ])
    output.seek(0)
    filename = f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})

def generate_tts_from_email(email_body: str, language: str = "en") -> StreamingResponse:
    try:
        text = email_body.replace("\n", "")
        tts = gTTS(text=text, lang=language, slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return StreamingResponse(audio_file, media_type="audio/mpeg", headers={"Content-Disposition": "attachment; filename=email_audio.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de"]

@app.post("/generate-email-audio/")
def generate_email_audio(payload: AudioRequest):
    language = payload.language or "en"
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Supported languages are: {', '.join(SUPPORTED_LANGUAGES)}")
    return generate_tts_from_email(email_body=payload.email_body, language=language)

# ---------------- Static and templates ----------------

BASE_DIR = Path(__file__).resolve().parent  # repo root if main.py is at root
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Optional: add a simple root page if you have templates/index.html
@app.get("/")
def index(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return {"status": "ok"}  # fallback JSON

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
