# Email Campaign Generator 📧

An intelligent, FastAPI-based email campaign generator powered by Google’s Gemini AI. It creates personalized, context-aware email campaigns with A/B testing, tone and language customization, export options (CSV and audio), and smart scheduling recommendations.

## Features

- AI-powered email generation: contextual email copy using Gemini AI with configurable tone and language.  
- A/B testing support: deterministic contact split into A/B groups for fair comparisons.  
- Multi-language support: generate email content in multiple languages.  
- Tone customization: formal, casual, enthusiastic, or neutral.  
- Export capabilities:  
  - Export campaigns to CSV.  
  - Generate audio versions of emails.  
- Smart scheduling: AI-recommended send times based on industry patterns.  
- Rate limiting: 10 requests/min per IP protection to mitigate abuse.  
- Interactive API documentation: Swagger UI at /docs.

## Quick Start

### Prerequisites

- Python 3.8+  
- A Google Gemini API key (GEMINI_API_KEY)  

### Installation

1) Clone the repository  
```bash
git clone https://github.com/kunalp-singh/Email_Generator.git
cd Email_Generator
```

2) Create and activate a virtual environment  
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3) Install dependencies  
```bash
pip install -r requirements.txt
```

4) Configure environment  
```bash
# Create .env
echo GEMINI_API_KEY=your_api_key_here > .env
# Optional
echo PORT=8000 >> .env
```

### Run the app

Development (auto-reload):  
```bash
python -m uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 to use the web UI.  
API docs available at http://localhost:8000/docs.

## API Documentation

Interactive docs: http://localhost:8000/docs

Key Endpoints:
- POST /generate-campaigns/ — Generate email campaigns for one or more accounts/contacts.  
- POST /export-campaigns-csv/ — Export generated campaigns as a CSV file.  
- POST /generate-email-audio/ — Generate an audio version (TTS) of email content.

### Example Request

```python
campaign_request = {
    "accounts": [{
        "account_name": "Tech Corp",
        "industry": "technology",
        "pain_points": ["efficiency", "automation"],
        "contacts": [{
            "name": "John Doe",
            "email": "john@example.com",
            "job_title": "CTO"
        }],
        "campaign_objective": "nurturing",
        "tone": "formal",
        "language": "en",
        "interest": "AI technology"
    }],
    "number_of_emails": 3
}
```

## Frontend

- Templates: Jinja2 (templates/index.html)  
- Static assets: HTML5, CSS3, JavaScript (static/css/styles.css, static/js/script.js)  
- The frontend provides a postal-themed UI for campaign entry, results review, CSV export, audio playback, copying, and printing.

## Project Structure

```
html email/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── .env
├── .gitignore
├── main.py
├── Procfile
├── README.md
└── requirements.txt
```

Notes:
- static/ holds CSS/JS.  
- templates/ holds the HTML template rendered by the FastAPI/Jinja2 server.  
- main.py exposes the API routes and serves the UI.  

## Configuration

Environment variables:
- GEMINI_API_KEY: Google Gemini access key (required).  
- PORT: Port to run the server (default: 8000).

Optional recommendations:
- Configure CORS allowlist for frontends in development or when hosting separately.  
- Add additional AI safety/tone parameters as needed.

## Security

- Rate limiting: requests capped at 10 per minute per IP.  
- CORS: enabled and configurable.  
- Secrets via .env: never commit .env to source control.  
- Pydantic validation: request payloads validated at the API boundary.

## Production Notes

- Use a production ASGI server setup (e.g., multiple workers):  
  ```bash
  python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
  ```
- Serve static assets via a reverse proxy/CDN for better caching.  
- Consider enabling TLS at the proxy layer.  
- Configure logging and error monitoring (e.g., JSON logs, structured traces).  
- If performing heavy workloads, consider queueing or background tasks.

## A/B Testing

- Contacts are split deterministically (e.g., by hashing email) into A/B cohorts so repeated runs preserve cohorts.  
- Store cohort choice in the exported CSV for analysis.

## CSV Export

- The API returns a CSV file attachment with campaign details suitable for spreadsheet and BI tools.  
- Includes subjects, bodies, CTAs, send-time suggestions, variants, and contact metadata.

## Audio Generation

- The API returns an audio stream of an email body (TTS).  
- Intended for accessibility, review via listening, or voicemail-style delivery experiments.  
- Ensure the client handles binary responses and sets correct content type on playback.

## Development Tips

- Keep script.js bindings after DOM ready for reliable button actions.  
- Use optional chaining with arrays (e.g., obj?.arr?.) to avoid runtime errors.  
- For printing, open a new window, write the document, and call print after the window load event fires.  
- For CSV Blob downloads, revoke the URL after a tick to avoid browser quirks.

## Contributing

1) Fork the repository  
2) Create a feature branch:  
```bash
git checkout -b feature/AmazingFeature
```
3) Commit changes:  
```bash
git commit -m "Add some AmazingFeature"
```
4) Push the branch:  
```bash
git push origin feature/AmazingFeature
```
5) Open a Pull Request

Please include clear descriptions, tests where applicable, and screenshots for UI changes.

## License

MIT License. See LICENSE for details.

## Authors

- Kunal Singh — Initial work — https://github.com/kunalp-singh

## Acknowledgments

- Google Gemini AI for the language model interface  
- FastAPI for the web framework  
- All contributors and users

## Support

For help or bug reports, create an issue in the GitHub repository.
