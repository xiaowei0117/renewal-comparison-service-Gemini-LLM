# Renewal Comparison Service

FastAPI microservice for extracting and comparing insurance policy documents.

## Features

- **`POST /extract`** — Extract structured data (premium, coverages, doc type) from PDF or image files
  - Uses **Gemini LLM** for intelligent extraction when `GEMINI_API_KEY` is set
  - Falls back to regex-based extraction if LLM fails or API key not provided
- **`POST /compare`** — Compare two policies and return premium/coverage differences
  - Generates **LLM-powered summary** of changes when Gemini is available
- **`GET /health`** — Health check endpoint

## Requirements

- Python 3.11+
- Tesseract OCR (for image processing)

## Setup

### Local Development

```bash
# Install tesseract (macOS)
brew install tesseract

# Install tesseract (Ubuntu)
sudo apt-get install tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Set Gemini API key for LLM-based extraction
export GEMINI_API_KEY=your_api_key_here
export GEMINI_MODEL=gemini-2.0-flash

# Run the service
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

### Docker

```bash
docker build -t renewal-comparison-service .
docker run -p 8001:8001 renewal-comparison-service
```

## Usage

### Extract Document

```bash
curl -X POST http://localhost:8001/extract \
  -F "file=@policy.pdf"
```

Response:
```json
{
  "text": "...",
  "premium": 1250.00,
  "doc_type": "homeowners",
  "coverages": {
    "Coverage A (Dwelling)": "$350,000",
    "Deductible": "$1,000"
  }
}
```

### Compare Policies

```bash
curl -X POST http://localhost:8001/compare \
  -H "Content-Type: application/json" \
  -d '{
    "old_premium": 1200.00,
    "old_coverages": {"Coverage A (Dwelling)": "$350,000"},
    "new_premium": 1250.00,
    "new_coverages": {"Coverage A (Dwelling)": "$375,000"}
  }'
```

Response:
```json
{
  "premium_diff": "Premium increase: $50.00 (from $1200.00 to $1250.00).",
  "coverage_diffs": [
    "Coverage A (Dwelling): $350,000 → $375,000."
  ]
}
```

## Integration with Next.js

Set the environment variable in `.env`:

```env
RENEWAL_SERVICE_URL=http://localhost:8001
```

Then update your API route to call this service instead of doing extraction inline (see `route-refactored.ts`).
