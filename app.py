from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import PyPDF2
import pytesseract
from PIL import Image
import io
import re
import os
import json
import asyncio
from typing import Dict, List, Optional
import httpx

# Explicitly set tesseract path for Windows
if os.name == "nt":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

app = FastAPI()

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

COVERAGE_TEMPLATES = {
    "auto": [
        {"key": "bodily_injury", "label": "Bodily Injury", "regex": r"bodily injury|\bbi\b"},
        {"key": "property_damage", "label": "Property Damage", "regex": r"property damage|\bpd\b"},
        {"key": "medical", "label": "Medical Payments", "regex": r"medical payments|med pay|medpay"},
        {"key": "uninsured", "label": "Uninsured Motorist", "regex": r"uninsured motorist|\bum\b"},
        {"key": "collision", "label": "Collision", "regex": r"collision"},
        {"key": "comprehensive", "label": "Comprehensive", "regex": r"comprehensive"},
    ],
    "homeowners": [
        {"key": "coverage_a", "label": "Coverage A (Dwelling)", "regex": r"coverage\s*a\b|dwelling\b"},
        {"key": "coverage_b", "label": "Coverage B (Other Structures)", "regex": r"coverage\s*b\b|other structures"},
        {"key": "coverage_c", "label": "Coverage C (Personal Property)", "regex": r"coverage\s*c\b|personal property"},
        {"key": "liability", "label": "Personal Liability", "regex": r"personal liability|liability"},
        {"key": "deductible", "label": "Deductible", "regex": r"deductible"},
    ],
    "dwelling_fire": [
        # Coverage A: "A. DWELLING $ 308,386" — value on same line
        {"key": "coverage_a", "label": "Coverage A (Dwelling)", "regex": r"a\.\s*dwelling|^dwelling\s+\$"},
        # Other Structures: label on one line, "Limit: $30,839" on next line
        {"key": "other_structures", "label": "Other Structures", "regex": r"other structures", "next_line": True},
        # Personal Liability: "C. PERSONAL LIABILITY – EACH OCCURRENCE $ 500,000"
        {"key": "liability", "label": "Personal Liability", "regex": r"c\.\s*personal liability|personal liability.*occurrence"},
        # Medical Payments: "D. MEDICAL PAYMENTS TO OTHERS $ 5,000"
        {"key": "medical", "label": "Medical Payments", "regex": r"d\.\s*medical payments|medical payments to others"},
        # Fair Rental Value: label on one line, "Limit: $61,678" on next line
        {"key": "fair_rental", "label": "Fair Rental Value", "regex": r"fair rental value", "next_line": True},
        # Deductible: appears as "$ 3,084 which applies to all perils" (standalone line)
        {"key": "deductible", "label": "Deductible", "regex": r"^\$\s*[\d,]+\s+which applies"},
    ],
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2, with OCR fallback for scanned PDFs."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Check if extraction looks valid (not garbled)
        # Garbled PDFs often have patterns like /i255 /1 /2 /3
        sample = text[:500]
        slash_count = sample.count('/')
        token_pattern_count = len(re.findall(r'/\w+\s+/\w+', sample))

        # Consider it garbled if:
        # 1. Too many slashes (> 20% of characters)
        # 2. Has PDF token patterns like "/i255 /1"
        # 3. Very short text
        is_garbled = (
            slash_count > len(sample) * 0.2 or
            token_pattern_count > 5 or
            len(text.strip()) < 100
        )

        if is_garbled:
            print(f"[extract_text_from_pdf] Text extraction looks garbled (slashes={slash_count}, tokens={token_pattern_count}, len={len(text)}), falling back to OCR")
            return ""  # Return empty to trigger OCR fallback in caller

        return text
    except Exception as e:
        print(f"[extract_text_from_pdf] PyPDF2 failed: {e}")
        return ""


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using pytesseract OCR."""
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


def detect_doc_type(text: str) -> str:
    """Detect document type from text content."""
    normalized = text.lower()
    # Check dwelling fire FIRST — these also contain "coverage a" so must be checked before homeowners
    if re.search(r"dwelling fire|dp-?3|dp-?1", normalized):
        return "dwelling_fire"
    if re.search(r"homeowners|homeowner|ho-?3", normalized):
        return "homeowners"
    if re.search(r"bodily injury|property damage|vehicle|auto policy|collision", normalized):
        return "auto"
    return "unknown"


def extract_numeric_tokens(line: str) -> List[str]:
    """Extract numeric values like $1,000.00 or 100/300."""
    tokens = []
    currency = re.findall(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", line)
    tokens.extend(currency)
    ratios = re.findall(r"\d{1,3}/\d{1,3}(?:,\d{3})?", line)
    tokens.extend(ratios)
    return tokens


def pick_coverage_value(line: str) -> str:
    """Pick the most relevant coverage value from a line."""
    tokens = extract_numeric_tokens(line)
    if not tokens:
        return ""
    ratio = next((t for t in tokens if "/" in t), None)
    return ratio or tokens[-1]


def parse_premium(text: str) -> Optional[float]:
    """Extract premium amount from text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Priority 1: look for "total premium" or "annual premium" lines
    for line in lines:
        if re.search(r"total\s+premium|annual\s+premium", line, re.I):
            tokens = extract_numeric_tokens(line)
            for token in tokens:
                value = float(token.replace("$", "").replace(",", ""))
                if 100 < value < 100000:  # sanity check: plausible premium range
                    return value

    # Priority 2: look for a standalone premium line (not "at no additional premium", "your premium", etc.)
    for line in lines:
        if re.search(r"^\s*(basic\s+)?premium\s*[\$\d]", line, re.I):
            tokens = extract_numeric_tokens(line)
            for token in tokens:
                value = float(token.replace("$", "").replace(",", ""))
                if 100 < value < 100000:
                    return value

    return None


def is_toc_line(line: str) -> bool:
    """Return True if line looks like a table of contents entry (e.g. 'Coverage B .... 2')."""
    return len(re.findall(r'\.{3,}', line)) > 0


def parse_coverages(text: str, doc_type: str) -> Dict[str, str]:
    """Extract coverage values based on document type."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Remove table of contents lines
    lines = [l for l in lines if not is_toc_line(l)]

    coverages = {}
    patterns = COVERAGE_TEMPLATES.get(doc_type, [])

    for i, line in enumerate(lines):
        for pattern in patterns:
            if pattern["label"] in coverages:
                continue  # already found
            if re.search(pattern["regex"], line, re.I):
                if pattern.get("next_line"):
                    # Value is on the next non-empty line
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    value = pick_coverage_value(next_line)
                else:
                    value = pick_coverage_value(line)
                if value:
                    coverages[pattern["label"]] = value
    return coverages


async def call_gemini(prompt: str) -> str:
    """Call Gemini API for LLM-based extraction."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
            },
        )

        if response.status_code == 429:
            # Rate limit - wait and retry once
            await asyncio.sleep(5)
            response = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
            })

        if not response.is_success:
            raise ValueError(f"Gemini API error: {response.status_code} {response.text[:200]}")

        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")


async def llm_extract_policy_data(text: str) -> tuple[Optional[float], str, Dict[str, str]]:
    """Use LLM to extract premium, doc type, and coverages from policy text."""
    prompt = f"""You are an insurance policy data extraction assistant. Extract structured data from the following policy document text.

Policy Text:
{text[:3000]}

Extract and return ONLY a valid JSON object with these fields (no other text):
{{
  "premium": <number or null>,
  "doc_type": "<homeowners|auto|dwelling_fire|unknown>",
  "coverages": {{
    "<coverage name>": "<coverage value>",
    ...
  }}
}}

Rules:
- premium: extract the annual or total premium amount (number only, no $ or commas)
- doc_type: identify the policy type based on content
- coverages: extract all coverage amounts/limits mentioned (e.g., "Coverage A (Dwelling)": "$350,000", "Bodily Injury": "100/300")
- Return valid JSON only, no markdown formatting or extra text
"""

    try:
        llm_response = await call_gemini(prompt)

        # Strip markdown code blocks if present
        llm_response = llm_response.strip()
        if llm_response.startswith("```"):
            llm_response = re.sub(r"```json\n?|```\n?", "", llm_response).strip()

        # Parse JSON
        data = json.loads(llm_response)
        premium = data.get("premium")
        doc_type = data.get("doc_type", "unknown")
        coverages = data.get("coverages", {})

        return (premium, doc_type, coverages)
    except Exception as e:
        print(f"[LLM extraction failed]: {e}")
        # Fall back to regex-based extraction
        return (None, "unknown", {})


class ExtractResponse(BaseModel):
    text: str
    premium: Optional[float]
    doc_type: str
    coverages: Dict[str, str]


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    """Extract structured data from a policy document (PDF or image)."""
    file_bytes = await file.read()
    filename = file.filename or ""

    # Determine file type and extract text
    if filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
        text = extract_text_from_pdf(file_bytes)
        # If PDF extraction failed/garbled, try OCR as fallback
        if not text or len(text.strip()) < 100:
            print("[extract] PDF text extraction failed/garbled, trying OCR on PDF pages")
            try:
                import pymupdf
                pdf_doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                text = ""
                # OCR first 3 pages
                num_pages = min(3, len(pdf_doc))
                for page_num in range(num_pages):
                    page = pdf_doc[page_num]
                    # Render page to image (at 2x resolution for better OCR)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img) + "\n"
                pdf_doc.close()
                print(f"[extract] OCR extracted {len(text)} characters from {num_pages} PDF pages")
            except Exception as e:
                print(f"[extract] OCR fallback failed: {e}")
    elif file.content_type and file.content_type.startswith("image/"):
        text = extract_text_from_image(file_bytes)
    else:
        text = ""

    # Try LLM-based extraction first if API key is available
    if GEMINI_API_KEY and text:
        try:
            llm_premium, llm_doc_type, llm_coverages = await llm_extract_policy_data(text)

            # Use LLM results if they seem valid
            if llm_doc_type != "unknown" or llm_premium is not None or llm_coverages:
                print(f"[extract] Using LLM extraction (doc_type={llm_doc_type}, premium={llm_premium})")
                return ExtractResponse(
                    text=text,
                    premium=llm_premium,
                    doc_type=llm_doc_type,
                    coverages=llm_coverages,
                )
        except Exception as e:
            print(f"[extract] LLM extraction failed, falling back to regex: {e}")

    # Fallback to regex-based extraction
    print("[extract] Using regex-based extraction")
    doc_type = detect_doc_type(text)
    premium = parse_premium(text)
    coverages = parse_coverages(text, doc_type)

    return ExtractResponse(
        text=text,
        premium=premium,
        doc_type=doc_type,
        coverages=coverages,
    )


class CompareRequest(BaseModel):
    old_premium: Optional[float]
    old_coverages: Dict[str, str]
    new_premium: Optional[float]
    new_coverages: Dict[str, str]


class CompareResponse(BaseModel):
    premium_diff: Optional[str]
    coverage_diffs: List[str]
    llm_summary: Optional[str] = None


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """Compare two policy documents and return differences."""
    diffs = []

    # Premium comparison
    premium_diff = None
    if req.old_premium is not None and req.new_premium is not None:
        delta = req.new_premium - req.old_premium
        direction = "increase" if delta >= 0 else "decrease"
        premium_diff = f"Premium {direction}: ${abs(delta):.2f} (from ${req.old_premium:.2f} to ${req.new_premium:.2f})."

    # Coverage comparison
    all_keys = set(req.old_coverages.keys()) | set(req.new_coverages.keys())
    for key in all_keys:
        before = req.old_coverages.get(key, "")
        after = req.new_coverages.get(key, "")
        if before and not after:
            diffs.append(f"{key}: removed (was {before}).")
        elif not before and after:
            diffs.append(f"{key}: added ({after}).")
        elif before and after and before != after:
            diffs.append(f"{key}: {before} → {after}.")

    # Generate LLM summary if available
    llm_summary = None
    if GEMINI_API_KEY and (premium_diff or diffs):
        try:
            summary_prompt = f"""You are an insurance agent explaining policy changes to a client. Summarize the following renewal changes in 2-3 clear, conversational sentences.

Premium change: {premium_diff or "No change"}
Coverage changes: {', '.join(diffs) if diffs else "No coverage changes detected"}

Write a brief, client-friendly summary focusing on what matters most (premium impact and major coverage changes). Keep it under 100 words."""

            llm_summary = await call_gemini(summary_prompt)
            llm_summary = llm_summary.strip()
            print(f"[compare] Generated LLM summary: {llm_summary[:100]}...")
        except Exception as e:
            print(f"[compare] LLM summary failed: {e}")

    return CompareResponse(
        premium_diff=premium_diff,
        coverage_diffs=diffs,
        llm_summary=llm_summary,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
