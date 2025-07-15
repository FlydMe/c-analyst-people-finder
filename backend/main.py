from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import requests
import json
import os
from datetime import datetime, timedelta
import threading
import re
from dotenv import load_dotenv
import calendar
import traceback
import logging
from logging.handlers import RotatingFileHandler
from fastapi import Request
from fastapi.responses import JSONResponse
import uvicorn
load_dotenv()

app = FastAPI(title="VC Analyst Search Tool API")

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",  # React dev server
        "https://c-analyst-people-finder-1.onrender.com"  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

MAX_QUERIES_PER_DAY = 100
query_count = 0
last_reset_date = datetime.utcnow().date()
lock = threading.Lock()

# --- Simple in-memory cache for Google Custom Search results ---
google_search_cache = {}

COMPANY_VARIANTS = {
    "OpenAI": ["OpenAI", "OpenAI LP", "OpenAI Inc", "openai.com"],
    "Google": ["Google", "Google LLC", "Alphabet", "google.com"],
    "Amazon": ["Amazon", "Amazon.com, Inc.", "amazon.com"],
    "Meta": ["Meta", "Meta Platforms", "Facebook", "facebook.com", "meta.com"],
    "Apple": ["Apple", "Apple Inc.", "apple.com"],
    "Netflix": ["Netflix", "Netflix, Inc.", "netflix.com"],
    # Add more as needed
}

def enhance_company_variants(company: str) -> list:
    return COMPANY_VARIANTS.get(company, [company])

def reset_query_count_if_needed():
    global query_count, last_reset_date
    today = datetime.utcnow().date()
    if today != last_reset_date:
        query_count = 0
        last_reset_date = today

def get_date_range_from_quit_window(quit_window: str):
    """Parse quit window like '6 months' or '1 year' and return (start_date, end_date) as YYYY-MM-DD."""
    now = datetime.utcnow().date()
    if not quit_window:
        return None, None
    quit_window = quit_window.lower().strip()
    if "month" in quit_window:
        num = int(''.join(filter(str.isdigit, quit_window)))
        start = now - timedelta(days=30*num)
    elif "year" in quit_window:
        num = int(''.join(filter(str.isdigit, quit_window)))
        start = now - timedelta(days=365*num)
    else:
        return None, None
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

class SearchParams(BaseModel):
    company: str
    roles: Optional[List[str]] = []
    seniority: Optional[str] = ""
    quitWindow: Optional[str] = ""
    geography: Optional[str] = ""
    includeKeywords: Optional[List[str]] = []
    excludeKeywords: Optional[List[str]] = []
    sources: Optional[List[str]] = None  # New: list of sources (e.g. ["linkedin", "twitter", ...])

class SearchResult(BaseModel):
    rank: int
    snippet: str
    domain: str
    date: str
    link: str
    relevanceScore: float

class QueryRequest(BaseModel):
    queries: List[str]
    searchParams: SearchParams
    sources: Optional[List[str]] = None  # New: for explicit override

def get_best_openai_model():
    """Return the best available OpenAI model name."""
    # Try GPT-4o, then GPT-4, then GPT-3.5-turbo
    for model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
        try:
            # This is a soft check; if the model is not available, OpenAI will error
            return model
        except Exception:
            continue
    return "gpt-3.5-turbo"

def get_role_variants(role):
    """Return a list of common LinkedIn variants/synonyms for a given role using ChatGPT if available, else fallback to static dictionary."""
    role = role.lower().strip()
    # Try dynamic expansion with ChatGPT
    if OPENAI_API_KEY:
        try:
            model = get_best_openai_model()
            prompt = (
                f"List the most common LinkedIn job titles, synonyms, and abbreviations for the role '{role}'. "
                "Focus on real LinkedIn nomenclature, including abbreviations, internal titles, and common variants. "
                "Return a comma-separated list, no commentary."
            )
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256
            )
            content = None
            if isinstance(response, dict) and "choices" in response and isinstance(response["choices"], list):
                first_choice = response["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice and "content" in first_choice["message"]:
                    content = first_choice["message"]["content"].strip()
            if content:
                # Parse comma-separated list
                variants = [v.strip() for v in content.split(",") if v.strip()]
                if role not in variants:
                    variants.append(role)
                return list(set(variants))
        except Exception as e:
            print(f"ChatGPT role expansion failed for '{role}': {e}")
    # Fallback to static dictionary
    variants = {
        "engineer": ["engineer", "software engineer", "developer", "swe", "sde", "technical lead", "programmer", "coder", "dev", "full stack engineer", "frontend engineer", "backend engineer", "systems engineer", "platform engineer", "infrastructure engineer", "site reliability engineer", "sre", "qa engineer", "test engineer", "embedded engineer", "mobile engineer", "ios engineer", "android engineer", "web developer", "application engineer", "cloud engineer", "data engineer", "ml engineer", "ai engineer", "machine learning engineer", "deep learning engineer", "automation engineer", "release engineer", "build engineer", "integration engineer", "support engineer", "network engineer", "security engineer", "devops engineer", "blockchain engineer", "game developer", "game engineer", "graphics engineer", "simulation engineer", "hardware engineer", "firmware engineer", "solutions engineer", "consulting engineer", "principal engineer", "lead engineer", "senior engineer", "junior engineer", "associate engineer", "entry level engineer", "graduate engineer", "intern engineer"],
        "product manager": ["product manager", "pm", "product owner", "product lead", "product director", "group product manager", "senior product manager", "associate product manager", "apm", "principal product manager", "vp product", "head of product", "product strategist", "product analyst", "product specialist", "product coordinator", "product consultant"],
        "designer": ["designer", "ux designer", "ui designer", "product designer", "visual designer", "interaction designer", "graphic designer", "web designer", "motion designer", "creative director", "art director", "ux researcher", "ux architect", "ux strategist", "ux writer", "service designer", "experience designer", "industrial designer", "brand designer", "senior designer", "junior designer", "lead designer", "principal designer", "design manager", "design director", "head of design"],
        "data scientist": ["data scientist", "ml engineer", "ai engineer", "machine learning engineer", "data analyst", "data engineer", "research scientist", "applied scientist", "statistician", "quantitative analyst", "quant", "business intelligence analyst", "bi analyst", "analytics engineer", "senior data scientist", "junior data scientist", "lead data scientist", "principal data scientist", "chief data scientist"],
        "manager": ["manager", "engineering manager", "team lead", "lead", "supervisor", "project manager", "scrum master", "agile coach", "program manager", "delivery manager", "development manager", "product manager", "operations manager", "support manager", "qa manager", "test manager", "release manager", "build manager", "integration manager", "network manager", "security manager", "devops manager", "senior manager", "junior manager", "associate manager", "principal manager", "director", "head of"],
        "director": ["director", "head of", "vp", "vice president", "chief", "cto", "ceo", "cpo", "coo", "cfo", "cio", "ciso", "founder", "co-founder", "managing director", "executive director", "board member", "chairman", "president", "senior director", "associate director", "principal director", "lead director", "managing partner", "general manager", "country manager", "regional manager", "area manager", "business unit manager"],
        # Add more as needed
    }
    for key, vals in variants.items():
        if role in key or key in role:
            return list(set(vals + [role]))
    return [role]

def generate_xray_queries(params: SearchParams) -> tuple[list[str], str]:
    """Generate Google X-ray search queries using advanced prompt engineering and ChatGPT, with robust role expansion and always 8 queries."""
    def get_after_before_from_quit_window(quit_window: str = ""):
        now = datetime.utcnow().date()
        if not quit_window:
            return None, None
        quit_window = quit_window.lower().strip()
        if "month" in quit_window:
            num = int(''.join(filter(str.isdigit, quit_window)))
            start = now - timedelta(days=30*num)
        elif "year" in quit_window:
            num = int(''.join(filter(str.isdigit, quit_window)))
            start = now - timedelta(days=365*num)
        else:
            return None, None
        return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

    company_variants = enhance_company_variants(params.company)
    after, before = get_after_before_from_quit_window(params.quitWindow or "")

    # --- Expand roles using synonym dictionary or ChatGPT ---
    expanded_roles = []
    for role in (params.roles or []):
        expanded_roles.extend(get_role_variants(role))
    expanded_roles = list(set(expanded_roles))
    if not expanded_roles and params.seniority:
        expanded_roles = [params.seniority]

    # --- Use GPT-4o as the default model for 9 queries, with company match as non-negotiable ---
    model = "gpt-4o"
    try:
        openai.Model.retrieve(model)
    except Exception:
        try:
            model = "gpt-4-turbo"
            openai.Model.retrieve(model)
        except Exception:
            model = "gpt-3.5-turbo"

    system_prompt = (
        "You are a world-class AI assistant for top-tier venture capital analysts. Your job is to generate 9 high-quality, diverse, and realistic Google X-ray queries that return relevant LinkedIn profiles of individuals who have recently left a given company and held roles similar to the ones provided.\n\n"
        "---\n\n"
        "### You MUST:\n"
        "- Start every query with: `site:linkedin.com/in`\n"
        "- The company must be a clear match: only ex-employees of the specified company.\n"
        "- The role/job must be similar: use synonyms, fuzzy matching, and real-world LinkedIn job title variants.\n"
        "- For each query, split job title synonyms into logical buckets (e.g., leadership, ICs, support roles).\n"
        "- Limit OR chains to 4–6 terms per group in each query.\n"
        "- Make 9 queries, each with a different focus or bucket, instead of stuffing all synonyms into one query.\n"
        "- Use natural phrasings like: 'ex-Company', 'former Company', 'previously at Company', 'left Company', 'was at Company', 'Company alumni'.\n"
        "- Expand the `Role` field to real-world job title synonyms (e.g., engineer → SWE, SDE, developer, software engineer, backend engineer).\n"
        "- If the role is vague (e.g., 'growth'), cover possible actual titles ('growth lead', 'performance marketer', 'growth PM').\n"
        "- Include optional seniority, keywords, and locations in a realistic way.\n"
        "- Exclude unwanted profiles by using terms like -\"intern\" or -\"contractor\" where needed.\n"
        "- Vary the structure and phrasing across all 9 queries to maximize result diversity.\n"
        "- Avoid over-nesting parentheses and using intitle: or other restrictive operators.\n"
        "- The quit window (e.g., 'past 2 years') is a preference for recency, but do NOT filter out results just because the quit date is ambiguous or missing.\n"
        "- Never return zero queries. Always maximize coverage.\n\n"
        "---\n\n"
        "### Input Variables:\n"
        "- Company: [Insert company name]\n"
        "- Role: [Insert one or more functional roles]\n"
        "- Seniority: [Optional; e.g., junior, senior, lead, staff]\n"
        "- Quit Window: [Human-readable; e.g., 'past 2 years', 'left in 2024']\n"
        "- Location: [Optional geography; e.g., United States, India, remote]\n"
        "- Include Keywords: [Any topical focus; e.g., 'AI', 'fintech', 'payments']\n"
        "- Exclude Keywords: [Optional; e.g., 'intern', 'contractor']\n\n"
        "---\n\n"
        "### Output Format:\n"
        "Return exactly 9 queries, one per line, with NO explanation or extra characters. Each should be a valid and clean Google query that is likely to return relevant LinkedIn results.\n\n"
        "Use smart variations across the 9 queries to improve coverage. Ensure each line is standalone.\n"
    )

    user_json = {
        "Company": params.company,
        "Role": ";".join(expanded_roles) if expanded_roles else "",
        "Seniority": params.seniority,
        "Quit Window": params.quitWindow,
        "Location": params.geography,
        "Include Keywords": params.includeKeywords,
        "Exclude Keywords": params.excludeKeywords,
    }
    user_prompt = json.dumps(user_json, ensure_ascii=False)
    prompt = user_prompt

    queries = []
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is not set. Cannot generate queries.")
    else:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=1800
            )
            content = None
            if isinstance(response, dict) and "choices" in response and isinstance(response["choices"], list):
                first_choice = response["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice and "content" in first_choice["message"]:
                    content = first_choice["message"]["content"].strip()
            if content:
                queries = list(dict.fromkeys([line.strip() for line in content.splitlines() if line.strip()]))
        except Exception as e:
            print(f"[ERROR] Error generating queries with OpenAI: {e}")
            queries = []
    # --- Fallback if OpenAI fails or returns too few queries ---
    if not queries or len(queries) < 9:
        company_or = " OR ".join([f'\"{v}\"' for v in company_variants])
        roles_or = " OR ".join([f'\"{role}\"' for role in expanded_roles]) if expanded_roles else f'\"{params.seniority}\"'
        keywords_or = " OR ".join([f'\"{kw}\"' for kw in params.includeKeywords]) if params.includeKeywords else ""
        exclude_terms = " OR ".join([f'\"{kw}\"' for kw in params.excludeKeywords]) if params.excludeKeywords else ""
        fallback = [
            f'site:linkedin.com/in ("ex-" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("former" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("previously at" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("left" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("was at" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("{params.company} alumni") AND ({roles_or})',
            f'site:linkedin.com/in ("resigned from" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("departed" {company_or}) AND ({roles_or})',
            f'site:linkedin.com/in ("quit" {company_or}) AND ({roles_or})',
        ]
        if keywords_or:
            fallback = [f"{q} AND ({keywords_or})" for q in fallback]
        if exclude_terms:
            fallback = [f"{q} AND -({exclude_terms})" for q in fallback]
        if after and before:
            fallback = [f"{q} after:{after} before:{before}" for q in fallback]
        queries = fallback
    # --- Ensure exactly 9 queries ---
    queries = [q for q in queries if q][:9]
    while len(queries) < 9:
        queries.append(queries[-1] if queries else "site:linkedin.com/in")
    return queries, prompt

def search_google_custom(query: str, num_results: int = 15, quit_window: str = "") -> List[dict]:
    """Enhanced search using Google Custom Search API with more results and better filtering"""
    cache_key = f"{query}|{num_results}|{quit_window}"
    if cache_key in google_search_cache:
        return google_search_cache[cache_key]
    if not GOOGLE_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return []  # No mock data, just return empty
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": query,
            "num": min(num_results, 10),  # Google CSE max is 10 per request
            "sort": "date:r:20200101:20241231"  # Sort by date, recent first
        }
        # Add dateRestrict if quit_window is set
        if quit_window:
            quit_window = quit_window.lower().strip()
            if "month" in quit_window:
                num = int(''.join(filter(str.isdigit, quit_window)))
                params["dateRestrict"] = f"m[{num}]"
            elif "year" in quit_window:
                num = int(''.join(filter(str.isdigit, quit_window)))
                params["dateRestrict"] = f"y[{num}]"
        
        all_results = []
        # Make multiple requests to get more results
        for start_index in range(1, min(num_results + 1, 21), 10):
            params["start"] = start_index
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", []):
                all_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "displayLink": item.get("displayLink", ""),
                    "pagemap": item.get("pagemap", {})
                })
            
            # Break if we got fewer results than requested (end of results)
            if len(data.get("items", [])) < 10:
                break
        
        google_search_cache[cache_key] = all_results
        return all_results
    except Exception as e:
        print(f"Error searching Google: {e}")
        return []

def mock_search_results(query: str) -> List[dict]:
    """Mock search results for development/demo (now disabled)"""
    return []

def calculate_relevance_score(result: dict, search_params: SearchParams, query: str) -> float:
    """Calculate enhanced relevance score for a search result"""
    score = 5.0  # Base score
    
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    content = f"{title} {snippet}"
    
    # Company mention bonus (enhanced)
    company_lower = search_params.company.lower()
    if company_lower in content:
        score += 3.0
        # Bonus for exact company name match
        if f"ex-{company_lower}" in content or f"former {company_lower}" in content:
            score += 2.0
    
    # Enhanced role relevance with fuzzy matching
    if search_params.roles:
        roles = [role.strip().lower() for role in search_params.roles]
        role_matches = 0
        for role in roles:
            # Exact match
            if role in content:
                role_matches += 2.0
            # Partial match (for abbreviations)
            elif any(word in content for word in role.split()):
                role_matches += 1.0
        score += role_matches
    
    # Enhanced seniority indicators
    seniority_terms = ["senior", "staff", "principal", "lead", "manager", "director", "vp", "head of", "chief"]
    seniority_matches = sum(1 for term in seniority_terms if term in content)
    score += seniority_matches * 1.0
    
    # Enhanced departure indicators with confidence scoring
    departure_indicators = {
        "ex-": 3.0,
        "former": 2.5,
        "left": 2.0,
        "departed": 2.0,
        "quit": 2.5,
        "resigned": 2.5,
        "no longer": 2.0,
        "previously": 1.5,
        "formerly": 2.0,
        "stepped down": 2.0,
        "retired": 1.5,
        "moved on": 1.5,
        "transitioned": 1.5
    }
    
    for indicator, weight in departure_indicators.items():
        if indicator in content:
            score += weight
            break  # Only count the strongest indicator
    
    # Enhanced keyword matching
    if search_params.includeKeywords:
        include_terms = [term.strip().lower() for term in search_params.includeKeywords]
        include_matches = sum(1 for term in include_terms if term in content)
        score += include_matches * 1.2
    
    # Enhanced exclude keywords penalty
    if search_params.excludeKeywords:
        exclude_terms = [term.strip().lower() for term in search_params.excludeKeywords]
        exclude_matches = sum(1 for term in exclude_terms if term in content)
        score -= exclude_matches * 2.0  # Stronger penalty
    
    # Enhanced domain authority bonus
    domain_authority = {
        "linkedin.com": 4.0,
        "twitter.com": 2.5,
        "x.com": 2.5,
        "github.com": 2.0,
        "medium.com": 1.5,
        "techcrunch.com": 1.0,
        "angel.co": 1.5,
        "wellfound.com": 1.5
    }
    
    display_link = result.get("displayLink", "").lower()
    for domain, bonus in domain_authority.items():
        if domain in display_link:
            score += bonus
            break
    
    # Recency bonus (if date is recent)
    date_match = re.search(r'20(2[3-4]|1[9-9])', content)  # 2019-2024
    if date_match:
        year = int(date_match.group())
        if year >= 2023:
            score += 2.0
        elif year >= 2021:
            score += 1.0
    
    # Profile completeness bonus
    if "linkedin.com/in/" in result.get("link", ""):
        if len(snippet) > 200:  # Longer snippets often indicate more complete profiles
            score += 1.0
    
    return max(0, min(10, score))  # Clamp between 0-10

def extract_date_from_text(text: str) -> str:
    """Try to extract a date from a text snippet. Return 'N/A' if not found."""
    if not text:
        return "N/A"
    # Look for common date patterns
    patterns = [
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ .,-]*\d{4}',  # e.g. March 2024
        r'\d{1,2}/\d{1,2}/\d{4}',  # e.g. 01/15/2024
        r'\d{4}-\d{2}-\d{2}',      # e.g. 2024-01-15
        r'\d{4}'                    # e.g. 2024
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return "N/A"

def extract_all_end_dates(text: str):
    """Extract all possible end dates (year, month+year, range) from LinkedIn snippet/title."""
    # Patterns: "2019 – 2024", "2018–2023", "until 2024", "left in March 2024", "until March 2024", "ex-Google until 2023"
    patterns = [
        r'(\d{4})\s*[–-]\s*(\d{4})',  # e.g. 2019–2024
        r'until (\w+ \d{4})',           # e.g. until March 2024
        r'left in (\w+ \d{4})',         # e.g. left in March 2024
        r'until (\d{4})',                # e.g. until 2024
        r'(\w+ \d{4})',                 # e.g. March 2024
        r'(\d{4})'                       # fallback: any year
    ]
    found = []
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            if isinstance(match, tuple):
                # For range, take the end year
                found.append(match[-1])
            else:
                found.append(match)
    return found

def extract_all_transitions(text: str, company: str):
    """Enhanced transition extraction with better pattern matching and date parsing"""
    transitions = []
    
    # Enhanced experience block pattern for LinkedIn-style date ranges
    exp_block_patterns = [
        # Standard LinkedIn format: "Company Name • Jan 2020 - Dec 2023"
        re.compile(rf"{re.escape(company)}.*?(\w{{3,9}} \d{{4}})\s*[–-]\s*(\w{{3,9}} \d{{4}}|Present|present|Current|current|\d{{4}})", re.IGNORECASE | re.DOTALL),
        # Alternative format: "Company Name • 2020 - 2023"
        re.compile(rf"{re.escape(company)}.*?(\d{{4}})\s*[–-]\s*(\d{{4}}|Present|present|Current|current)", re.IGNORECASE | re.DOTALL),
        # With bullet points: "• Company Name • Jan 2020 - Dec 2023"
        re.compile(rf"•\s*{re.escape(company)}.*?(\w{{3,9}} \d{{4}})\s*[–-]\s*(\w{{3,9}} \d{{4}}|Present|present|Current|current|\d{{4}})", re.IGNORECASE | re.DOTALL),
    ]
    
    for pattern in exp_block_patterns:
        for match in pattern.finditer(text):
            start_str, end_str = match.group(1), match.group(2)
            # Only consider if end_str is not 'Present' or 'Current'
            if end_str.lower() not in ['present', 'current']:
                try:
                    dt = parse_end_date(end_str)
                    if dt:
                        transitions.append({
                            "type": "experience_end", 
                            "value": end_str, 
                            "full": match.group(0), 
                            "date": dt,
                            "confidence": "high"
                        })
                except Exception:
                    continue
    
    # Enhanced quit/transition patterns with confidence scoring
    quit_patterns = [
        # High confidence patterns
        (rf'left {company} in (\w+ \d{{4}})', 'left_in_month_year', 'high'),
        (rf'left {company} in (\d{{4}})', 'left_in_year', 'high'),
        (rf'quit {company} in (\w+ \d{{4}})', 'quit_in_month_year', 'high'),
        (rf'quit {company} in (\d{{4}})', 'quit_in_year', 'high'),
        (rf'resigned from {company} in (\w+ \d{{4}})', 'resigned_in_month_year', 'high'),
        (rf'resigned from {company} in (\d{{4}})', 'resigned_in_year', 'high'),
        (rf'stepped down from {company} in (\w+ \d{{4}})', 'stepped_down_in_month_year', 'high'),
        (rf'stepped down from {company} in (\d{{4}})', 'stepped_down_in_year', 'high'),
        
        # Medium confidence patterns
        (rf'ex[-\s\"\']*{company}', 'ex', 'medium'),
        (rf'former {company}', 'former', 'medium'),
        (rf'formerly at {company}', 'formerly', 'medium'),
        (rf'previously at {company}', 'previously', 'medium'),
        (rf'departed {company}', 'departed', 'medium'),
        (rf'no longer at {company}', 'no_longer', 'medium'),
        (rf'ended at {company}', 'ended', 'medium'),
        (rf'retired from {company}', 'retired', 'medium'),
        (rf'moved on from {company}', 'moved_on', 'medium'),
        (rf'transitioned from {company}', 'transitioned', 'medium'),
        
        # Date-specific patterns
        (rf'until (\w+ \d{{4}})', 'until_month_year', 'high'),
        (rf'until (\d{{4}})', 'until_year', 'high'),
        (rf'left in (\w+ \d{{4}})', 'left_in_month_year', 'high'),
        (rf'left in (\d{{4}})', 'left_in_year', 'high'),
        
        # New company indicators (high confidence for quit inference)
        (r'joined ([A-Za-z0-9&.,\- ]+) in (\w+ \d{4})', 'joined_new_month_year', 'high'),
        (r'joined ([A-Za-z0-9&.,\- ]+) in (\d{4})', 'joined_new_year', 'high'),
        (r'started at ([A-Za-z0-9&.,\- ]+) in (\w+ \d{4})', 'started_new_month_year', 'high'),
        (r'started at ([A-Za-z0-9&.,\- ]+) in (\d{4})', 'started_new_year', 'high'),
        (r'now at ([A-Za-z0-9&.,\- ]+)', 'now_at', 'medium'),
        
        # Low confidence patterns (fallbacks)
        (rf'left {company}', 'left', 'low'),
        (rf'quit {company}', 'quit', 'low'),
        (rf'resigned from {company}', 'resigned', 'low'),
        (r'(\w+ \d{4})', 'month_year', 'low'),
        (r'(\d{4})', 'year', 'low'),
    ]
    
    for pattern, label, confidence in quit_patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            if isinstance(match, tuple):
                value = match[-1]  # Take the last element for date
                transitions.append({
                    'type': label, 
                    'value': value, 
                    'full': match, 
                    'confidence': confidence
                })
            else:
                transitions.append({
                    'type': label, 
                    'value': match, 
                    'full': match, 
                    'confidence': confidence
                })
    
    # Sort by confidence (high > medium > low) and then by date
    confidence_order = {'high': 3, 'medium': 2, 'low': 1}
    transitions.sort(key=lambda x: (
        confidence_order.get(x.get('confidence', 'low'), 0),
        x.get("date", datetime.min)
    ), reverse=True)
    
    return transitions

# In infer_quit_date_from_transitions, if no explicit quit date is found, use most recent experience end date as proxy if present

def infer_quit_date_from_transitions(transitions, company):
    """Infer the most likely quit date from transitions, prioritizing explicit left/quit, then join at new company, then experience end date, then education end date, then fallback."""
    # Priority: explicit left/quit > until > joined new company > experience end date > education end date > fallback
    for t in transitions:
        if t['type'] in ['left_in_month_year', 'left_in_year', 'until_month_year', 'until_year']:
            return t['value']
    # If joined new company, use that date as quit date
    for t in transitions:
        if t['type'] in ['joined_new_month_year', 'joined_new_year', 'started_new_month_year', 'started_new_year']:
            return t['value']
    # If experience end date is present, use as proxy
    for t in transitions:
        if t['type'] in ['experience_end']:
            return t['value']
    # If education end date is present, use as proxy
    for t in transitions:
        if t['type'] in ['education_graduation_month_year', 'education_graduation_year', 'education_range', 'education_until_month_year', 'education_until_year', 'education_month_year', 'education_year']:
            return t['value']
    # Fallback: any month-year or year
    for t in transitions:
        if t['type'] in ['month_year', 'year']:
            return t['value']
    return None

def parse_end_date(date_str: str):
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    # Try 'Mon YYYY' (e.g., 'Jul 2020')
    try:
        return datetime.strptime(date_str, "%b %Y")
    except Exception:
        pass
    # Try 'Month YYYY' (e.g., 'January 2020')
    try:
        return datetime.strptime(date_str, "%B %Y")
    except Exception:
        pass
    # Try 'YYYY'
    try:
        return datetime.strptime(date_str, "%Y")
    except Exception:
        pass
    # Try extracting year from 'in 2024', 'to 2026', etc.
    import re
    m = re.search(r"(\d{4})", date_str)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y")
        except Exception:
            pass
    # If all fails, return None
    return None

def best_quit_date(dates: list):
    """Return the most recent valid quit date from a list of date strings."""
    parsed = [parse_end_date(d) for d in dates]
    parsed = [d for d in parsed if d]
    if not parsed:
        return None
    return max(parsed)

def is_within_quit_window_smart(quit_date, quit_window: str):
    if not quit_window or quit_window == "0":
        return True  # No filtering if quit_window is 0 or empty
    quit_window = quit_window.lower().strip()
    now = datetime.utcnow().date()
    if "month" in quit_window:
        num = int(''.join(filter(str.isdigit, quit_window)))
        window_days = 30 * num
    elif "year" in quit_window:
        num = int(''.join(filter(str.isdigit, quit_window)))
        window_days = 365 * num
    else:
        return None
    if isinstance(quit_date, datetime):
        quit_date = quit_date.date()
    return (now - quit_date).days <= window_days

def get_source_from_display_link(display_link: str):
    dl = display_link.lower()
    if "linkedin.com" in dl:
        return "linkedin"
    if "twitter.com" in dl or "x.com" in dl:
        return "twitter"
    if "github.com" in dl:
        return "github"
    if "medium.com" in dl:
        return "medium"
    if "wellfound.com" in dl or "angel.co" in dl:
        return "wellfound"
    if "substack.com" in dl or "blogspot.com" in dl or "wordpress.com" in dl:
        return "blog"
    return "other"

def chatgpt_rerank_and_extract(snippets: list[str], company: str, quit_window: str) -> Optional[list[dict]]:
    """Enhanced ChatGPT extraction with better prompt engineering for higher quality results"""
    if not OPENAI_API_KEY or not snippets:
        return [{"score": 0.0, "name": None, "role": None, "quit_status": None, "quit_date": None, "company": None, "linkedin_url": None, "evidence_phrase": None, "confidence": 0, "rationale": "No LLM", "rawGoogleResult": True} for _ in snippets]

    # Create structured prompts with enhanced extraction
    snippets_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(snippets)])
    
    prompts = [
        # Enhanced Prompt 1: Comprehensive extraction with better scoring
        f"""Extract detailed ex-employee information from each LinkedIn snippet. Return a JSON array with exactly {len(snippets)} objects.

For each snippet, extract:
- name: Full name (First Last) if found, null if not
- role: Job title/role if found, null if not  
- quit_status: "ex-employee" if they left, "current" if still there, "ambiguous" if unclear
- quit_date: Date they left (YYYY-MM-DD format), null if not found
- company: "{company}"
- linkedin_url: LinkedIn URL if in snippet, null if not
- evidence_phrase: Key phrase showing they left, null if not found
- confidence: 0-10 score based on evidence strength:
  * 10: Explicit "ex-{company}", "left {company} in [date]"
  * 8-9: "former {company}", "quit {company}", "resigned from {company}"
  * 6-7: "previously at {company}", "departed {company}"
  * 4-5: "joined [new company]" after {company} mention
  * 2-3: Ambiguous or weak indicators
  * 0-1: No clear evidence
- rationale: Brief explanation of confidence score and reasoning

Key indicators to look for:
- "ex-{company}", "former {company}", "left {company}"
- "quit {company}", "resigned from {company}", "departed {company}"
- "until [date]", "left in [date]", "ex-{company} until [date]"
- "joined [new company]", "now at [new company]", "started at [new company]"
- "stepped down from {company}", "retired from {company}"

Return ONLY valid JSON array. No commentary.

Snippets:
{snippets_text}""",

        # Enhanced Prompt 2: Focus on quit detection with date extraction
        f"""For each snippet, determine if someone left {company} and extract detailed info. Return JSON array with {len(snippets)} objects.

Each object: {{"name": "...", "role": "...", "quit_status": "...", "quit_date": "...", "company": "{company}", "linkedin_url": "...", "evidence_phrase": "...", "confidence": 0-10, "rationale": "..."}}

Confidence scoring:
- 10: "ex-{company}" with date
- 9: "left {company} in [date]" or "quit {company} in [date]"
- 8: "former {company}" or "resigned from {company}"
- 7: "previously at {company}" or "departed {company}"
- 6: "joined [new company] in [date]" after {company}
- 5: "now at [new company]" after {company}
- 4: "stepped down" or "retired" from {company}
- 3: Ambiguous departure indicators
- 2: Weak or unclear evidence
- 1: No clear departure evidence
- 0: Still at {company} or no relevant info

Look for date patterns: "2024", "2023", "Dec 2023", "January 2024", "until 2024"

Snippets:
{snippets_text}""",

        # Enhanced Prompt 3: Role and seniority focused
        f"""Extract ex-employee data with focus on roles and seniority. Return JSON array with {len(snippets)} objects.

Format: [{{"name": "...", "role": "...", "quit_status": "...", "quit_date": "...", "company": "{company}", "linkedin_url": "...", "evidence_phrase": "...", "confidence": 0-10, "rationale": "..."}}]

Role extraction: Look for job titles like "Software Engineer", "Product Manager", "Data Scientist", "Designer", "Engineering Manager", "Tech Lead", "Principal Engineer", "Director", "VP"

Seniority indicators: "Senior", "Staff", "Principal", "Lead", "Manager", "Director", "VP", "Head of", "Chief"

Quit indicators: "ex-{company}", "former {company}", "left {company}", "quit {company}", "resigned from {company}", "until [date]", "joined [new company]"

Snippets:
{snippets_text}"""
    ]
    
    for idx, prompt in enumerate(prompts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Zero temperature for most consistent output
                max_tokens=4000
            )
            content = None
            if isinstance(response, dict) and "choices" in response and isinstance(response["choices"], list):
                first_choice = response["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice and "content" in first_choice["message"]:
                    content = first_choice["message"]["content"].strip()
            
            if not content:
                print(f"[DEBUG] LLM output is empty for prompt {idx+1}. Trying next prompt.")
                continue
                
            # Try to extract JSON from the response
            try:
                # Clean the content - remove any markdown formatting
                cleaned_content = content.strip()
                if cleaned_content.startswith('```json'):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith('```'):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()
                
                # Look for JSON array in the content
                json_start = cleaned_content.find('[')
                json_end = cleaned_content.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = cleaned_content[json_start:json_end]
                    # Try to fix common JSON issues
                    json_content = json_content.replace('\n', ' ').replace('\r', ' ')
                    # Remove any trailing commas before closing brackets
                    json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
                    results = json.loads(json_content)
                else:
                    results = json.loads(cleaned_content)
                
                if isinstance(results, list) and len(results) == len(snippets):
                    print(f"[DEBUG] LLM extraction succeeded with prompt {idx+1}.")
                    return results
                else:
                    print(f"[DEBUG] LLM output is not a valid array or wrong length for prompt {idx+1}. Length: {len(results) if isinstance(results, list) else 'not list'}. Trying next prompt.")
                    continue
            except json.JSONDecodeError as e:
                print(f"[DEBUG] LLM output is not valid JSON for prompt {idx+1}: {e}")
                # Try to fix the JSON manually
                try:
                    # Extract just the array part and try to fix common issues
                    array_start = content.find('[')
                    array_end = content.rfind(']') + 1
                    if array_start != -1 and array_end > array_start:
                        json_part = content[array_start:array_end]
                        # Remove newlines and fix trailing commas
                        json_part = re.sub(r',(\s*[}\]])', r'\1', json_part)
                        json_part = json_part.replace('\n', ' ').replace('\r', ' ')
                        # Try to parse again
                        results = json.loads(json_part)
                        if isinstance(results, list) and len(results) == len(snippets):
                            print(f"[DEBUG] LLM extraction succeeded with manual JSON fix for prompt {idx+1}.")
                            return results
                except:
                    pass
                print(f"[DEBUG] Content preview: {content[:300]}...")
                continue
        except Exception as e:
            print(f"ChatGPT rerank error (prompt {idx+1}): {e}")
            continue
    
    print("[DEBUG] All LLM prompts failed. Returning raw Google results.")
    return None

# TODO: Multi-search aggregation placeholder
# In the search endpoint, before LLM extraction, aggregate results from Google, Bing, Brave, etc. Deduplicate by URL/snippet. Pass merged set to LLM.

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vc-analyst-backend")

# Setup logger
LOG_FILE = "vc_analyst_audit.log"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vc-analyst-search")
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(file_handler)

# Helper to read last N lines from log file
import os
def get_last_log_lines(n=10):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-n:]

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    # Audit log
    with open(LOG_FILE, "a") as f:
        f.write(f"[EXCEPTION] {exc}\n")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)}
    )

# Check required env vars at startup
missing_env = []
for var in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"]:
    if not os.getenv(var):
        missing_env.append(var)
if missing_env:
    logger.error(f"Missing required environment variables: {', '.join(missing_env)}")
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_env)}")

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

def extract_candidate_name(snippet: str, title: str) -> str:
    """Try to extract a candidate's name from the snippet or title."""
    # Heuristic: Look for capitalized words at the start of the snippet or title
    for text in [title, snippet]:
        match = re.match(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", text)
        if match:
            return match.group(1)
    return "N/A"

def convert_datetimes(obj):
    if isinstance(obj, dict):
        return {k: convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d")
    else:
        return obj

# --- Helper: Split roles on spaces, commas, semicolons ---
def split_roles(roles):
    if not roles:
        return []
    if isinstance(roles, str):
        roles = [roles]
    split = []
    for r in roles:
        split += [s.strip() for s in re.split(r'[ ,;]+', r) if s.strip()]
    return list(set(split))

def split_terms(field):
    """Split a string field on spaces, commas, or semicolons, and deduplicate."""
    if not field:
        return []
    if isinstance(field, list):
        field = ' '.join(field)
    return list(set([s.strip() for s in re.split(r'[ ,;]+', field) if s.strip()]))

# --- Helper: Fuzzy match role ---
def is_similar_role(query, role, threshold=60):  # Lowered threshold for less strict matching
    if not fuzz:
        return query.lower() in role.lower()
    return fuzz.partial_ratio(query.lower(), role.lower()) >= threshold

# --- Stub: Zuabacorp/ZaubaCorp API integration ---
def search_zuabacorp(person_name):
    # TODO: Integrate with ZaubaCorp or similar company registration API
    # Example: requests.get('https://api.zaubacorp.com/search', params={'name': person_name, 'api_key': ...})
    # Return list of companies registered by this person
    return []

@app.post("/generate-queries")
async def generate_queries_endpoint(params: SearchParams):
    """Generate X-ray search queries"""
    reset_query_count_if_needed()
    global query_count
    with lock:
        if query_count >= MAX_QUERIES_PER_DAY:
            logger.warning("Reached daily free limit. Try again tomorrow.")
            return JSONResponse(status_code=429, content={"error": "Reached daily free limit. Try again tomorrow."})
        query_count += 1
    # Explicit OpenAI API key check
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is missing. Set OPENAI_API_KEY in your .env file.")
        return JSONResponse(status_code=500, content={"error": "OpenAI API key is missing. Set OPENAI_API_KEY in your .env file."})
    # Log incoming payload
    logger.info(f"/generate-queries payload: {params}")
    with open(LOG_FILE, "a") as f:
        f.write(f"[GENERATE-QUERIES] {params}\n")
    # Validate types
    if not isinstance(params.roles, list):
        params.roles = [params.roles] if params.roles else []
    if not isinstance(params.includeKeywords, list):
        params.includeKeywords = [params.includeKeywords] if params.includeKeywords else []
    if not isinstance(params.excludeKeywords, list):
        params.excludeKeywords = [params.excludeKeywords] if params.excludeKeywords else []
    try:
        queries, prompt = generate_xray_queries(params)
        logger.info(f"Generated queries: {queries}")
        with open(LOG_FILE, "a") as f:
            f.write(f"[GENERATED-QUERIES] {queries}\n")
        return JSONResponse(content={"queries": queries, "prompt": prompt})
    except Exception as e:
        logger.error(f"Failed to generate queries: {str(e)}")
        with open(LOG_FILE, "a") as f:
            f.write(f"[ERROR] Failed to generate queries: {str(e)}\n")
        return JSONResponse(status_code=500, content={"error": f"Failed to generate queries: {str(e)}"})

@app.post("/search")
async def search_endpoint(request: QueryRequest):
    """Execute search queries and return ranked results"""
    reset_query_count_if_needed()
    global query_count
    with lock:
        if query_count >= MAX_QUERIES_PER_DAY:
            logger.warning("Reached daily free limit. Try again tomorrow.")
            return JSONResponse(status_code=429, content={"error": "Reached daily free limit. Try again tomorrow."})
        query_count += 1
    try:
        all_results = []
        seen_urls = set()
        logger.info(f"Queries: {request.queries}")
        with open(LOG_FILE, "a") as f:
            f.write(f"[SEARCH] {request.queries}\n")
        # --- Broaden role search: split and fuzzy match ---
        roles = split_roles(request.searchParams.roles)
        if not roles and request.searchParams.seniority:
            roles = split_roles([request.searchParams.seniority])
        for i, query in enumerate(request.queries):
            google_results = search_google_custom(query, quit_window=str(request.searchParams.quitWindow or '')) if GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID else []
            combined_results = google_results
            logger.info(f"Google results for query {i+1}: {len(combined_results)}")
            for result in combined_results:
                url = result.get("link", "")
                if url in seen_urls or not url:
                    continue
                seen_urls.add(url)
                snippet = result.get("snippet", "")
                title = result.get("title", "")
                display_link = result.get("displayLink", "")
                # --- Fuzzy/partial match for roles ---
                content = (snippet + " " + title).lower()
                # Use fuzzy/expanded role matching, not strict equality
                if roles:
                    if not any(is_similar_role(role, content) for role in roles):
                        continue
                # Enhanced filtering for better quality results
                # Filter out non-profile links more aggressively
                exclude_patterns = [
                    "/jobs", "/careers", "/company", "/news", "/updates", "/about", "/press", 
                    "/blog", "/events", "/groups", "/pages", "/stories", "/media", "/services", 
                    "/solutions", "/products", "/webinar", "/podcast", "/award", "/recognition", 
                    "/announcement", "/help", "/support", "/contact", "/legal", "/privacy", 
                    "/terms", "/cookie", "/sitemap", "/robots", "/ads", "/advertising"
                ]
                if any(pattern in url.lower() for pattern in exclude_patterns):
                    continue
                # Enhanced profile link detection
                profile_domains = [
                    "linkedin.com/in/",
                    "twitter.com/",
                    "x.com/",
                    "github.com/",
                    "angel.co/",
                    "wellfound.com/",
                    "medium.com/@",
                    "substack.com/@",
                    "dev.to/",
                    "hashnode.dev/",
                    "personal-website.com",
                    ".me/",
                    ".io/"
                ]
                is_profile = any(domain in url.lower() for domain in profile_domains)
                if not is_profile:
                    continue
                # --- Company match is non-negotiable: only exclude if company is NOT clearly mentioned as a past employer ---
                content_lower = (snippet + " " + title).lower()
                company = request.searchParams.company.lower()
                has_company_mention = False
                # Check for direct company mention patterns
                company_mention_patterns = [
                    f"ex-{company}", f"ex {company}", f"former {company}",
                    f"formerly at {company}", f"previously at {company}", f"past {company}",
                    f"left {company}", f"departed {company}", f"quit {company}",
                    f"resigned from {company}", f"no longer at {company}",
                    f"ended at {company}", f"retired from {company}",
                    f"stepped down from {company}", f"moved on from {company}",
                    f"transitioned from {company}", f"leaving {company}",
                    f"until {company}", f"at {company} until"
                ]
                if any(pattern in content_lower for pattern in company_mention_patterns):
                    has_company_mention = True
                # Also check for company variants
                company_variants = enhance_company_variants(request.searchParams.company)
                for variant in company_variants:
                    variant_lower = variant.lower()
                    variant_patterns = [
                        f"ex-{variant_lower}", f"ex {variant_lower}", f"former {variant_lower}",
                        f"formerly at {variant_lower}", f"previously at {variant_lower}",
                        f"left {variant_lower}", f"departed {variant_lower}",
                        f"resigned from {variant_lower}", f"no longer at {variant_lower}"
                    ]
                    if any(pattern in content_lower for pattern in variant_patterns):
                        has_company_mention = True
                        break
                if not has_company_mention:
                    continue
                # Do NOT filter out for ambiguous/missing quit date. Always include if company and role are similarish.
                # Set safe defaults for removed fields
                relevance_score = None
                source = None
                confidence = 1
                transitions = None
                # Extract name and quit date
                name = extract_candidate_name(snippet, title)
                quit_date = None
                if "quit_date" in locals() and quit_date:
                    quit_date_str = quit_date.strftime("%Y-%m-%d")
                else:
                    quit_date_str = None
                # Always include all expected fields, even if null/empty
                all_results.append({
                    "rank": len(all_results) + 1,
                    "name": name,
                    "snippet": snippet,
                    "domain": display_link,
                    "link": url,
                    "relevanceScore": relevance_score,
                    "source": source,
                    "quitDate": quit_date_str,
                    "quitConfidence": confidence,
                    "transitions": transitions,
                    # LLM fields (default to None)
                    "llmName": None,
                    "llmRole": None,
                    "llmQuitStatus": None,
                    "llmQuitDate": None,
                    "llmCompany": None,
                    "llmLinkedinUrl": None,
                    "llmEvidencePhrase": None,
                    "llmConfidence": None,
                    "llmRationale": None,
                    "llmRawGoogleResult": None,
                    "rawGoogleResult": True
                })
        # Defensive: If no results, return empty
        if not all_results:
            logger.info("No Google results after filtering.")
            with open(LOG_FILE, "a") as f:
                f.write("[NO-RESULTS] No Google results after filtering.\n")
            return {"results": []}
        # --- LLM rerank and extraction for ALL results (if OpenAI available) ---
        if OPENAI_API_KEY and len(all_results) > 0:
            snippets = [r["snippet"] for r in all_results]
            company = request.searchParams.company
            quit_window = str(request.searchParams.quitWindow or '')
            llm_results = chatgpt_rerank_and_extract(snippets, company, quit_window)
            logger.info(f"LLM output length: {len(llm_results) if llm_results else 0}")
            if llm_results is not None:
                for i, r in enumerate(all_results):
                    llm = llm_results[i]
                    r["llmName"] = llm.get("name") if llm.get("name") else None
                    r["llmRole"] = llm.get("role") if llm.get("role") else None
                    r["llmQuitStatus"] = llm.get("quit_status") if llm.get("quit_status") else None
                    r["llmQuitDate"] = llm.get("quit_date") if llm.get("quit_date") else None
                    r["llmCompany"] = llm.get("company") if llm.get("company") else None
                    r["llmLinkedinUrl"] = llm.get("linkedin_url") if llm.get("linkedin_url") else None
                    r["llmEvidencePhrase"] = llm.get("evidence_phrase") if llm.get("evidence_phrase") else None
                    r["llmConfidence"] = llm.get("confidence") if llm.get("confidence") else None
                    r["llmRationale"] = llm.get("rationale") if llm.get("rationale") else None
                    r["llmRawGoogleResult"] = llm.get("rawGoogleResult", False)
                    r["rawGoogleResult"] = False
        # Enhanced result ranking and sorting
        # Sort by multiple criteria: source priority, LLM confidence, relevance score, quit confidence
        def sort_key(result):
            # Source priority: LinkedIn > Twitter > GitHub > others
            source_priority = {"linkedin": 4, "twitter": 3, "github": 2, "medium": 1}
            source_score = source_priority.get(result.get("source", ""), 0)
            
            # LLM confidence (if available)
            llm_confidence = result.get("llmConfidence", 0) or 0
            
            # Relevance score
            relevance_score = result.get("relevanceScore", 0) or 0
            
            # Quit confidence
            quit_confidence = result.get("quitConfidence", 0) or 0
            
            # Recency bonus (if quit date is recent)
            recency_bonus = 0
            quit_date = result.get("quitDate")
            if quit_date:
                try:
                    date_obj = datetime.strptime(quit_date, "%Y-%m-%d")
                    days_ago = (datetime.now() - date_obj).days
                    if days_ago <= 365:  # Within 1 year
                        recency_bonus = 2.0
                    elif days_ago <= 730:  # Within 2 years
                        recency_bonus = 1.0
                except:
                    pass
            
            return (
                source_score,
                llm_confidence,
                relevance_score + recency_bonus,
                quit_confidence
            )
        
        # Sort results by the enhanced criteria
        all_results.sort(key=sort_key, reverse=True)
        
        # Always ensure all fields are present after sorting
        for i, result in enumerate(all_results):
            result["rank"] = i + 1
            for field in ["llmName", "llmRole", "llmQuitStatus", "llmQuitDate", "llmCompany", "llmLinkedinUrl", "llmEvidencePhrase", "llmConfidence", "llmRationale", "llmRawGoogleResult", "rawGoogleResult", "quitDate", "quitConfidence", "transitions", "source"]:
                if field not in result:
                    result[field] = None
        
        # Enhanced relevance score calculation
        for r in all_results:
            # Use LLM confidence if available, otherwise use calculated score
            if r["llmConfidence"] is not None and r["llmConfidence"] > 0:
                r["relevanceScore"] = r["llmConfidence"]
            else:
                r["relevanceScore"] = r.get("relevanceScore", 0) or 0
            
            # Add bonus for high confidence quit detection
            if r.get("quitConfidence", 0) >= 3:
                r["relevanceScore"] += 1.0
            
            # Ensure score is within bounds
            r["relevanceScore"] = max(0, min(10, r["relevanceScore"]))
        # Convert all datetime objects to strings before returning
        return JSONResponse(content={"results": convert_datetimes(all_results)})
    except Exception as e:
        logger.error(f"SEARCH ERROR: {e}")
        with open(LOG_FILE, "a") as f:
            f.write(f"[ERROR] SEARCH ERROR: {e}\n")
        traceback.print_exc()
        return JSONResponse(content={"results": [], "error": str(e)})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy"})

@app.get("/debug")
async def debug_endpoint():
    """Debug endpoint to check config and env."""
    return JSONResponse(content={
        "OPENAI_API_KEY_present": bool(OPENAI_API_KEY),
        "GOOGLE_API_KEY_present": bool(GOOGLE_API_KEY),
        "GOOGLE_SEARCH_ENGINE_ID_present": bool(GOOGLE_SEARCH_ENGINE_ID),
        "SERPAPI_KEY_present": bool(SERPAPI_KEY),
        "query_count": query_count,
        "last_reset_date": str(last_reset_date),
        "last_10_log_lines": get_last_log_lines(10),
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
