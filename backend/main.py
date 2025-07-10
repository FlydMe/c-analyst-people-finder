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
    allow_origins=["https://c-analyst-people-finder-1.onrender.com"],  # or ["*"] for testing
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

def generate_xray_queries(params: SearchParams) -> tuple[list[str], str]:
    """Generate Google X-ray search queries using advanced prompt engineering and ChatGPT"""
    # --- New: Encode quit window as after:/before: in every query ---
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
    # --- Use only the elite QueryGen prompt, no example or commentary ---
    system_prompt = (
        "You are QueryGen, an elite AI assistant for venture capital analysts specializing in talent intelligence. Your mission is to generate exactly 7 highly optimized, diverse Google X-ray search queries to identify promising ex-employees who left a target company within a specified timeframe.\n\n"
        "QUERY GENERATION STRATEGY:\n"
        "1. **Primary Focus**: LinkedIn profiles (`site:linkedin.com/in`)\n"
        "2. **Boolean Logic**: Use advanced operators (AND, OR, quotes, -) for precision\n"
        "3. **Quit Language Variations**: Include multiple quit indicators:\n"
        "   - \"ex-Company\", \"ex Company\", \"exCompany\"\n"
        "   - \"formerly at Company\", \"previously at Company\"\n"
        "   - \"left Company\", \"departed Company\", \"quit Company\"\n"
        "   - \"resigned from Company\", \"no longer at Company\"\n"
        "   - \"until [date]\", \"left in [date]\", \"ex-Company until [date]\"\n"
        "4. **Role Variations**: Include different role phrasings:\n"
        "   - Exact roles from input\n"
        "   - Seniority + role combinations\n"
        "   - Common variations (\"Software Engineer\" vs \"SDE\" vs \"Developer\")\n"
        "5. **Geographic Targeting**: Add location filters if specified\n"
        "6. **Keyword Integration**: Seamlessly integrate include/exclude keywords\n"
        "7. **Exclusion Filters**: Always exclude \"intern\", \"junior\", \"freelancer\", \"contractor\"\n\n"
        "QUERY STRUCTURE PATTERNS:\n"
        "- Pattern 1: `site:linkedin.com/in \"[Role]\" AND (\"ex-Company\" OR \"left Company\") AND (\"[keywords]\")`\n"
        "- Pattern 2: `site:linkedin.com/in \"[Seniority] [Role]\" AND \"formerly at Company\" AND \"[location]\"`\n"
        "- Pattern 3: `site:linkedin.com/in \"[Role]\" AND (\"quit Company\" OR \"resigned from Company\") AND \"[keywords]\"`\n"
        "- Pattern 4: `site:linkedin.com/in \"[Role]\" AND \"ex-Company\" AND \"until [year]\"`\n"
        "- Pattern 5: `site:linkedin.com/in \"[Role]\" AND \"Company\" AND (\"left\" OR \"departed\") AND \"[keywords]\"`\n"
        "- Pattern 6: `site:linkedin.com/in \"[Seniority]\" AND \"Company\" AND (\"ex-\" OR \"former\") AND \"[location]\"`\n"
        "- Pattern 7: `site:linkedin.com/in \"[Role]\" AND \"Company\" AND (\"no longer\" OR \"previously\") AND \"[keywords]\"`\n\n"
        "INPUT FIELDS (company_variants is a list of all real variants):\n"
        "- `company_variants`: List of company name variants\n"
        "- `roles`: Array of job titles to search for\n"
        "- `seniority`: Seniority level (Senior, Lead, Principal, etc.)\n"
        "- `quit_window`: Time window (e.g., \"6 months\", \"1 year\")\n"
        "- `geography`: Location focus\n"
        "- `include_keywords`: Skills/topics to include\n"
        "- `exclude_keywords`: Terms to exclude\n\n"
        "OUTPUT: Exactly 7 one-line Google X-ray queries—no commentary, no numbering, no formatting—just the raw query strings ready for Google search."
    )
    user_json = {
        "company_variants": company_variants,
        "roles": params.roles,
        "seniority": params.seniority,
        "quit_window": params.quitWindow,
        "geography": params.geography,
        "include_keywords": params.includeKeywords,
        "exclude_keywords": params.excludeKeywords,
        "domains": params.sources or []
    }
    user_prompt = json.dumps(user_json, ensure_ascii=False)
    prompt = user_prompt
    if OPENAI_API_KEY:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            content = None
            if isinstance(response, dict) and "choices" in response and isinstance(response["choices"], list):
                first_choice = response["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice and "content" in first_choice["message"]:
                    content = first_choice["message"]["content"].strip()
            if not content:
                raise ValueError("No content from OpenAI API")
            queries = list(dict.fromkeys([line.strip() for line in content.splitlines() if line.strip()]))[:7]
            if after and before:
                queries = [f"{q} after:{after} before:{before}" for q in queries]
            if len(queries) < 7:
                raise ValueError("Too few queries generated")
            return queries, prompt
        except Exception as e:
            print(f"Error generating queries: {e}")
            # Fallback: robust hand-crafted queries (7 only)
            company_or = " OR ".join([f'\"{v}\"' for v in company_variants])
            fallback = [
                f'site:linkedin.com/in ({company_or}) AND ("ex" OR "former" OR "left" OR "departed") AND ("{params.roles}" OR "{params.seniority}")',
                f'site:linkedin.com/in ({company_or}) AND ("previously at" OR "until {params.quitWindow}" OR "left in {params.quitWindow}") AND ("{params.roles}" OR "{params.seniority}")',
                f'site:linkedin.com/in ({company_or}) AND ("Senior" OR "Lead") AND ("quit" OR "resigned")',
                f'site:linkedin.com/in ({company_or}) AND ("Backend Developer" OR "Engineer") AND ("recently at" OR "left in 2024")',
                f'site:linkedin.com/in ({company_or}) AND ("SDE" OR "Senior Developer") AND ("resigned from" OR "quit")',
                f'site:linkedin.com/in ({company_or}) AND ("Principal Engineer" OR "Tech Lead") AND ("ex" OR "left")',
                f'site:linkedin.com/in ({company_or}) AND ("Software Engineer" OR "Developer") AND ("formerly at")',
            ]
            if after and before:
                fallback = [f"{q} after:{after} before:{before}" for q in fallback]
            return fallback[:7], prompt
    # Fallback: return robust hand-crafted queries (7 only)
    company_or = " OR ".join([f'\"{v}\"' for v in company_variants])
    fallback = [
        f'site:linkedin.com/in ({company_or}) AND ("ex" OR "former" OR "left" OR "departed") AND ("{params.roles}" OR "{params.seniority}")',
        f'site:linkedin.com/in ({company_or}) AND ("previously at" OR "until {params.quitWindow}" OR "left in {params.quitWindow}") AND ("{params.roles}" OR "{params.seniority}")',
        f'site:linkedin.com/in ({company_or}) AND ("Senior" OR "Lead") AND ("quit" OR "resigned")',
        f'site:linkedin.com/in ({company_or}) AND ("Backend Developer" OR "Engineer") AND ("recently at" OR "left in 2024")',
        f'site:linkedin.com/in ({company_or}) AND ("SDE" OR "Senior Developer") AND ("resigned from" OR "quit")',
        f'site:linkedin.com/in ({company_or}) AND ("Principal Engineer" OR "Tech Lead") AND ("ex" OR "left")',
        f'site:linkedin.com/in ({company_or}) AND ("Software Engineer" OR "Developer") AND ("formerly at")',
    ]
    if after and before:
        fallback = [f"{q} after:{after} before:{before}" for q in fallback]
    return fallback[:7], prompt

def search_google_custom(query: str, num_results: int = 10, quit_window: str = "") -> List[dict]:
    """Search using Google Custom Search API, with dateRestrict if quit_window is set. Uses in-memory cache to reduce rate limit hits."""
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
            "num": num_results
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
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "displayLink": item.get("displayLink", "")
            })
        google_search_cache[cache_key] = results
        return results
    except Exception as e:
        print(f"Error searching Google: {e}")
        return []

def search_serpapi(query: str, num_results: int = 10) -> List[dict]:
    """Search using SerpAPI"""
    if not SERPAPI_KEY:
        return []
    try:
        url = "https://serpapi.com/search"
        params = {
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "q": query,
            "num": num_results
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "displayLink": item.get("displayed_link", "")
            })
        return results
    except Exception as e:
        print(f"Error searching SerpAPI: {e}")
        return []

def mock_search_results(query: str) -> List[dict]:
    """Mock search results for development/demo (now disabled)"""
    return []

def calculate_relevance_score(result: dict, search_params: SearchParams, query: str) -> float:
    """Calculate relevance score for a search result"""
    score = 5.0  # Base score
    
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    content = f"{title} {snippet}"
    
    # Company mention bonus
    if search_params.company.lower() in content:
        score += 2.0
    
    # Role relevance
    if search_params.roles:
        roles = [role.strip().lower() for role in search_params.roles]
        role_matches = sum(1 for role in roles if role in content)
        score += role_matches * 1.5
    
    # Seniority indicators
    seniority_terms = ["senior", "staff", "principal", "lead", "manager", "director"]
    seniority_matches = sum(1 for term in seniority_terms if term in content)
    score += seniority_matches * 0.5
    
    # Recent departure indicators
    departure_terms = ["senior", "staff", "principal", "lead", "manager", "director"]
    departure_matches = sum(1 for term in departure_terms if term in content)
    score += departure_matches * 1.0
    
    # Include keywords bonus
    if search_params.includeKeywords:
        include_terms = [term.strip().lower() for term in search_params.includeKeywords]
        include_matches = sum(1 for term in include_terms if term in content)
        score += include_matches * 0.8
    
    # Exclude keywords penalty
    if search_params.excludeKeywords:
        exclude_terms = [term.strip().lower() for term in search_params.excludeKeywords]
        exclude_matches = sum(1 for term in exclude_terms if term in content)
        score -= exclude_matches * 1.0
    
    # Domain authority bonus
    high_authority_domains = ["linkedin.com", "twitter.com", "x.com", "techcrunch.com", "medium.com", "github.com"]
    display_link = result.get("displayLink", "").lower()
    if any(domain in display_link for domain in high_authority_domains):
        score += 0.5
    
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
    transitions = []
    # Improved: Find all experience blocks for the company and extract end dates from date ranges
    exp_block_pattern = re.compile(
        rf"{re.escape(company)}.*?(?:\n|\r|\s)+.*?(\w{{3,9}} \d{{4}})\s*[–-]\s*(\w{{3,9}} \d{{4}}|Present|present|Current|current|\d{{4}})",
        re.IGNORECASE | re.DOTALL
    )
    for match in exp_block_pattern.finditer(text):
        start_str, end_str = match.group(1), match.group(2)
        # Only consider if end_str is not 'Present' or 'Current'
        if end_str.lower() not in ['present', 'current']:
            try:
                dt = parse_end_date(end_str)
                if dt:
                    transitions.append({"type": "experience_end", "value": end_str, "full": match.group(0), "date": dt})
            except Exception:
                continue
    # Fallback to original logic if no matches
    if not transitions:
        # Patterns for quit/transition events (expanded)
        patterns = [
            # Date ranges (LinkedIn, GitHub, etc)
            (r'(\d{4})\s*[–-]\s*(\d{4})', 'range'),
            # Explicit quit/left
            (rf'left {company} in (\w+ \d{{4}})', 'left_in_month_year'),
            (rf'left {company} in (\d{{4}})', 'left_in_year'),
            (rf'left {company}', 'left'),
            (rf'former {company}', 'former'),
            # Robust ex-company patterns (dash, space, quotes, case-insensitive)
            (rf'ex[-\s\"\']*{company}', 'ex'),
            (rf'ex[-\s\"\']*{company}', 'ex'),
            (rf'previously at {company}', 'previously'),
            (rf'departed {company}', 'departed'),
            (rf'quit {company}', 'quit'),
            (rf'resigned from {company}', 'resigned'),
            (rf'no longer at {company}', 'no_longer'),
            (rf'ended at {company}', 'ended'),
            (rf'retired from {company}', 'retired'),
            (rf'stepped down from {company}', 'stepped_down'),
            (rf'last day at {company}', 'last_day'),
            (rf'moved on from {company}', 'moved_on'),
            (rf'transitioned from {company}', 'transitioned'),
            (rf'leaving {company}', 'leaving'),
            (rf'why I left {company}', 'why_left'),
            (rf'until (\w+ \d{{4}})', 'until_month_year'),
            (rf'until (\d{{4}})', 'until_year'),
            # New company join
            (r'joined ([A-Za-z0-9&.,\- ]+) in (\w+ \d{4})', 'joined_new_month_year'),
            (r'joined ([A-Za-z0-9&.,\- ]+) in (\d{4})', 'joined_new_year'),
            (r'started at ([A-Za-z0-9&.,\- ]+) in (\w+ \d{4})', 'started_new_month_year'),
            (r'started at ([A-Za-z0-9&.,\- ]+) in (\d{4})', 'started_new_year'),
            (r'now at ([A-Za-z0-9&.,\- ]+)', 'now_at'),
            # Explicitly still at company
            (rf'current at {company}', 'current'),
            (rf'present at {company}', 'present'),
            (rf'still at {company}', 'still'),
            # Fallback: any month-year or year
            (r'(\w+ \d{4})', 'month_year'),
            (r'(\d{4})', 'year'),
            # --- Education section patterns ---
            (r'graduated in (\w+ \d{4})', 'education_graduation_month_year'),
            (r'graduated in (\d{4})', 'education_graduation_year'),
            (r'education.*(\d{4})\s*[–-]\s*(\d{4})', 'education_range'),
            (r'education.*until (\w+ \d{4})', 'education_until_month_year'),
            (r'education.*until (\d{4})', 'education_until_year'),
            (r'education.*(\w+ \d{4})', 'education_month_year'),
            (r'education.*(\d{4})', 'education_year'),
            # --- Experience section patterns ---
            (r'experience.*(\d{4})\s*[–-]\s*(\d{4})', 'experience_range'),
            (r'experience.*until (\w+ \d{4})', 'experience_until_month_year'),
            (r'experience.*until (\d{4})', 'experience_until_year'),
            (r'experience.*(\w+ \d{4})', 'experience_month_year'),
            (r'experience.*(\d{4})', 'experience_year'),
        ]
        for pattern, label in patterns:
            for match in re.findall(pattern, text, re.IGNORECASE):
                if isinstance(match, tuple):
                    transitions.append({'type': label, 'value': match[-1], 'full': match})
                else:
                    transitions.append({'type': label, 'value': match, 'full': match})
    transitions.sort(key=lambda x: x.get("date", datetime.min), reverse=True)
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
    """Check if quit_date (datetime.date) is within the quit window from today."""
    if not quit_date or not quit_window:
        return None  # ambiguous
    now = datetime.utcnow().date()
    quit_window = quit_window.lower().strip()
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
    """Call ChatGPT to extract entity info and score for each snippet. Returns a list of dicts. Multi-prompt fallback for robustness."""
    if not OPENAI_API_KEY or not snippets:
        return [{"score": 0.0, "name": None, "role": None, "quit_status": None, "quit_date": None, "company": None, "linkedin_url": None, "evidence_phrase": None, "confidence": 0, "rationale": "No LLM", "rawGoogleResult": True} for _ in snippets]

    # Create structured prompts with clear JSON format
    snippets_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(snippets)])
    
    prompts = [
        # Prompt 1: Simple and reliable extraction
        f"""Extract ex-employee information from each snippet. Return a JSON array with exactly {len(snippets)} objects.

For each snippet, extract:
- name: Full name if found, null if not
- role: Job title/role if found, null if not  
- quit_status: "ex-employee" if they left, "current" if still there, "ambiguous" if unclear
- quit_date: Date they left (YYYY-MM-DD), null if not found
- company: "{company}"
- linkedin_url: LinkedIn URL if in snippet, null if not
- evidence_phrase: Key phrase showing they left, null if not found
- confidence: 0-10 score (10=definite, 7-9=strong, 4-6=moderate, 1-3=weak, 0=no evidence)
- rationale: Brief explanation of reasoning

Look for: "ex-{company}", "former {company}", "left {company}", "quit {company}", "resigned from {company}", "until 2024", "joined [new company]"

Return ONLY valid JSON array. No commentary.

Snippets:
{snippets_text}""",

        # Prompt 2: Focus on quit detection
        f"""For each snippet, determine if someone left {company} and extract info. Return JSON array with {len(snippets)} objects.

Each object: {{"name": "...", "role": "...", "quit_status": "...", "quit_date": "...", "company": "{company}", "linkedin_url": "...", "evidence_phrase": "...", "confidence": 0-10, "rationale": "..."}}

Key indicators: "ex-{company}", "former {company}", "left {company}", "quit {company}", "resigned from {company}", "until 2024", "joined [new company]"

If uncertain, use null values and low confidence scores.

Snippets:
{snippets_text}""",

        # Prompt 3: Minimal format
        f"""Extract ex-employee data from snippets. Return JSON array with {len(snippets)} objects.

Format: [{{"name": "...", "role": "...", "quit_status": "...", "quit_date": "...", "company": "{company}", "linkedin_url": "...", "evidence_phrase": "...", "confidence": 0-10, "rationale": "..."}}]

Look for: "ex-{company}", "former {company}", "left {company}", "quit {company}"

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
        for i, query in enumerate(request.queries):
            # Choose search method based on available APIs
            if SERPAPI_KEY:
                search_results = search_serpapi(query)
            elif GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
                search_results = search_google_custom(query, quit_window=str(request.searchParams.quitWindow or ''))
            else:
                search_results = []
            logger.info(f"Google results for query {i+1}: {len(search_results)}")
            for result in search_results:
                url = result.get("link", "")
                if url in seen_urls or not url:
                    continue
                seen_urls.add(url)
                snippet = result.get("snippet", "")
                title = result.get("title", "")
                display_link = result.get("displayLink", "")
                # Filter out non-profile links (news, company pages, jobs, etc.)
                if any(x in url for x in ["/jobs", "/careers", "/company", "/news", "/updates", "/about", "/press", "/blog", "/events", "/groups", "/pages", "/stories", "/media", "/services", "/solutions", "/products", "/webinar", "/podcast", "/award", "/recognition", "/announcement"]):
                    continue
                if not ("linkedin.com/in/" in url or "twitter.com/" in url or "x.com/" in url or "github.com/" in url or "angel.co/" in url or "wellfound.com/" in url):
                    continue
                # --- Extract transitions, quit date, and confidence ---
                transitions = extract_all_transitions(snippet + " " + title, request.searchParams.company)
                logger.info(f"Transitions for URL {url}: {transitions}")
                inferred_quit_date_str = infer_quit_date_from_transitions(transitions, request.searchParams.company)
                logger.info(f"Inferred quit date string for URL {url}: {inferred_quit_date_str}")
                quit_date = parse_end_date(inferred_quit_date_str or "") if inferred_quit_date_str else None
                quit_date_str = quit_date.strftime("%Y-%m-%d") if quit_date else None
                logger.info(f"Parsed quit date for URL {url}: {quit_date_str}")
                # Confidence: explicit > inferred > ambiguous
                confidence = 1
                explicit_types = [
                    'left_in_month_year', 'left_in_year', 'until_month_year', 'until_year',
                    'left', 'former', 'ex', 'departed', 'quit', 'resigned', 'no_longer', 'ended', 'retired', 'stepped_down', 'last_day', 'moved_on', 'transitioned', 'leaving', 'why_left'
                ]
                if any(t['type'] in explicit_types for t in transitions):
                    confidence = 3
                    logger.info(f"Explicit quit transition found for URL: {url}")
                elif any(t['type'] in ['joined_new_month_year', 'joined_new_year', 'started_new_month_year', 'started_new_year', 'now_at'] for t in transitions):
                    confidence = 2
                    logger.info(f"Inferred quit from new job for URL: {url}")
                else:
                    logger.info(f"Ambiguous quit for URL: {url}")
                # Scoring: keyword and source authority
                keyword_hits = 0
                weight = 2.0
                for kw in (request.searchParams.includeKeywords or []):
                    if kw.strip() and kw.strip().lower() in (snippet + " " + title).lower():
                        keyword_hits += 1
                # Add recency bonus if quit_date is within 1 year
                recency_bonus = 0.0
                if quit_date and is_within_quit_window_smart(quit_date, request.searchParams.quitWindow or ""):
                    recency_bonus = 3.0
                # Add recency bonus if quit_date is within 2 years
                if quit_date and is_within_quit_window_smart(quit_date, request.searchParams.quitWindow or ""):
                    recency_bonus = 1.5
                score = (keyword_hits * 2.5) + recency_bonus
                # Source authority boost
                source = get_source_from_display_link(display_link)
                source_boosts = {
                    "linkedin": 4.0,
                    "twitter": 2.0,
                    "github": 1.0,
                    "medium": 0.5,
                    "wellfound": 1.0,
                    "blog": 0.5,
                    "other": 0.0
                }
                score += source_boosts.get(source, 0.0)
                relevance_score = score
                # Only include results that mention the target company as a former employer
                company = request.searchParams.company.lower()
                if not any(
                    kw in (snippet + " " + title).lower()
                    for kw in [f"ex-{company}", f"ex {company}", f"formerly at {company}", f"left {company}", f"departed {company}", f"resigned from {company}", f"no longer at {company}", f"previously at {company}"]
                ):
                    continue
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
                    "source": source or None,
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
        # --- Prioritize LinkedIn results at the top ---
        all_results.sort(key=lambda x: (x["source"] != "linkedin", -x["relevanceScore"]))
        # Always ensure all fields are present after sorting
        for i, result in enumerate(all_results):
            result["rank"] = i + 1
            for field in ["llmName", "llmRole", "llmQuitStatus", "llmQuitDate", "llmCompany", "llmLinkedinUrl", "llmEvidencePhrase", "llmConfidence", "llmRationale", "llmRawGoogleResult", "rawGoogleResult", "quitDate", "quitConfidence", "transitions", "source"]:
                if field not in result:
                    result[field] = None
        # In the /search endpoint, after LLM rerank, set 'relevanceScore' to the LLM confidence/score if present, else use the calculated score. Always include 'relevanceScore' in the result dict.
        for r in all_results:
            if r["llmConfidence"] is not None:
                r["relevanceScore"] = r["llmConfidence"]
            else:
                r["relevanceScore"] = r["llmRawGoogleResult"] if r["llmRawGoogleResult"] is not None else r["relevanceScore"]
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
