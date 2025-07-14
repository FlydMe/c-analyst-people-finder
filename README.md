# VC Analyst Search Tool

Find promising ex-employees with AI-powered Google X-ray queries.

## Features
- Parameter form: company, roles, seniority, quit window, geography, keywords
- ChatGPT-powered query builder (7 optimized queries)
- Google Custom Search API (or SerpAPI) executor
- Smart deduplication, robust quit date extraction (education/experience)
- Simple relevance scoring + (optional) ChatGPT rerank
- Results dashboard: sortable, exportable, audit log
- No complex user auth, clean SPA UX

## Setup

### Backend (FastAPI)
1. `cd VC Analyst Search Tool/backend`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Set environment variables (see below)
5. `uvicorn main:app --reload`

### Frontend (React + Tailwind)
1. `cd VC Analyst Search Tool/frontend`
2. `npm install`
3. `npm start`

## Environment Variables
- `OPENAI_API_KEY` (for ChatGPT rerank, optional)
- `GOOGLE_API_KEY` (for Google Custom Search)
- `GOOGLE_SEARCH_ENGINE_ID` (for Google Custom Search)
- `SERPAPI_KEY` (optional, for SerpAPI)

## Deploy
- Backend: Deploy with Docker, Render, Railway, or any FastAPI-compatible host
- Frontend: Deploy with Vercel, Netlify, or any static host

## Notes
- Google API quota is respected; results are cached in-memory to reduce rate limit hits
- Prompt templates and ranking logic are easy to tune in `backend/main.py`
- For best results, set all API keys

---
MIT License 