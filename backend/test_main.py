import pytest
from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)

def test_generate_queries():
    payload = {
        "company": "Stripe",
        "roles": ["Software Engineer", "Backend Developer"],
        "seniority": "Senior",
        "quitWindow": "past 6 months",
        "geography": "San Francisco OR Bay Area",
        "includeKeywords": ["Python", "AWS"],
        "excludeKeywords": ["intern", "freelancer"]
    }
    response = client.post("/generate-queries", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "queries" in data
    assert isinstance(data["queries"], list)
    assert len(data["queries"]) == 7
    assert all("linkedin.com/in" in q for q in data["queries"])

def test_search_empty():
    payload = {
        "queries": ["site:linkedin.com/in ex-Stripe"],
        "searchParams": {
            "company": "Stripe",
            "roles": ["Software Engineer"],
            "seniority": "Senior",
            "quitWindow": "past 6 months",
            "geography": "San Francisco OR Bay Area",
            "includeKeywords": ["Python"],
            "excludeKeywords": ["intern"]
        }
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

def test_search_speed():
    payload = {
        "queries": ["site:linkedin.com/in ex-Stripe"],
        "searchParams": {
            "company": "Stripe",
            "roles": ["Software Engineer"],
            "seniority": "Senior",
            "quitWindow": "past 6 months",
            "geography": "San Francisco OR Bay Area",
            "includeKeywords": ["Python"],
            "excludeKeywords": ["intern"]
        }
    }
    start = time.time()
    response = client.post("/search", json=payload)
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 300  # 5 minutes

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy" 