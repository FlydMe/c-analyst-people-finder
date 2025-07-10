import pytest
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_generate_queries():
    """Test query generation endpoint"""
    payload = {
        "company": "Google",
        "roles": "Senior Engineer",
        "seniority": "Senior",
        "quitWindow": "6 months",
        "geography": "San Francisco",
        "includeKeywords": "AI, ML",
        "excludeKeywords": "intern"
    }
    
    response = requests.post(f"{BASE_URL}/generate-queries", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "queries" in data
    assert "prompt" in data
    assert len(data["queries"]) > 0

def test_execute_search():
    """Test search execution endpoint"""
    # First generate queries
    query_payload = {
        "company": "Google",
        "roles": "Engineer",
        "quitWindow": "1 year"
    }
    
    query_response = requests.post(f"{BASE_URL}/generate-queries", json=query_payload)
    queries = query_response.json()["queries"]
    
    # Then execute search
    search_payload = {
        "queries": queries,
        "searchParams": query_payload
    }
    
    response = requests.post(f"{BASE_URL}/execute-search", json=search_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

if __name__ == "__main__":
    # Run basic tests
    test_health_check()
    print("✅ Health check passed")
    
    test_generate_queries()
    print("✅ Query generation passed")
    
    test_execute_search()
    print("✅ Search execution passed")
    
    print("🎉 All tests passed!")
