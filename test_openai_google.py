import asyncio
import requests
import json

BASE_URL = "http://localhost:8000"

async def test_openai_integration():
    """Test OpenAI ChatGPT integration"""
    print("🤖 Testing OpenAI ChatGPT Integration...")
    
    payload = {
        "company": "Google",
        "roles": "Senior Software Engineer, Staff Engineer",
        "seniority": "Senior",
        "quitWindow": "6 months",
        "geography": "San Francisco, New York",
        "includeKeywords": "AI, machine learning, distributed systems",
        "excludeKeywords": "intern, junior"
    }
    
    response = requests.post(f"{BASE_URL}/generate-queries", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {len(data['queries'])} queries")
        for i, query in enumerate(data['queries'], 1):
            print(f"   {i}. {query}")
        return data['queries']
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return []

async def test_google_search(queries):
    """Test Google Custom Search integration"""
    print("\n🔍 Testing Google Custom Search Integration...")
    
    if not queries:
        print("❌ No queries to test")
        return
    
    payload = {
        "queries": queries[:3],  # Test first 3 queries
        "searchParams": {
            "company": "Google",
            "roles": "Senior Software Engineer",
            "seniority": "Senior",
            "quitWindow": "6 months"
        }
    }
    
    response = requests.post(f"{BASE_URL}/execute-search", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        results = data['results']
        print(f"✅ Found {len(results)} results")
        
        for result in results[:3]:  # Show top 3
            print(f"   Rank {result['rank']}: {result['domain']}")
            print(f"   Score: {result['relevanceScore']:.1f}")
            print(f"   Snippet: {result['snippet'][:100]}...")
            print()
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")

async def test_health_check():
    """Test health check with service status"""
    print("🏥 Testing Health Check...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        
        services = data['services']
        print(f"   OpenAI ChatGPT: {services['openai_chatgpt']['status']}")
        print(f"   Google Search: {services['google_custom_search']['status']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

async def main():
    """Run all tests"""
    print("🚀 Testing VC Analyst Search Tool API")
    print("=" * 50)
    
    # Test health check
    await test_health_check()
    
    # Test OpenAI integration
    queries = await test_openai_integration()
    
    # Test Google Search integration
    await test_google_search(queries)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
