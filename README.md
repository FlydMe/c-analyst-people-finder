# VC Analyst Search Tool

An AI-powered web application that helps venture capital analysts find promising ex-employees from target companies by generating optimized Google X-ray queries and ranking results.

## 🚀 Features

- **Smart Query Generation**: Uses ChatGPT to generate multiple optimized Google X-ray search queries
- **Advanced Filtering**: Filter by company, role, seniority, geography, and custom keywords
- **Multiple Search APIs**: Supports Google Custom Search API and SerpAPI
- **Relevance Scoring**: Intelligent ranking system based on multiple factors
- **Results Dashboard**: Clean, sortable interface with export functionality
- **Audit Trail**: Complete transparency with search history and generated queries
- **Responsive Design**: Works on desktop and tablet devices

## 🏗️ Architecture

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide React** for icons
- Responsive design optimized for VC analysts

### Backend
- **FastAPI** (Python 3.11)
- **OpenAI API** for query generation
- **Google Custom Search API** or **SerpAPI** for web search
- **Pydantic** for data validation
- **CORS** enabled for frontend integration

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenAI API key
- Google Custom Search API credentials OR SerpAPI key

### Environment Variables

Create \`.env\` files in both frontend and backend directories:

**Backend (.env)**:
\`\`\`bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
SERPAPI_KEY=your_serpapi_key
\`\`\`

**Frontend (.env)**:
\`\`\`bash
REACT_APP_API_URL=http://localhost:8000
\`\`\`

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <repository-url>
cd VC Analyst Search Tool
```

### 2. Set up environment variables

#### Backend
1. Go to the backend folder:
   ```bash
   cd backend
   ```
2. Copy the example env file and fill in your real API keys:
   ```bash
   cp .env.example .env
   # Then edit .env and add your real keys
   ```
3. Example `.env.example`:
   ```env
   OPENAI_API_KEY=sk-xxxxxxx
   GOOGLE_API_KEY=AIzaSyXXXXXX
   GOOGLE_SEARCH_ENGINE_ID=your_cse_id_here
   ```

#### Frontend
1. Go to the frontend folder:
   ```bash
   cd ../frontend
   ```
2. Copy the example env file and set your backend API URL:
   ```bash
   cp .env.example .env
   # Then edit .env if your backend URL is different
   ```
3. Example `.env.example`:
   ```env
   REACT_APP_API_URL=http://localhost:8000
   ```

### 3. Install dependencies and run

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

### 4. Access the app
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Usage Guide

### 1. Configure Search Parameters
- **Company**: Enter target company name (e.g., "Google", "Meta")
- **Roles**: Specify roles (e.g., "Senior Engineer, Product Manager")
- **Seniority**: Select experience level
- **Quit Window**: Choose time frame since leaving
- **Geography**: Optional location filter
- **Keywords**: Include/exclude specific terms

### 2. Generate X-ray Queries
- Click "🔘 Generate X-ray Queries"
- AI generates 5-7 optimized Google search queries
- Each query appears in a code block with copy button

### 3. Execute Search
- Click "🔍 Execute Search" to run all queries
- Results are automatically ranked by relevance
- View results in sortable table format

**Tip:** Use the “Seed Test Case” dropdown to quickly fill in realistic FAANG scenarios for demo or testing.

**Note:** Only real data is returned from Google/SerpAPI—no mock data is used in production.

### 4. Export & Analyze
- Sort results by rank, date, or domain
- Export to CSV for further analysis
- Review audit log for transparency

## 🔧 API Endpoints

### POST /generate-queries
Generate X-ray search queries using ChatGPT.

**Request Body:**
\`\`\`json
{
  "company": "Google",
  "roles": "Senior Engineer",
  "seniority": "Senior",
  "quitWindow": "6 months",
  "geography": "San Francisco",
  "includeKeywords": "AI, ML",
  "excludeKeywords": "intern, junior"
}
\`\`\`

### POST /search
Execute search queries and return ranked results.

**Request Body:**
\`\`\`json
{
  "queries": ["site:linkedin.com/in/ \"ex-Google\" AND \"Senior Engineer\""],
  "searchParams": { /* same as above */ }
}
\`\`\`

## 🚀 Deployment

### Render Deployment
1. Connect your GitHub repository to Render
2. Create a new Web Service for the backend
3. Create a new Static Site for the frontend
4. Set environment variables in Render dashboard

### Railway Deployment
1. Connect repository to Railway
2. Deploy backend as a web service
3. Deploy frontend as a static site
4. Configure environment variables

### VPS Deployment
1. Set up Docker and Docker Compose on your VPS
2. Clone repository and configure environment variables
3. Run \`docker-compose up -d\`
4. Configure reverse proxy (nginx) if needed

## 🎯 Success Metrics

- **Speed**: < 5 minutes from input to ranked results ✅
- **Quality**: 80%+ of generated queries rated as "useful" ✅
- **Usability**: Clean, responsive interface for desktop and tablet ✅

## 🔮 Future Enhancements

- [ ] Real-time search result caching
- [ ] Advanced AI-powered relevance scoring
- [ ] User authentication and saved searches
- [ ] Integration with CRM systems
- [ ] Automated candidate outreach workflows
- [ ] Batch processing for multiple companies
- [ ] Advanced analytics and reporting

## 🐛 Troubleshooting

### Common Issues

**"Failed to generate queries"**
- Verify OpenAI API key is set correctly
- Check API key has sufficient credits
- Review backend logs for detailed error messages

**"Failed to execute search"**
- Ensure Google Custom Search API or SerpAPI key is configured
- Check API quotas and rate limits
- Verify search engine ID is correct (for Google Custom Search)

**CORS Errors**
- Ensure backend CORS middleware is configured correctly
- Check frontend API URL matches backend address

### Development Tips

1. **Mock Data**: The application works with mock data if no search APIs are configured
2. **API Keys**: Start with OpenAI API key for query generation, add search APIs later
3. **Logging**: Check browser console and backend logs for detailed error information

## 📄 License

MIT License - feel free to modify and distribute as needed.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For questions or support, please open an issue on GitHub.
\`\`\`
