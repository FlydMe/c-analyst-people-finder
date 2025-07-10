"use client"

import React from "react"
import { useState } from "react"
import { Search, Download, Copy, ExternalLink, ChevronDown, ChevronUp, Loader2, Star } from "lucide-react"

interface SearchParams {
  company: string
  roles: string[]
  seniority: string
  quit_window: string
  geography: string
  include_keywords: string[]
  exclude_keywords: string[]
}

interface SearchResult {
  rank: number;
  snippet: string;
  domain: string;
  link: string;
  relevanceScore: number;
  source?: string;
  quitDate?: string;
  quitConfidence?: number;
  transitions?: { type: string; value: string; full: string | string[] }[];
  llmName?: string;
  llmRole?: string;
  llmQuitStatus?: string;
  llmQuitDate?: string;
  llmCompany?: string;
  llmLinkedinUrl?: string;
  llmEvidencePhrase?: string;
  llmConfidence?: number;
  llmRationale?: string;
  llmRawGoogleResult?: boolean;
  rawGoogleResult?: boolean;
}

interface AuditEntry {
  timestamp: string
  input_parameters: SearchParams
  raw_gpt_queries: string[]
  gpt_prompt: string
  total_results: number
}

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000"

// TypeScript: declare process for env usage in browser
// @ts-ignore
// eslint-disable-next-line no-var
declare var process: any;

const App: React.FC = () => {
  const [searchParams, setSearchParams] = useState<SearchParams>({
    company: "",
    roles: [],
    seniority: "",
    quit_window: "",
    geography: "",
    include_keywords: [],
    exclude_keywords: [],
  })

  const [generatedQueries, setGeneratedQueries] = useState<string[]>([])
  const [results, setResults] = useState<SearchResult[]>([])
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [sortBy, setSortBy] = useState<"rank" | "date" | "relevance_score">("relevance_score")
  const [showAuditLog, setShowAuditLog] = useState(false)
  const [error, setError] = useState<string>("")
  const [lastUsedPrompt, setLastUsedPrompt] = useState("")
  const [linkedinOnly, setLinkedinOnly] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);

  // 🧠 Step 2: Call ChatGPT API to Generate X-Ray Strings
  const handleGenerateQueries = async () => {
    if (!searchParams.company.trim()) {
      setError("Company name is required")
      return
    }

    setIsGenerating(true)
    setError("")

    try {
      // Convert string inputs to arrays for API
      const apiParams = {
        company: searchParams.company,
        roles: searchParams.roles.length > 0 ? searchParams.roles : searchParams.roles,
        seniority: searchParams.seniority,
        quit_window: searchParams.quit_window,
        geography: searchParams.geography,
        include_keywords: searchParams.include_keywords,
        exclude_keywords: searchParams.exclude_keywords,
      }

      const response = await fetch(`${API_BASE_URL}/generate-queries`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(apiParams),
      })

      if (!response.ok) {
        throw new Error("Failed to generate queries")
      }

      const data = await response.json()
      setGeneratedQueries(data.queries)
      setLastUsedPrompt(data.prompt)

      // Add to audit log
      const auditEntry: AuditEntry = {
        timestamp: data.timestamp,
        input_parameters: searchParams,
        raw_gpt_queries: data.queries,
        gpt_prompt: data.prompt,
        total_results: 0,
      }
      setAuditLog((prev) => [auditEntry, ...prev])
    } catch (error) {
      setError("Failed to generate queries. Please try again.")
      console.error("Query generation error:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  // 🔍 Step 3: Execute search with Google/SerpAPI
  const handleSearch = async () => {
    if (generatedQueries.length === 0) {
      setError("Please generate queries first")
      return
    }

    setIsSearching(true)
    setError("")

    try {
      // Convert array fields to comma-separated strings for backend compatibility
      const searchParamsForBackend = {
        ...searchParams,
        roles: Array.isArray(searchParams.roles) ? searchParams.roles.join(",") : searchParams.roles,
        includeKeywords: Array.isArray(searchParams.include_keywords) ? searchParams.include_keywords.join(",") : searchParams.include_keywords,
        excludeKeywords: Array.isArray(searchParams.exclude_keywords) ? searchParams.exclude_keywords.join(",") : searchParams.exclude_keywords,
        quitWindow: searchParams.quit_window,
        include_keywords: undefined,
        exclude_keywords: undefined,
        quit_window: undefined,
      }
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queries: generatedQueries,
          searchParams: searchParamsForBackend,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to execute search")
      }

      const data = await response.json()
      setResults(data.results)

      // Update audit log with results count
      setAuditLog((prev: AuditEntry[]) =>
        prev.map((entry: AuditEntry, index: number) => (index === 0 ? { ...entry, total_results: data.results.length } : entry)),
      )
    } catch (error) {
      setError("Failed to execute search. Please try again.")
      console.error("Search error:", error)
    } finally {
      setIsSearching(false)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  // 📊 Step 4: Sort & Export functionality
  const sortedResults = [...results].sort((a: SearchResult, b: SearchResult) => {
    switch (sortBy) {
      case "rank":
        return a.rank - b.rank
      case "date":
        const dateA = a.llmQuitDate || a.quitDate || "";
        const dateB = b.llmQuitDate || b.quitDate || "";
        return new Date(dateB).getTime() - new Date(dateA).getTime()
      case "relevance_score":
        return b.relevanceScore - a.relevanceScore
      default:
        return 0
    }
  })

  const filteredResults = linkedinOnly
    ? sortedResults.filter((r) => r.llmLinkedinUrl)
    : sortedResults;

  const exportToCSV = () => {
    const headers = ["Rank", "Snippet", "Domain", "Link", "Quit Date", "Relevance Score", "Name", "Role", "Quit Status", "Confidence"];
    const csvContent = [
      headers.join(","),
      ...results.map((result) =>
        [
          result.rank,
          `"${result.snippet.replace(/"/g, '""')}"`,
          result.domain,
          result.link,
          result.llmQuitDate || result.quitDate || "",
          result.relevanceScore,
          result.llmName || "",
          result.llmRole || "",
          result.llmQuitStatus || "",
          result.llmConfidence || result.quitConfidence || "",
        ].join(","),
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vc-search-results-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Helper function to handle array inputs
  const handleArrayInput = (
    value: string,
    field: keyof Pick<SearchParams, "roles" | "include_keywords" | "exclude_keywords">,
  ) => {
    const array = value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0)
    setSearchParams((prev) => ({ ...prev, [field]: array }))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 text-center">VC Analyst Search Tool</h1>
          <p className="text-gray-600 text-center mt-2">ChatGPT → Google X-ray Queries → Ranked Results → Export</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {/* Error Display */}
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">{error}</div>}

        {/* 🌐 Step 1: Form Section */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h2 className="text-xl font-semibold mb-4">Search Parameters</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {/* Company */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Company <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                placeholder="e.g., Google"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.company}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, company: e.target.value }))}
              />
            </div>

            {/* Roles */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Roles <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                placeholder="Senior Engineer, Product Manager (comma-separated)"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                onChange={(e) => handleArrayInput(e.target.value, "roles")}
              />
            </div>

            {/* Seniority */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Seniority</label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.seniority}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, seniority: e.target.value }))}
              >
                <option value="">Select seniority</option>
                <option value="Junior">Junior</option>
                <option value="Mid">Mid</option>
                <option value="Senior">Senior</option>
                <option value="Staff">Staff</option>
                <option value="Principal">Principal</option>
                <option value="Director">Director</option>
                <option value="VP">VP</option>
              </select>
            </div>

            {/* Quit Window */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Quit Window</label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.quit_window}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, quit_window: e.target.value }))}
              >
                <option value="">Select time window</option>
                <option value="Last 3 months">Last 3 months</option>
                <option value="6 months">6 months</option>
                <option value="1 year">1 year</option>
                <option value="2 years">2 years</option>
                <option value="until 2024">Until 2024</option>
              </select>
            </div>

            {/* Geography */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Geography</label>
              <input
                type="text"
                placeholder="e.g., San Francisco, New York"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.geography}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, geography: e.target.value }))}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {/* Include Keywords */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Include Keywords <span className="text-green-600 text-xs">(higher score)</span>
              </label>
              <textarea
                placeholder="AI, ML, startup (comma-separated)"
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                onChange={(e) => handleArrayInput(e.target.value, "include_keywords")}
              />
            </div>

            {/* Exclude Keywords */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Exclude Keywords</label>
              <textarea
                placeholder="intern, junior (comma-separated)"
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                onChange={(e) => handleArrayInput(e.target.value, "exclude_keywords")}
              />
            </div>
          </div>

          <button
            onClick={handleGenerateQueries}
            disabled={isGenerating}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Calling ChatGPT API...
              </>
            ) : (
              "🧠 Generate X-ray Queries with ChatGPT"
            )}
          </button>
        </div>

        {/* 🧾 Generated X-ray Query Output Section */}
        {generatedQueries.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold mb-4">🧾 Generated X-ray Queries</h2>
            <p className="text-gray-600 mb-4">ChatGPT generated {generatedQueries.length} search strings</p>

            <div className="space-y-3 mb-4">
              {generatedQueries.map((query, index) => (
                <div key={index} className="relative">
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                    <code>{query}</code>
                  </div>
                  <button
                    className="absolute top-2 right-2 bg-gray-700 hover:bg-gray-600 text-white p-1 rounded"
                    onClick={() => copyToClipboard(query)}
                  >
                    <Copy className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>

            <button
              onClick={handleSearch}
              disabled={isSearching}
              className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isSearching ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Searching Google API...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />🔍 Execute Search with Google API
                </>
              )}
            </button>
          </div>
        )}

        {/* LLM Prompt Display */}
        {lastUsedPrompt && (
          <div className="bg-white rounded-lg shadow-sm border p-4 mb-4">
            <button
              onClick={() => setShowPrompt((v) => !v)}
              className="flex items-center text-blue-700 hover:text-blue-900 font-semibold mb-2"
            >
              {showPrompt ? 'Hide' : 'Show'} LLM Prompt Used
            </button>
            {showPrompt && (
              <pre className="bg-gray-100 rounded p-3 text-xs overflow-x-auto whitespace-pre-wrap border text-gray-800">
                {lastUsedPrompt}
              </pre>
            )}
          </div>
        )}

        {/* 📊 Step 4: Results Table/Card View */}
        {results.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-4 gap-2">
              <div>
                <h2 className="text-xl font-semibold">📊 Ranked Results</h2>
                <p className="text-gray-600">Found {filteredResults.length} potential candidates</p>
              </div>
              <div className="flex flex-col md:flex-row gap-2 items-center">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={linkedinOnly}
                    onChange={() => setLinkedinOnly((v) => !v)}
                    className="accent-blue-600"
                  />
                  <span className="text-blue-700 font-medium">LinkedIn Only</span>
                </label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="relevance_score">🎯 Relevance Score</option>
                  <option value="rank">📊 Rank</option>
                  <option value="date">📅 Date</option>
                </select>
                <button
                  onClick={exportToCSV}
                  className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 flex items-center"
                >
                  <Download className="mr-2 h-4 w-4" />
                  Export CSV
                </button>
              </div>
            </div>
            <div className="space-y-4">
              {filteredResults.map((result: SearchResult) => (
                <div
                  key={result.rank}
                  className={`border rounded-lg p-4 hover:bg-gray-50 transition-shadow shadow-sm ${result.source === 'linkedin' ? 'border-blue-500 bg-blue-50 shadow-md' : ''}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-medium">
                        #{result.rank}
                      </span>
                      <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm font-medium flex items-center gap-1">
                        <Star className="h-3 w-3" />
                        {result.relevanceScore}
                      </span>
                      {result.llmLinkedinUrl && (
                        <span className="ml-2 px-2 py-1 bg-blue-700 text-white rounded text-xs font-bold shadow inline-flex items-center gap-1">
                          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-10h3v10zm-1.5-11.268c-.966 0-1.75-.784-1.75-1.75s.784-1.75 1.75-1.75 1.75.784 1.75 1.75-.784 1.75-1.75 1.75zm15.5 11.268h-3v-5.604c0-1.337-.025-3.063-1.868-3.063-1.868 0-2.154 1.459-2.154 2.967v5.7h-3v-10h2.881v1.367h.041c.401-.761 1.379-1.563 2.838-1.563 3.034 0 3.595 1.997 3.595 4.59v5.606z"/></svg>
                          LinkedIn
                        </span>
                      )}
                      {result.llmCompany && (
                        <span className="ml-2 px-2 py-1 rounded text-xs font-semibold bg-green-200 text-green-800">Company: {result.llmCompany}</span>
                      )}
                      {result.llmQuitStatus && (
                        <span className="ml-2 px-2 py-1 rounded text-xs font-semibold bg-yellow-100 text-yellow-800">{result.llmQuitStatus}</span>
                      )}
                      {typeof result.llmConfidence === 'number' && (
                        <span className={`ml-2 px-2 py-1 rounded text-xs font-semibold ${result.llmConfidence >= 9 ? 'bg-green-500 text-white' : result.llmConfidence >= 7 ? 'bg-green-200 text-green-900' : 'bg-gray-200 text-gray-700'}`}>LLM Confidence: {result.llmConfidence}</span>
                      )}
                    </div>
                    <span className="text-sm text-gray-500">{result.llmQuitDate || result.quitDate || ""}</span>
                  </div>
                  <p className="text-gray-700 mb-2 font-medium">{result.snippet}</p>
                  <div className="text-sm text-gray-500 mb-2">{result.domain}</div>
                  {/* LLM extracted fields */}
                  <div className="mt-3 p-3 bg-gray-50 rounded-lg border-l-4 border-blue-500">
                    <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      AI Analysis
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                      {result.llmName && (
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">Name:</span>
                          <span className="text-gray-900">{result.llmName}</span>
                        </div>
                      )}
                      {result.llmRole && (
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">Role:</span>
                          <span className="text-gray-900">{result.llmRole}</span>
                        </div>
                      )}
                      {result.llmQuitDate && (
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">Quit Date:</span>
                          <span className="text-gray-900">{result.llmQuitDate}</span>
                        </div>
                      )}
                      {result.llmEvidencePhrase && (
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">Evidence:</span>
                          <span className="italic text-blue-700 bg-blue-100 px-2 py-1 rounded">{result.llmEvidencePhrase}</span>
                        </div>
                      )}
                      {result.llmConfidence && (
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">Confidence:</span>
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                            result.llmConfidence >= 9 ? 'bg-green-500 text-white' : 
                            result.llmConfidence >= 7 ? 'bg-green-200 text-green-900' : 
                            result.llmConfidence >= 5 ? 'bg-yellow-200 text-yellow-900' : 
                            'bg-gray-200 text-gray-700'
                          }`}>
                            {result.llmConfidence}/10
                          </span>
                        </div>
                      )}
                    </div>
                    {result.llmRationale && (
                      <div className="mt-2 p-2 bg-white rounded border">
                        <span className="font-medium text-gray-700">Rationale:</span>
                        <span className="text-gray-800 ml-2">{result.llmRationale}</span>
                      </div>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3 items-center">
                    <a
                    href={result.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 flex items-center gap-1 font-semibold"
                    >
                    Open Link <ExternalLink className="h-4 w-4" />
                    </a>
                    {result.llmLinkedinUrl && (
                      <a
                        href={result.llmLinkedinUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-700 hover:underline text-xs font-semibold"
                      >
                        View LinkedIn Profile
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 🕵️‍♂️ Step 5: Audit Log Section */}
        {auditLog.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <button
              onClick={() => setShowAuditLog(!showAuditLog)}
              className="flex items-center justify-between w-full text-left"
            >
              <h2 className="text-xl font-semibold">🕵️‍♂️ Audit Log</h2>
              {showAuditLog ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {showAuditLog && (
              <div className="mt-4 space-y-4">
                {auditLog.map((entry: AuditEntry, index: number) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <p className="font-medium">Search #{auditLog.length - index}</p>
                        <p className="text-sm text-gray-600">{entry.total_results} results found</p>
                      </div>
                      <span className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleString()}</span>
                    </div>

                    <details className="text-sm mt-2">
                      <summary className="cursor-pointer text-blue-600 hover:text-blue-800 mb-2">
                        View Raw GPT Queries ({entry.raw_gpt_queries.length})
                      </summary>
                      <div className="bg-gray-50 p-3 rounded">
                        {entry.raw_gpt_queries.map((query, qIndex) => (
                          <div key={qIndex} className="font-mono text-xs mb-1">
                            {qIndex + 1}. {query}
                          </div>
                        ))}
                      </div>
                    </details>

                    <details className="text-sm mt-2">
                      <summary className="cursor-pointer text-blue-600 hover:text-blue-800 mb-2">
                        View Input Parameters
                      </summary>
                      <div className="bg-gray-50 p-3 rounded">
                        <pre className="text-xs overflow-x-auto">{JSON.stringify(entry.input_parameters, null, 2)}</pre>
                      </div>
                    </details>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
