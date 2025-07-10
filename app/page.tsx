"use client"

import { useState } from "react"
import { Search, Download, Copy, ExternalLink, ChevronDown, ChevronUp, Loader2 } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface SearchParams {
  company: string
  roles: string
  seniority: string
  quitWindow: string
  geography: string
  includeKeywords: string
  excludeKeywords: string
}

interface SearchResult {
  rank: number
  snippet: string
  domain: string
  date: string
  link: string
  relevanceScore: number
  // LLM/AI fields (all optional)
  rawGoogleResult?: boolean
  llmFuzzyMatch?: boolean
  llmUncertain?: boolean
  llmScore?: number
  llmQuitStatus?: string
  llmCompanyMatch?: boolean
  llmMatchedCompanyName?: string
  llmCompanyNameConfidence?: number
  llmCompanyNameRationale?: string
  llmConfidence?: number
  llmReason?: string
}

interface AuditEntry {
  timestamp: string
  prompt: string
  queries: string[]
  resultsCount: number
}

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000"

export default function VCAnalystSearchTool() {
  const [searchParams, setSearchParams] = useState<SearchParams>({
    company: "",
    roles: "",
    seniority: "",
    quitWindow: "",
    geography: "",
    includeKeywords: "",
    excludeKeywords: "",
  })

  const [generatedQueries, setGeneratedQueries] = useState<string[]>([])
  const [results, setResults] = useState<SearchResult[]>([])
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [sortBy, setSortBy] = useState<"rank" | "date" | "domain">("rank")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc")
  const [showAuditLog, setShowAuditLog] = useState(false)
  const [error, setError] = useState<string>("")
  const [showGptPrompt, setShowGptPrompt] = useState(false)
  const [lastUsedPrompt, setLastUsedPrompt] = useState("")
  const [formErrors, setFormErrors] = useState<{ [key: string]: boolean }>({})

  const validateForm = () => {
    const errors: { [key: string]: boolean } = {}

    if (!searchParams.company.trim()) {
      errors.company = true
    }
    if (!searchParams.roles.trim()) {
      errors.roles = true
    }
    if (!searchParams.quitWindow.trim()) {
      errors.quitWindow = true
    }

    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleGenerateQueries = async () => {
    if (!validateForm()) {
      setError("Please fill in all required fields")
      return
    }

    setIsGenerating(true)
    setError("")

    try {
      const response = await fetch("/api/generate-queries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(searchParams),
      })

      if (!response.ok) {
        throw new Error("Failed to generate queries")
      }

      const data = await response.json()
      setGeneratedQueries(data.queries)
      setLastUsedPrompt(data.prompt)

      // Add to audit log
      const auditEntry: AuditEntry = {
        timestamp: new Date().toISOString(),
        prompt: data.prompt,
        queries: data.queries,
        resultsCount: 0,
      }
      setAuditLog((prev) => [auditEntry, ...prev])
    } catch (error) {
      setError("Failed to generate queries. Please try again.")
      console.error("Query generation error:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleSearch = async () => {
    if (generatedQueries.length === 0) {
      setError("Please generate queries first")
      return
    }

    setIsSearching(true)
    setError("")

    try {
      const response = await fetch("/api/execute-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queries: generatedQueries,
          searchParams,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to execute search")
      }

      const data = await response.json()
      setResults(data.results)

      // Update audit log with results count
      setAuditLog((prev) =>
        prev.map((entry, index) => (index === 0 ? { ...entry, resultsCount: data.results.length } : entry)),
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

  const sortedResults = [...results].sort((a, b) => {
    let comparison = 0

    switch (sortBy) {
      case "rank":
        comparison = a.rank - b.rank
        break
      case "date":
        comparison = new Date(b.date).getTime() - new Date(a.date).getTime()
        break
      case "domain":
        comparison = a.domain.localeCompare(b.domain)
        break
    }

    return sortOrder === "desc" ? -comparison : comparison
  })

  const exportToCSV = () => {
    const headers = ["Rank", "Snippet", "Domain", "Date", "Link", "Relevance Score"]
    const csvContent = [
      headers.join(","),
      ...results.map((result) =>
        [
          result.rank,
          `"${result.snippet.replace(/"/g, '""')}"`,
          result.domain,
          result.date,
          result.link,
          result.relevanceScore,
        ].join(","),
      ),
    ].join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `vc-search-results-${new Date().toISOString().split("T")[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 text-center">VC Analyst Search Tool</h1>
          <p className="text-gray-600 text-center mt-2">Find promising ex-employees with AI-powered X-ray queries</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {/* Error Display */}
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">{error}</div>}

        {/* Form Section */}
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
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  formErrors.company ? "border-red-500 bg-red-50" : "border-gray-300"
                }`}
                value={searchParams.company}
                onChange={(e) => {
                  setSearchParams((prev) => ({ ...prev, company: e.target.value }))
                  if (formErrors.company && e.target.value.trim()) {
                    setFormErrors((prev) => ({ ...prev, company: false }))
                  }
                }}
              />
              {formErrors.company && <p className="text-red-500 text-xs mt-1">Company name is required</p>}
            </div>

            {/* Role(s) */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Role(s) <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                placeholder="e.g., Senior Engineer, Product Manager"
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  formErrors.roles ? "border-red-500 bg-red-50" : "border-gray-300"
                }`}
                value={searchParams.roles}
                onChange={(e) => {
                  setSearchParams((prev) => ({ ...prev, roles: e.target.value }))
                  if (formErrors.roles && e.target.value.trim()) {
                    setFormErrors((prev) => ({ ...prev, roles: false }))
                  }
                }}
              />
              {formErrors.roles && <p className="text-red-500 text-xs mt-1">Role(s) are required</p>}
            </div>

            {/* Seniority */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Seniority{" "}
                <span style={{ color: "#f59e0b" }} className="text-xs font-normal">
                  (recommended)
                </span>
              </label>
              <Select
                value={searchParams.seniority}
                onValueChange={(value: string) => setSearchParams((prev) => ({ ...prev, seniority: value }))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select seniority" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Junior">Junior</SelectItem>
                  <SelectItem value="Mid">Mid</SelectItem>
                  <SelectItem value="Senior">Senior</SelectItem>
                  <SelectItem value="Director">Director</SelectItem>
                  <SelectItem value="VP">VP</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Quit Window */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Quit Window <span className="text-red-500">*</span>
              </label>
              <Select
                value={searchParams.quitWindow}
                onValueChange={(value: string) => {
                  setSearchParams((prev) => ({ ...prev, quitWindow: value }))
                  if (formErrors.quitWindow && value.trim()) {
                    setFormErrors((prev) => ({ ...prev, quitWindow: false }))
                  }
                }}
              >
                <SelectTrigger className={`w-full ${formErrors.quitWindow ? "border-red-500 bg-red-50" : ""}`}>
                  <SelectValue placeholder="Select time window" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Last 3 months">Last 3 months</SelectItem>
                  <SelectItem value="6 months">6 months</SelectItem>
                  <SelectItem value="1 year">1 year</SelectItem>
                  <SelectItem value="2 years">2 years</SelectItem>
                </SelectContent>
              </Select>
              {formErrors.quitWindow && <p className="text-red-500 text-xs mt-1">Quit window is required</p>}
            </div>

            {/* Geography */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Geography</label>
              <input
                type="text"
                placeholder="e.g., India, San Francisco"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.geography}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, geography: e.target.value }))}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {/* Include Keywords */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Include Keywords</label>
              <textarea
                placeholder="e.g., AI, ML, cloud"
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.includeKeywords}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, includeKeywords: e.target.value }))}
              />
            </div>

            {/* Exclude Keywords */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Exclude Keywords</label>
              <textarea
                placeholder="e.g., intern, junior"
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchParams.excludeKeywords}
                onChange={(e) => setSearchParams((prev) => ({ ...prev, excludeKeywords: e.target.value }))}
              />
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800">Pro Tip</h3>
                <div className="mt-2 text-sm text-blue-700">
                  <p>
                    <span className="text-red-500">*</span> Required fields ensure high-quality query generation.{" "}
                    <span style={{ color: "#f59e0b" }} className="font-medium">
                      (recommended)
                    </span>{" "}
                    fields help improve search relevance.
                  </p>
                </div>
              </div>
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
                Generating...
              </>
            ) : (
              "🔘 Generate X-ray Queries"
            )}
          </button>
        </div>

        {/* X-ray Query Output Section */}
        {generatedQueries.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-semibold mb-4">🧾 Generated X-ray Queries</h2>
            <p className="text-gray-600 mb-4">Copy and paste these queries into Google Search</p>

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

            {/* GPT Prompt Audit Section */}
            {lastUsedPrompt && (
              <div className="mt-4 border-t pt-4">
                <button
                  onClick={() => setShowGptPrompt(!showGptPrompt)}
                  className="flex items-center text-sm text-blue-600 hover:text-blue-800 mb-2"
                >
                  {showGptPrompt ? <ChevronUp className="h-4 w-4 mr-1" /> : <ChevronDown className="h-4 w-4 mr-1" />}
                  View GPT Prompt Used
                </button>

                {showGptPrompt && (
                  <div className="bg-gray-50 border rounded-lg p-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Generated Prompt:</h4>
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap overflow-x-auto bg-white p-3 rounded border">
                      {lastUsedPrompt}
                    </pre>
                  </div>
                )}
              </div>
            )}

            <button
              onClick={handleSearch}
              disabled={isSearching}
              className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isSearching ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />🔍 Execute Search
                </>
              )}
            </button>
          </div>
        )}

        {/* Results Dashboard Section */}
        {results.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-xl font-semibold">🔍 Search Results</h2>
                <p className="text-gray-600">Found {results.length} potential candidates</p>
              </div>
              <div className="flex gap-2">
                <Select value={sortBy} onValueChange={(value: string) => setSortBy(value as "rank" | "date" | "domain")}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rank">✅ Rank</SelectItem>
                    <SelectItem value="date">📅 Date</SelectItem>
                    <SelectItem value="domain">🌐 Domain</SelectItem>
                  </SelectContent>
                </Select>
                <button
                  onClick={exportToCSV}
                  className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 flex items-center"
                >
                  <Download className="mr-2 h-4 w-4" />🔁 Export CSV
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-300 px-4 py-2 text-left">Rank</th>
                    <th className="border border-gray-300 px-4 py-2 text-left">Snippet</th>
                    <th className="border border-gray-300 px-4 py-2 text-left">Domain</th>
                    <th className="border border-gray-300 px-4 py-2 text-left">Date</th>
                    <th className="border border-gray-300 px-4 py-2 text-left">Link</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.length === 0 ? (
                    <tr><td colSpan={5} className="text-center py-8 text-gray-500">No results found.</td></tr>
                  ) : (
                    sortedResults.map((result) => (
                    <tr key={result.rank} className="hover:bg-gray-50">
                      <td className="border border-gray-300 px-4 py-2 font-medium">{result.rank}</td>
                      <td className="border border-gray-300 px-4 py-2 max-w-md">
                        <p className="line-clamp-2">{result.snippet}</p>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {result.rawGoogleResult && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-200 text-gray-800 border border-gray-400" title="This is a raw Google result. No AI extraction.">Raw Google Result</span>
                            )}
                            {result.llmFuzzyMatch && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-300" title="AI marked this as a fuzzy or possible match.">Fuzzy/Possible</span>
                            )}
                            {result.llmUncertain && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-800 border border-orange-300" title="AI was uncertain about this match.">Uncertain</span>
                            )}
                            {typeof result.llmScore === 'number' && (
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${result.llmScore >= 9 ? 'bg-green-500 text-white' : result.llmScore >= 7 ? 'bg-green-200 text-green-900' : 'bg-gray-200 text-gray-700'}`} title={`AI confidence score: ${result.llmScore}`}>LLM Score: {result.llmScore}</span>
                            )}
                            {result.llmQuitStatus && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800" title="AI-inferred quit status.">{result.llmQuitStatus}</span>
                            )}
                            {typeof result.llmCompanyMatch === 'boolean' && (
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${result.llmCompanyMatch ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'}`} title={result.llmCompanyMatch ? 'AI matched company' : 'AI did not match company'}>{result.llmCompanyMatch ? 'Company Match' : 'No Match'}</span>
                            )}
                            {result.llmMatchedCompanyName && (
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${result.llmCompanyNameConfidence !== undefined && result.llmCompanyNameConfidence < 8 ? 'bg-yellow-100 text-yellow-800 border border-yellow-300' : 'bg-green-100 text-green-800 border border-green-300'}`}
                                title={result.llmCompanyNameRationale || 'AI-inferred company normalization'}
                              >
                                Company: {result.llmMatchedCompanyName} (Conf: {result.llmCompanyNameConfidence ?? '?'})
                              </span>
                            )}
                            {result.llmConfidence !== undefined && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800" title="AI confidence (0-10)">Confidence: {result.llmConfidence}</span>
                            )}
                            {result.llmReason && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-700 border border-gray-300" title={result.llmReason}>Rationale</span>
                            )}
                            {result.rawGoogleResult && !result.llmFuzzyMatch && !result.llmUncertain && result.llmScore === undefined && !result.llmQuitStatus && result.llmCompanyMatch === undefined && !result.llmMatchedCompanyName && (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-700 border border-gray-300">No LLM info available</span>
                            )}
                          </div>
                      </td>
                      <td className="border border-gray-300 px-4 py-2">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {result.domain}
                        </span>
                      </td>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">{result.date}</td>
                      <td className="border border-gray-300 px-4 py-2">
                        <a
                          href={result.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800"
                        >
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      </td>
                    </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* If all results are rawGoogleResult, show a user-friendly message */}
        {results.length > 0 && results.every(r => r.rawGoogleResult) && (
          <div className="text-center text-sm text-gray-500 mt-4">These are raw Google results. AI extraction was not possible for these entries.</div>
        )}

        {/* Audit Log Section */}
        {auditLog.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <button
              onClick={() => setShowAuditLog(!showAuditLog)}
              className="flex items-center justify-between w-full text-left"
            >
              <h2 className="text-xl font-semibold">🧾 Audit Log</h2>
              {showAuditLog ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {showAuditLog && (
              <div className="mt-4 space-y-4">
                {auditLog.map((entry, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <p className="font-medium">Search #{auditLog.length - index}</p>
                        <p className="text-sm text-gray-600">{entry.resultsCount} results found</p>
                      </div>
                      <span className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleString()}</span>
                    </div>
                    <details className="text-sm">
                      <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                        View generated prompt
                      </summary>
                      <div className="mt-2 p-2 bg-gray-50 rounded">
                        <pre className="text-xs overflow-x-auto whitespace-pre-wrap">{entry.prompt}</pre>
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
