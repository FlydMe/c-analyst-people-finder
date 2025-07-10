import { type NextRequest, NextResponse } from "next/server"

interface SearchResult {
  rank: number
  snippet: string
  domain: string
  date: string
  link: string
  relevanceScore: number
}

// Mock search function with more realistic results
async function mockGoogleSearch(query: string, index: number): Promise<SearchResult[]> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 300))

  const mockResults = [
    {
      rank: index * 3 + 1,
      snippet:
        "Ravi Kumar – left Google, now in AI startup. Former Senior Engineer with 6 years experience in distributed systems.",
      domain: "LinkedIn",
      date: "May 2024",
      link: "https://linkedin.com/in/ravi-kumar-swe",
      relevanceScore: 8.5,
    },
    {
      rank: index * 3 + 2,
      snippet:
        "Ex-Google Staff Engineer in Bangalore looking for new opportunities. Specialized in ML infrastructure and cloud platforms.",
      domain: "GitHub",
      date: "April 2024",
      link: "https://github.com/ex-googler-ml",
      relevanceScore: 7.8,
    },
    {
      rank: index * 3 + 3,
      snippet:
        "Sarah Chen announces departure from Google after 5 years. Senior Product Manager now exploring startup opportunities.",
      domain: "Twitter",
      date: "March 2024",
      link: "https://twitter.com/sarahchen/status/123456789",
      relevanceScore: 7.2,
    },
  ]

  // Return 1-3 random results per query
  const numResults = Math.floor(Math.random() * 3) + 1
  return mockResults.slice(0, numResults)
}

export async function POST(request: NextRequest) {
  try {
    const { queries, searchParams } = await request.json()

    const allResults: SearchResult[] = []
    const seenLinks = new Set<string>()

    // Execute each query
    for (let i = 0; i < queries.length; i++) {
      const query = queries[i]
      try {
        const searchResults = await mockGoogleSearch(query, i)

        for (const result of searchResults) {
          // Skip duplicates
          if (seenLinks.has(result.link)) continue
          seenLinks.add(result.link)

          allResults.push(result)
        }
      } catch (queryError) {
        console.error(`Error executing query "${query}":`, queryError)
        // Continue with other queries
      }
    }

    // Sort by relevance score
    allResults.sort((a, b) => b.relevanceScore - a.relevanceScore)

    // Update ranks after sorting
    for (let i = 0; i < allResults.length; i++) {
      allResults[i].rank = i + 1
    }

    return NextResponse.json({
      results: allResults,
    })
  } catch (error) {
    console.error("Search execution error:", error)
    return NextResponse.json({ error: "Failed to execute search" }, { status: 500 })
  }
}
