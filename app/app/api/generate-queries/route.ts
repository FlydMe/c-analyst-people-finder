import { type NextRequest, NextResponse } from "next/server"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

const SYSTEM_PROMPT = `You are QueryGen, an elite AI assistant for venture capital analysts. Your job is to generate exactly 7 highly optimized, diverse Google X-ray search queries to help identify promising ex-employees (e.g., senior engineers who left a company recently), with a **primary focus on LinkedIn**.

You must:
- Use advanced Boolean logic (AND, OR, quotes, -) and proper phrasing.
- Include multiple variants of each:
  - Roles (e.g. “Software Engineer”, “Developer”, “SDE”)
  - Seniority (e.g. “Senior”, “Lead”, “Principal”)
  - Quit language (e.g. “ex-Company”, “formerly at Company”, “left Company”, “quit Company”, “resigned from Company”)
- Simulate recency using the quit window context.
- Apply keyword filters using `include_keywords` and `exclude_keywords`.
- Focus almost entirely on LinkedIn (`site:linkedin.com/in` or `site:linkedin.com/pub`), but vary phrasing, structure, and keywords.
- Always output exactly 7 one-line Google X-ray queries—no commentary, no numbering, no formatting—just the raw query strings.

You will be given a JSON input with these fields:
- `company`: the company they left (e.g. "Stripe")
- `roles`: list of titles (e.g. ["Software Engineer", "Backend Developer"])
- `seniority`: level (e.g. "Senior")
- `quit_window`: natural language string (e.g. "past 6 months")
- `geography`: location focus (e.g. "San Francisco OR Bay Area")
- `include_keywords`: skills or topics to filter by
- `exclude_keywords`: negative keywords like "intern", "freelancer"

The goal: return 7 diverse, production-ready Google search strings to paste directly into Google and find LinkedIn profiles of relevant ex-employees.`

export async function POST(request: NextRequest) {
  try {
    const searchParams = await request.json()
    const userPrompt = `JSON Input:\n${JSON.stringify(searchParams, null, 2)}\n\nOutput: 7 Google X-ray queries (raw, ready to paste into Google)`

    const { text } = await generateText({
      model: openai("gpt-3.5-turbo"),
      prompt: `${SYSTEM_PROMPT}\n\n${userPrompt}`,
      temperature: 0.7,
    })

    // Parse the response: expect 7 raw queries, one per line
    let queries: string[] = text
      .split(/\r?\n/)
      .map(q => q.trim())
      .filter(q => q.length > 0 && !q.startsWith("#"))
      .slice(0, 7)

    // Fallback: create 7 queries manually if AI fails
    if (queries.length !== 7) {
      const c = searchParams.company || "Company"
      const roles = (searchParams.roles && Array.isArray(searchParams.roles)) ? searchParams.roles : ["Engineer"]
      const seniority = searchParams.seniority || "Senior"
      const quit = searchParams.quit_window || "recently"
      const geo = searchParams.geography ? ` AND (${searchParams.geography})` : ""
      const include = searchParams.include_keywords && searchParams.include_keywords.length ? ` AND (${searchParams.include_keywords.join(" OR ")})` : ""
      const exclude = searchParams.exclude_keywords && searchParams.exclude_keywords.length ? searchParams.exclude_keywords.map((k: string) => `-${k}`).join(" ") : ""
      queries = [
        `site:linkedin.com/in ("${roles[0]}" OR "${roles[1] || roles[0]}") AND ("ex-${c}" OR "left ${c}") AND ("${seniority}" OR "Lead") AND (${quit})${geo}${include} ${exclude}`.trim(),
        `site:linkedin.com/in ("${c}") AND ("${seniority} ${roles[0]}") AND ("quit" OR "resigned") AND (${quit})${geo}${include} ${exclude}`.trim(),
        `site:linkedin.com/in ("${roles[0]}" OR "Developer") AND ("formerly at ${c}") AND ("${seniority}") AND (${quit})${geo}${include} ${exclude}`.trim(),
        `site:linkedin.com/in ("ex-${c}") AND ("${roles[1] || roles[0]}" OR "Engineer") AND ("recently at ${c}" OR "left in 2024") AND (${quit})${geo}${include} ${exclude}`.trim(),
        `site:linkedin.com/in ("${c}") AND ("SDE" OR "${seniority} Developer") AND ("resigned from ${c}" OR "quit ${c}") AND (${quit})${geo}${include} ${exclude}`.trim(),
        `site:linkedin.com/in ("Backend Engineer" OR "Software Developer") AND ("${c}") AND ("ex-" OR "left") AND (${quit})${geo}${include} ${exclude}`.trim(),
        `site:linkedin.com/in ("Lead Engineer" OR "Staff Developer") AND ("${c}") AND ("ex-" OR "formerly") AND (${quit})${geo}${include} ${exclude}`.trim(),
      ]
    }

    if (!Array.isArray(queries) || queries.length !== 7) {
      throw new Error("Failed to generate exactly 7 queries")
    }

    return NextResponse.json({ queries, prompt: SYSTEM_PROMPT })
  } catch (error) {
    console.error("Query generation error:", error)
    return NextResponse.json({ error: "Failed to generate search queries" }, { status: 500 })
  }
}
