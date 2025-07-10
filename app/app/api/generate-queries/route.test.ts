import { POST } from "./route"
import { NextRequest } from "next/server"

const mockInput = {
  company: "Stripe",
  roles: ["Software Engineer", "Backend Developer"],
  seniority: "Senior",
  quit_window: "past 6 months",
  geography: "San Francisco OR Bay Area",
  include_keywords: ["Python", "AWS"],
  exclude_keywords: ["intern", "freelancer"]
}

describe("POST /api/generate-queries", () => {
  it("returns exactly 7 LinkedIn-focused queries", async () => {
    const req = {
      json: async () => mockInput
    } as unknown as NextRequest
    const res = await POST(req)
    const data = await res.json()
    expect(Array.isArray(data.queries)).toBe(true)
    expect(data.queries.length).toBe(7)
    data.queries.forEach((q: string) => {
      expect(q).toMatch(/site:linkedin\.com\/in/)
      expect(typeof q).toBe("string")
      expect(q.length).toBeGreaterThan(10)
    })
  })

  it("returns fallback queries if AI fails", async () => {
    // Simulate a broken AI response by patching generateText
    const originalGenerateText = jest.requireActual("ai").generateText
    jest.spyOn(require("ai"), "generateText").mockResolvedValueOnce({ text: "" })
    const req = {
      json: async () => ({ ...mockInput, company: "TestCo" })
    } as unknown as NextRequest
    const res = await POST(req)
    const data = await res.json()
    expect(Array.isArray(data.queries)).toBe(true)
    expect(data.queries.length).toBe(7)
    data.queries.forEach((q: string) => {
      expect(q).toMatch(/site:linkedin\.com\/in/)
    })
    // Restore
    require("ai").generateText = originalGenerateText
  })
}) 