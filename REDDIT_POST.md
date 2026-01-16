# Reddit Post

## Title
"I built a Claude Code skill that eliminates the manual tool loop in AI agents (Open Responses API)"

## Body

Hey everyone,

I just open sourced a Claude Code skill that solves something that's been bugging me for a while: **the manual tool execution loop in autonomous agents**.

### The Problem

If you've built AI agents before, you know the drill:
1. Send request to LLM
2. LLM wants to use a tool
3. Execute tool manually
4. Send result back to LLM
5. Repeat steps 2-4 until done

This loop is repetitive, error-prone, and honestly just annoying to maintain.

### The Solution

The **Open Responses API** (via HuggingFace) handles this automatically. You make ONE request, and the API manages the entire multi-step workflow for you. Plus, you get full visibility into the agent's reasoning process.

What I built is a complete Claude Code skill that makes this super easy to use with:
- TypeScript, Python, and Rust examples
- OpenAI SDK integration (just change the `baseURL`)
- Multi-provider routing (Groq, Together AI, Nebius)
- Complete migration guide from Chat Completion API

### Why This Matters

**Before (Chat Completion):**
```typescript
while (true) {
  const response = await openai.chat.completions.create(...);
  if (response.choices[0].finish_reason === "tool_calls") {
    // Manually execute tools, manage state, loop again...
  } else break;
}
```

**After (Open Responses):**
```typescript
const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  instructions: "You are a helpful assistant.",
  input: "Search and summarize",
  tools: [...]
});
// Done! Complete execution trace in response.output
```

### Built With Love

This is open source (MIT license), and I genuinely want your feedback. Whether you're building production agents or just experimenting, I'd love to hear:
- What works well?
- What could be better?
- What use cases are you tackling?

**GitHub:** https://github.com/OthmanAdi/open-responses-agent-skill

I'm not affiliated with HuggingFace or Anthropic - just someone who got tired of writing the same agentic loop code over and over. Hope this helps someone else!

---

*Ahmad Othman Adi*
