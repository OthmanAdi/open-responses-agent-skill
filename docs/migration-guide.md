# Migration Guide: Chat Completion to Open Responses API

This guide helps you migrate from the Chat Completion API to the Open Responses API for building autonomous agents.

## Why Migrate?

The Chat Completion API was designed for conversational AI, not autonomous agents. The Open Responses API provides:

| Feature | Chat Completion | Open Responses |
|---------|-----------------|----------------|
| Endpoint | `/v1/chat/completions` | `/v1/responses` (unified) |
| Input format | `messages: [...]` array | `input: "..."` + `instructions: "..."` |
| Tool execution | Manual loop required | Automatic sub-agent loop |
| Reasoning access | Not available | content/summary/encrypted |
| Streaming | Raw tokens | Semantic events |
| Multi-step tasks | Multiple API calls | Single request |
| Provider selection | Separate endpoints | Model suffix (:groq, :together, etc.) |

## Quick Migration

### Before (Chat Completion)
```typescript
const response = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`,
  },
  body: JSON.stringify({
    model: 'gpt-4',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Search for sales data and create a report.' },
    ],
    tools: [...],
  }),
});

// Then manually handle tool calls in a loop...
```

### After (Open Responses - Using OpenAI SDK)
```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  instructions: "You are a helpful assistant.",
  input: "Search for sales data and create a report.",
  tools: [...],
  tool_choice: "auto",
});

// Response contains complete execution trace!
console.log(response.output_text);
for (const item of response.output) {
  console.log(item.type, item);
}
```

## Step-by-Step Migration

### Step 1: Use OpenAI SDK with Custom Base URL

```diff
- const response = await fetch('https://api.openai.com/v1/chat/completions', {...});
+ import OpenAI from "openai";
+ const client = new OpenAI({
+   baseURL: "https://router.huggingface.co/v1",
+   apiKey: process.env.HF_TOKEN,
+ });
```

### Step 2: Update Model Format with Provider Suffix

```diff
- model: 'gpt-4',
+ model: 'moonshotai/Kimi-K2-Instruct-0905:groq',
```

### Step 3: Convert Messages to Instructions + Input

```diff
- messages: [
-   { role: 'system', content: systemPrompt },
-   { role: 'user', content: userInput },
- ],
+ instructions: systemPrompt,
+ input: userInput,
```

### Step 4: Update Tool Definition Format

Tools are defined at TOP LEVEL - NOT nested in `function`:

```diff
// Old format (nested)
- {
-   type: "function",
-   function: {
-     name: "search",
-     description: "Search for info",
-     parameters: {...}
-   }
- }

// New format (top level)
+ {
+   type: "function",
+   name: "search",
+   description: "Search for info",
+   parameters: {...}
+ }
```

### Step 5: Update Response Handling

```diff
- // Chat Completion response
- const message = response.choices[0].message;
- if (message.tool_calls) {
-   // Handle tool calls manually in a loop
- }

+ // Open Responses response - use output (NOT items)
+ for (const item of response.output) {
+   switch (item.type) {
+     case 'reasoning':
+       console.log('Thinking:', item.content || item.summary);
+       break;
+     case 'function_call':  // NOT tool_call
+       console.log('Called:', item.name);
+       break;
+     case 'function_call_output':  // NOT tool_result
+       console.log('Result:', item.output);
+       break;
+     case 'message':
+       console.log('Response:', item.content);
+       break;
+   }
+ }
+
+ // Or use the convenience helper
+ console.log(response.output_text);
```

### Step 6: Remove Manual Tool Loop

The biggest change is removing the manual tool execution loop. With Open Responses:

**Before:** You had to implement the loop yourself
```typescript
while (hasToolCalls) {
  // 1. Send request
  // 2. Check for tool calls
  // 3. Execute tools
  // 4. Send results back
  // 5. Repeat until done
}
```

**After:** The API handles the loop for you
```typescript
const response = await client.responses.create({ ... });
// Response already contains the complete execution trace
```

## Response Structure Comparison

### Chat Completion Response
```json
{
  "id": "chatcmpl-abc123",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Final response",
      "tool_calls": [...]
    },
    "finish_reason": "stop"
  }],
  "usage": { ... }
}
```

### Open Responses Response
```json
{
  "id": "resp_abc123",
  "model": "moonshotai/Kimi-K2-Instruct-0905",
  "output": [
    { "type": "reasoning", "content": "Let me think..." },
    { "type": "function_call", "name": "search", "arguments": {...} },
    { "type": "function_call_output", "output": "Found data..." },
    { "type": "message", "content": "Final response" }
  ],
  "output_text": "Final response",
  "usage": {
    "input_tokens": 100,
    "output_tokens": 200
  }
}
```

## Provider Migration

The Open Responses API uses a SINGLE unified endpoint. Provider selection is via model suffix:

```typescript
// SINGLE endpoint for all providers
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

// Select provider via model suffix
const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",      // Groq
  // model: "meta-llama/Llama-3.1-70B-Instruct:together", // Together AI
  // model: "meta-llama/Llama-3.1-70B-Instruct:nebius",   // Nebius
  // model: "meta-llama/Llama-3.1-70B-Instruct:auto",     // Auto
  instructions: "You are a helpful assistant.",
  input: "Your task here",
});
```

## Handling Reasoning

A key benefit of Open Responses is access to agent reasoning:

```typescript
function getReasoning(item): string {
  // Open weight models (via Groq/Together/Nebius)
  if (item.content && !item.encrypted_content) {
    return item.content; // Full raw reasoning
  }

  // Some proprietary models
  if (item.summary) {
    return item.summary; // Sanitized summary
  }

  // Most proprietary models
  if (item.encrypted_content) {
    return '[Encrypted - not accessible]';
  }

  return '';
}
```

Control reasoning depth with:
```typescript
reasoning: { effort: "low" | "medium" | "high" }
```

## Common Migration Issues

### Issue 1: System Prompt
**Problem:** Where does the system prompt go?

**Solution:** Use the `instructions` field.

### Issue 2: Response Field
**Problem:** Can't find `items` in response.

**Solution:** The field is `output`, not `items`. Also use `output_text` for convenience.

### Issue 3: Tool Call Types
**Problem:** Expecting `tool_call` and `tool_result`.

**Solution:** The types are `function_call` and `function_call_output`.

## Checklist

- [ ] Using OpenAI SDK with `baseURL: "https://router.huggingface.co/v1"`
- [ ] Model has provider suffix (e.g., `:groq`, `:together`)
- [ ] Using `instructions` field for system prompt
- [ ] Using `input` field for user request
- [ ] Tools defined at top level (not nested in `function`)
- [ ] Using `response.output` (not `items`)
- [ ] Using `response.output_text` convenience helper
- [ ] Handling `function_call` and `function_call_output` types
- [ ] Removed manual tool execution loop
- [ ] Added reasoning visibility handling
- [ ] Using HF_TOKEN environment variable

## Need Help?

- See `examples/` for complete implementations in TypeScript, Python, and Rust
- See `templates/` for production-ready starter code
- See `docs/provider-comparison.md` for provider-specific details
- HuggingFace Docs: https://huggingface.co/docs/inference-providers/en/guides/responses-api
- Open Responses Spec: https://www.openresponses.org/specification
