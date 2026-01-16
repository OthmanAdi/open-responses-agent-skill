# Open Responses Agent Development - Codex Instructions

## Context
You are working on building autonomous agents using the Open Responses API via the HuggingFace Inference Providers router.

## Critical Rules

### 1. SINGLE UNIFIED ENDPOINT
```
CORRECT:   POST https://router.huggingface.co/v1/responses
INCORRECT: Separate provider endpoints
```

### 2. Provider Selection via Model Suffix
```
model: "moonshotai/Kimi-K2-Instruct-0905:groq"    # Groq provider
model: "meta-llama/Llama-3.1-70B-Instruct:together" # Together AI
model: "meta-llama/Llama-3.1-70B-Instruct:nebius"   # Nebius (EU)
model: "meta-llama/Llama-3.1-70B-Instruct:auto"     # Auto selection
```

### 3. Request Format
```json
{
  "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
  "instructions": "You are a helpful assistant.",
  "input": "user task",
  "tools": [...],
  "tool_choice": "auto",
  "reasoning": { "effort": "medium" }
}
```

### 4. Response Structure
```json
{
  "id": "resp_...",
  "model": "...",
  "output": [...],
  "output_text": "convenience helper",
  "usage": { "input_tokens": 100, "output_tokens": 200 }
}
```

### 5. Output Item Types
Handle ALL these types (NOT tool_call/tool_result):
- `reasoning`: Agent's thinking process
- `function_call`: Tool call request (name, arguments)
- `function_call_output`: Tool execution result (output)
- `message`: Final response to user

### 6. Tool Definition Format
Tools at TOP LEVEL - NOT nested in `function`:
```json
{
  "type": "function",
  "name": "search",
  "description": "Search for information",
  "parameters": { ... }
}
```

### 7. Reasoning Visibility
Check for reasoning in this order:
1. `content` - Raw traces (open weight models via Groq/Together/Nebius)
2. `summary` - Sanitized summary (some proprietary)
3. `encrypted_content` - Not accessible (most proprietary)

Control with: `"reasoning": { "effort": "low" | "medium" | "high" }`

## SDK Usage (Recommended)

### TypeScript
```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  instructions: "You are a helpful assistant.",
  input: "Your task here",
});

console.log(response.output_text);
for (const item of response.output) {
  console.log(item.type, item);
}
```

### Python
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    instructions="You are a helpful assistant.",
    input="Your task here",
)

print(response.output_text)
for item in response.output:
    print(item.type, item)
```

## Anti-Patterns to Avoid
- Using separate provider endpoints (use unified router)
- Using `response.items` (use `response.output`)
- Using `tool_call`/`tool_result` (use `function_call`/`function_call_output`)
- Nesting tool definition in `function` object
- Missing `instructions` field in request
- Using fake headers like `OpenResponses-Version`
- Not using model suffix for provider selection

## Resources
- HuggingFace Docs: https://huggingface.co/docs/inference-providers/en/guides/responses-api
- Open Responses Spec: https://www.openresponses.org/specification
- Examples: `examples/typescript/`, `examples/python/`, `examples/rust/`
