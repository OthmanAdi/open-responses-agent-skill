# Open Responses Agent Development

## About This Skill
Build autonomous agents using the Open Responses API via the HuggingFace Inference Providers router.

## Core Knowledge

### Single Unified Endpoint
All requests go to ONE endpoint:
```
https://router.huggingface.co/v1/responses
```

Provider selection is done via MODEL SUFFIX (not separate URLs).

### Provider Selection via Model Suffix
```
moonshotai/Kimi-K2-Instruct-0905:groq      # Groq (fast inference)
meta-llama/Llama-3.1-70B-Instruct:together # Together AI
meta-llama/Llama-3.1-70B-Instruct:nebius   # Nebius (EU infrastructure)
meta-llama/Llama-3.1-70B-Instruct:auto     # Auto selection
```

### Request Structure
```json
{
  "model": "moonshotai/Kimi-K2-Instruct-0905:groq",
  "instructions": "You are a helpful assistant.",
  "input": "User's task",
  "tools": [...],
  "tool_choice": "auto",
  "reasoning": { "effort": "medium" }
}
```

### Response Structure
```json
{
  "id": "resp_abc123",
  "model": "moonshotai/Kimi-K2-Instruct-0905",
  "output": [
    { "type": "reasoning", "content": "..." },
    { "type": "function_call", "name": "...", "arguments": {...} },
    { "type": "function_call_output", "output": "..." },
    { "type": "message", "content": "..." }
  ],
  "output_text": "convenience helper",
  "usage": { "input_tokens": 100, "output_tokens": 200 }
}
```

### Tool Definition Format
Tools are defined at TOP LEVEL - NOT nested in `function`:
```json
{
  "type": "function",
  "name": "search",
  "description": "Search for information",
  "parameters": { ... }
}
```

### Reasoning Visibility
| Level | Field | Providers |
|-------|-------|-----------|
| RAW | `content` | Groq, Together, Nebius (open weight) |
| SUMMARY | `summary` | Some proprietary models |
| ENCRYPTED | `encrypted_content` | Most proprietary models |

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

## Best Practices
1. Always use unified endpoint: `https://router.huggingface.co/v1/responses`
2. Select provider via model suffix: `model-id:provider`
3. Use `instructions` field for system prompt
4. Handle all output types: reasoning, function_call, function_call_output, message
5. Use `response.output` (NOT `items`) and `response.output_text` helper
6. Use HF_TOKEN environment variable
7. Use OpenAI SDK with custom `base_url` for easiest integration

## Environment Variables
- `HF_TOKEN`: HuggingFace token (required)
- `MODEL`: Model with provider suffix (optional)
- `REASONING_EFFORT`: low, medium, or high (optional)

## Supported Languages
- TypeScript/JavaScript
- Python

See `examples/` and `templates/` directories for implementation patterns.
