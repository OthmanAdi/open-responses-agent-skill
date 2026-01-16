# Provider Comparison - Open Responses API

This document compares the providers available via the HuggingFace router for the Open Responses API.

## Key Concept: Single Endpoint, Multiple Providers

All requests go to ONE unified endpoint:
```
https://router.huggingface.co/v1/responses
```

Provider selection is done via MODEL SUFFIX, not separate URLs.

## Quick Comparison

| Feature | Groq | Together AI | Nebius | Auto |
|---------|------|-------------|--------|------|
| Suffix | `:groq` | `:together` | `:nebius` | `:auto` |
| Reasoning | **RAW** | **RAW** | **RAW** | Varies |
| Speed | Very Fast | Fast | Fast | Varies |
| Region | US | US | **EU** | Varies |

## Provider Details

### Groq (`:groq`)
**Model Example:** `moonshotai/Kimi-K2-Instruct-0905:groq`

**Key Characteristics:**
- Extremely fast inference (purpose-built hardware)
- **Full raw reasoning visibility**
- Open weight models
- Competitive pricing

**Best For:**
- Low-latency applications
- Real-time agent interactions
- Development and testing

---

### Together AI (`:together`)
**Model Example:** `meta-llama/Llama-3.1-70B-Instruct:together`

**Key Characteristics:**
- Specialized in open weight models
- **Full raw reasoning visibility**
- Optimized inference infrastructure
- Large model selection

**Best For:**
- Production use of open weight models
- When you need raw reasoning traces
- Cost-effective high-performance inference

---

### Nebius AI (`:nebius`)
**Model Example:** `meta-llama/Llama-3.1-70B-Instruct:nebius`

**Key Characteristics:**
- **European infrastructure**
- **Full raw reasoning visibility**
- GDPR-friendly
- Data sovereignty compliance

**Best For:**
- European deployments
- Data sovereignty requirements
- GDPR compliance needs

---

### Auto (`:auto`)
**Model Example:** `meta-llama/Llama-3.1-70B-Instruct:auto`

**Key Characteristics:**
- Automatic provider selection
- Router chooses best available provider
- Load balancing
- Fallback handling

**Best For:**
- When you don't need a specific provider
- High availability requirements
- Cost optimization

---

## Reasoning Visibility Comparison

### RAW (Full Transparency)
Available with: **Groq, Together AI, Nebius** (open weight models)

```json
{
  "type": "reasoning",
  "content": "Let me think about this step by step...\n\nFirst, I need to analyze the user's request...\n\nI should search for the relevant data...\n\nBased on my findings, I'll create a comprehensive report..."
}
```

You get the complete, unfiltered thinking process of the model.

### Control Reasoning Depth
```json
{
  "reasoning": { "effort": "low" | "medium" | "high" }
}
```

- `low` - Minimal reasoning, faster responses
- `medium` - Balanced reasoning depth (default)
- `high` - Deep reasoning, more thorough analysis

## Code for Provider Selection

### TypeScript (Recommended)
```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

// Provider is specified via model suffix
const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",  // Change suffix to switch
  instructions: "You are a helpful assistant.",
  input: "Your task here",
  reasoning: { effort: "medium" },
});

console.log(response.output_text);
```

### Python (Recommended)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

# Provider is specified via model suffix
response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",  # Change suffix to switch
    instructions="You are a helpful assistant.",
    input="Your task here",
    reasoning={"effort": "medium"},
)

print(response.output_text)
```

## Choosing a Provider

### For Maximum Speed
Choose **Groq** (`:groq`) - Purpose-built hardware for fast inference.

### For Maximum Transparency
Any of **Groq**, **Together AI**, or **Nebius** - All expose raw reasoning traces.

### For European Compliance
Choose **Nebius** (`:nebius`) - European infrastructure, GDPR-friendly.

### For Automatic Optimization
Choose **Auto** (`:auto`) - Let the router choose the best provider.

## Provider Switching Example

Switching providers is trivial - just change the model suffix:

```typescript
// Groq - fast inference
model: "moonshotai/Kimi-K2-Instruct-0905:groq"

// Together AI - open weight specialist
model: "meta-llama/Llama-3.1-70B-Instruct:together"

// Nebius - European infrastructure
model: "meta-llama/Llama-3.1-70B-Instruct:nebius"

// Auto - automatic selection
model: "meta-llama/Llama-3.1-70B-Instruct:auto"
```

## Recommendation

For most autonomous agent development:

1. **Development:** Groq (`:groq`) - Fast iteration, full visibility
2. **Production:** Together AI (`:together`) or Nebius (`:nebius`)
3. **EU Compliance:** Nebius (`:nebius`)
4. **High Availability:** Auto (`:auto`)

The unified endpoint makes it trivial to switch between providers, so you can start with one and migrate to another as needs change.

## Resources

- [HuggingFace Responses API Docs](https://huggingface.co/docs/inference-providers/en/guides/responses-api)
- [Open Responses Specification](https://www.openresponses.org/specification)
- [HuggingFace Blog: Open Responses](https://huggingface.co/blog/open-responses)
