# Open Responses Agent Development Skill

Build autonomous agents using the **Open Responses API** via the HuggingFace Inference Providers router.

## What is Open Responses?

Open Responses is an open-source API standard for autonomous agent development. It provides:

- **Sub-agent loops**: Multi-step workflows in a single request
- **Reasoning visibility**: Access to agent thinking (raw, summary, or encrypted)
- **Semantic streaming**: Structured events instead of raw tokens
- **Provider-agnostic design**: Single endpoint, provider selection via model suffix

## Key Concepts

### Single Unified Endpoint
```
https://router.huggingface.co/v1/responses
```

### Provider Selection via Model Suffix
```
moonshotai/Kimi-K2-Instruct-0905:groq      # Groq (fast)
meta-llama/Llama-3.1-70B-Instruct:together # Together AI
meta-llama/Llama-3.1-70B-Instruct:nebius   # Nebius (EU)
meta-llama/Llama-3.1-70B-Instruct:auto     # Auto selection
```

## Supported Platforms

This skill works with:
- Claude Code
- Cursor (via `.cursor/rules/`)
- OpenCode (via `.opencode/` and `AGENTS.md`)
- Codex (via `.codex/` and `AGENTS.md`)

## Supported Languages

- TypeScript/JavaScript
- Python
- Rust

## Quick Start

### TypeScript (Recommended - Using OpenAI SDK)
```bash
npm install openai
export HF_TOKEN=your-token
```

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

const response = await client.responses.create({
  model: "moonshotai/Kimi-K2-Instruct-0905:groq",
  instructions: "You are a helpful assistant.",
  input: "Explain quantum entanglement in simple terms.",
});

console.log(response.output_text);
```

### Python (Recommended - Using OpenAI SDK)
```bash
pip install openai
export HF_TOKEN=your-token
```

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

response = client.responses.create(
    model="moonshotai/Kimi-K2-Instruct-0905:groq",
    instructions="You are a helpful assistant.",
    input="Explain quantum entanglement in simple terms.",
)

print(response.output_text)
```

## Structure

```
open-responses-agent-dev/
├── .claude-plugin/          # Claude Code plugin configuration
├── .cursor/rules/           # Cursor rules
├── .opencode/               # OpenCode configuration
├── .codex/                  # Codex configuration
├── skills/                  # Skill definition
│   └── open-responses-agent-dev/
│       └── SKILL.md         # Main skill instructions
├── examples/                # Complete examples
│   ├── typescript/          # TypeScript examples
│   ├── python/              # Python examples
│   └── rust/                # Rust examples
├── templates/               # Production-ready templates
│   ├── typescript/          # TypeScript starter
│   └── python/              # Python starter
├── docs/                    # Documentation
│   ├── migration-guide.md   # Chat Completion → Open Responses
│   └── provider-comparison.md # Provider comparison
├── AGENTS.md                # Agent instructions (OpenCode/Codex)
└── README.md                # This file
```

## Examples

### Basic Agent
Simple request with reasoning visibility:
- `examples/typescript/basic-agent.ts`
- `examples/python/basic_agent.py`
- `examples/rust/basic_agent.rs`

### Sub-Agent Loop
Multi-step workflows with tools:
- `examples/typescript/sub-agent-loop.ts`
- `examples/python/sub_agent_loop.py`
- `examples/rust/sub_agent_loop.rs`

### Multi-Provider
Provider switching via model suffix:
- `examples/typescript/multi-provider.ts`
- `examples/python/multi_provider.py`
- `examples/rust/multi_provider.rs`

### Reasoning Visibility
Accessing agent thinking:
- `examples/typescript/reasoning-visibility.ts`
- `examples/python/reasoning_visibility.py`
- `examples/rust/reasoning_visibility.rs`

## Provider Suffixes

| Suffix | Provider | Description | Reasoning |
|--------|----------|-------------|-----------|
| `:groq` | Groq | Fast inference | **RAW** |
| `:together` | Together AI | Open weight specialist | **RAW** |
| `:nebius` | Nebius AI | European infrastructure | **RAW** |
| `:auto` | Auto | Automatic selection | Varies |

## Request & Response

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
    { "type": "reasoning", "content": "Let me think..." },
    { "type": "function_call", "name": "search", "arguments": {...} },
    { "type": "function_call_output", "output": "..." },
    { "type": "message", "content": "Final response" }
  ],
  "output_text": "Final response (convenience helper)",
  "usage": { "input_tokens": 100, "output_tokens": 200 }
}
```

## Environment Variables

```bash
# Required
export HF_TOKEN=hf_...

# Optional
export MODEL=moonshotai/Kimi-K2-Instruct-0905:groq
export REASONING_EFFORT=medium  # low, medium, high
```

## Reasoning Visibility

- **RAW**: `content` field (open weight models via Groq/Together/Nebius)
- **Summary**: `summary` field (some proprietary)
- **Encrypted**: `encrypted_content` field (most proprietary)

## Documentation

- [Migration Guide](docs/migration-guide.md) - Migrate from Chat Completion API
- [Provider Comparison](docs/provider-comparison.md) - Compare providers

## Resources

- [HuggingFace Responses API Docs](https://huggingface.co/docs/inference-providers/en/guides/responses-api)
- [Open Responses Specification](https://www.openresponses.org/specification)
- [HuggingFace Blog: Open Responses](https://huggingface.co/blog/open-responses)

## License

MIT

## Author

Ahmad Othman Adi

## Credits

Based on [Open Responses](https://github.com/openresponses/openresponses) specification.
