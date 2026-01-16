"""
Multi-Provider Example - Open Responses API

Demonstrates provider routing via model suffixes using the unified
HuggingFace router endpoint. Single codebase, multiple providers.

Usage:
    pip install openai
    export HF_TOKEN=your-token
    python multi_provider.py
"""

import os
import time
from openai import OpenAI


# Configure client with HuggingFace router endpoint
# This SINGLE endpoint routes to different providers via model suffix
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)


# Provider information - providers are specified via model SUFFIX
# e.g., "model-name:groq" or "model-name:together"
PROVIDERS = [
    {
        "suffix": ":groq",
        "name": "Groq",
        "description": "Fast inference provider",
        "example_model": "moonshotai/Kimi-K2-Instruct-0905:groq",
    },
    {
        "suffix": ":together",
        "name": "Together AI",
        "description": "Open weight model specialist",
        "example_model": "meta-llama/Llama-3.1-70B-Instruct:together",
    },
    {
        "suffix": ":nebius",
        "name": "Nebius AI",
        "description": "European infrastructure",
        "example_model": "meta-llama/Llama-3.1-70B-Instruct:nebius",
    },
    {
        "suffix": ":auto",
        "name": "Auto",
        "description": "Automatic provider selection",
        "example_model": "meta-llama/Llama-3.1-70B-Instruct:auto",
    },
]


def create_agent(model: str, input_text: str, instructions: str | None = None):
    """
    Create an agent with a specific model (provider specified via suffix).

    Args:
        model: Model identifier with provider suffix
        input_text: The user's request
        instructions: Optional system prompt

    Returns:
        Response object
    """
    response = client.responses.create(
        model=model,
        instructions=instructions or "You are a helpful assistant.",
        input=input_text,
    )

    return response


def compare_providers(prompt: str, models: list[str]) -> None:
    """
    Compare the same prompt across different providers.

    Args:
        prompt: The prompt to send
        models: List of model identifiers with provider suffixes
    """
    print("\n" + "=" * 70)
    print("MULTI-PROVIDER COMPARISON")
    print("=" * 70)
    print(f"Endpoint: https://router.huggingface.co/v1/responses")
    print(f'Prompt: "{prompt}"')
    print(f"Models: {', '.join(models)}")
    print("=" * 70 + "\n")

    results = []

    for model in models:
        # Extract provider suffix for display
        suffix = model.split(":")[-1] if ":" in model else "default"
        print(f"\n--- Testing {suffix.upper()} ({model}) ---")

        try:
            print("Sending request...")

            start_time = time.time()
            response = create_agent(model, prompt)
            duration = int((time.time() - start_time) * 1000)

            print(f"Response received in {duration}ms")
            in_tokens = response.usage.input_tokens if response.usage else 0
            out_tokens = response.usage.output_tokens if response.usage else 0
            print(f"Tokens: {in_tokens} in / {out_tokens} out")

            # Display reasoning (if available)
            reasoning_items = [i for i in response.output if i.type == "reasoning"]
            if reasoning_items:
                print(f"\nReasoning ({len(reasoning_items)} items):")
                for item in reasoning_items:
                    text = getattr(item, "content", None) or getattr(item, "summary", None) or "[no content]"
                    print(f"  - {text[:150]}{'...' if len(text) > 150 else ''}")

            # Display final response using convenience helper
            print(f"\nResponse:")
            output_text = response.output_text or ""
            print(f"  {output_text[:300]}{'...' if len(output_text) > 300 else ''}")

            results.append({"model": model, "response": response, "duration": duration})

        except Exception as e:
            print(f"Error: {e}")
            results.append({"model": model, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for result in results:
        suffix = result["model"].split(":")[-1] if ":" in result["model"] else "default"
        if "response" in result:
            response = result["response"]
            in_tokens = response.usage.input_tokens if response.usage else 0
            out_tokens = response.usage.output_tokens if response.usage else 0
            total_tokens = in_tokens + out_tokens
            print(f"{suffix:12} | SUCCESS | {result['duration']}ms | {total_tokens} tokens")
        else:
            print(f"{suffix:12} | FAILED  | {result['error']}")


def demonstrate_provider_switching() -> None:
    """Demonstrate provider switching via model suffix."""
    print("\n" + "=" * 70)
    print("PROVIDER SWITCHING DEMONSTRATION")
    print("=" * 70)
    print("\nKey concept: Provider is specified via MODEL SUFFIX")
    print("Endpoint is ALWAYS: https://router.huggingface.co/v1")
    print("=" * 70)

    # Default model - uses Groq provider
    model = os.environ.get("MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq")

    print(f"\nUsing model: {model}")
    suffix = model.split(":")[-1] if ":" in model else "default"
    print(f"Provider (from suffix): {suffix}")

    print(f"\nSending request...")
    response = create_agent(model, "What is 2 + 2? Explain your reasoning.")

    print(f"\nResponse ID: {response.id}")
    print(f"Model: {response.model}")

    # Show output items
    print(f"\nOutput items:")
    for item in response.output:
        match item.type:
            case "reasoning":
                text = getattr(item, "content", None) or getattr(item, "summary", None) or "[no content]"
                print(f"  [REASONING] {text[:100]}...")
            case "message":
                print(f"  [MESSAGE] {getattr(item, 'content', '')}")
            case _:
                print(f"  [{item.type.upper()}]", item)

    # Show convenience helper
    print(f"\n--- Output Text (convenience helper) ---")
    print(response.output_text)

    # Show available providers
    print("\n" + "=" * 70)
    print("AVAILABLE PROVIDERS (via model suffix)")
    print("=" * 70)
    for provider in PROVIDERS:
        print(f"  {provider['suffix']:12} - {provider['name']}: {provider['description']}")
        print(f"               Example: {provider['example_model']}")


def main() -> None:
    """Main execution."""
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        exit(1)

    mode = os.environ.get("MODE", "switch")

    try:
        if mode == "compare":
            # Compare multiple providers
            models = [
                "moonshotai/Kimi-K2-Instruct-0905:groq",
                # Add more models with different provider suffixes to compare
                # "meta-llama/Llama-3.1-70B-Instruct:together",
                # "meta-llama/Llama-3.1-70B-Instruct:nebius",
            ]
            compare_providers("Explain quantum entanglement in one paragraph.", models)
        else:
            # Demonstrate provider switching
            demonstrate_provider_switching()

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
