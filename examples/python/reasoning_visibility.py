"""
Reasoning Visibility Example - Open Responses API

Demonstrates how to access and display agent reasoning.
Shows differences between open weight models (raw traces) and
proprietary models (summaries/encrypted).

Usage:
    pip install openai
    export HF_TOKEN=your-token
    python reasoning_visibility.py
"""

import os
from enum import Enum
from openai import OpenAI


# Configure client with HuggingFace router endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)


class ReasoningLevel(Enum):
    """Reasoning visibility levels."""
    RAW = "raw"           # Full thinking traces (open weight)
    SUMMARY = "summary"   # Sanitized summary (some proprietary)
    ENCRYPTED = "encrypted"  # No visibility (most proprietary)
    NONE = "none"         # No reasoning at all


def analyze_reasoning_visibility(response):
    """
    Analyze reasoning visibility for a response.

    Args:
        response: The Open Responses response

    Returns:
        Dictionary with visibility level and details
    """
    reasoning_items = [item for item in response.output if item.type == "reasoning"]

    if not reasoning_items:
        return {
            "level": ReasoningLevel.NONE,
            "reasoning_items": [],
            "total_reasoning_tokens": 0,
            "details": "No reasoning items found in response.",
        }

    # Check what type of reasoning is available
    has_raw_content = any(
        getattr(item, "content", None) and not getattr(item, "encrypted_content", None)
        for item in reasoning_items
    )
    has_encrypted = any(getattr(item, "encrypted_content", None) for item in reasoning_items)
    has_summary = any(getattr(item, "summary", None) for item in reasoning_items)

    # Estimate tokens (rough approximation)
    total_reasoning_tokens = sum(
        len(getattr(item, "content", "") or getattr(item, "summary", "") or "") // 4
        for item in reasoning_items
    )

    if has_raw_content:
        return {
            "level": ReasoningLevel.RAW,
            "reasoning_items": reasoning_items,
            "total_reasoning_tokens": total_reasoning_tokens,
            "details": "Full raw reasoning traces available. This model provides complete transparency.",
        }

    if has_summary:
        return {
            "level": ReasoningLevel.SUMMARY,
            "reasoning_items": reasoning_items,
            "total_reasoning_tokens": total_reasoning_tokens,
            "details": "Summarized reasoning available. Raw traces are not exposed.",
        }

    if has_encrypted:
        return {
            "level": ReasoningLevel.ENCRYPTED,
            "reasoning_items": reasoning_items,
            "total_reasoning_tokens": 0,
            "details": "Reasoning is encrypted and not accessible to the client.",
        }

    return {
        "level": ReasoningLevel.NONE,
        "reasoning_items": reasoning_items,
        "total_reasoning_tokens": 0,
        "details": "Unknown reasoning format.",
    }


def display_reasoning(reasoning_items, level: ReasoningLevel) -> None:
    """
    Pretty print reasoning items.

    Args:
        reasoning_items: List of reasoning items
        level: The reasoning visibility level
    """
    print("\n" + "-" * 60)
    print("REASONING TRACE")
    print("-" * 60)
    print(f"Visibility Level: {level.value.upper()}")
    print(f"Total Items: {len(reasoning_items)}")
    print("-" * 60 + "\n")

    for i, item in enumerate(reasoning_items):
        print(f"[Step {i + 1}]")

        content = getattr(item, "content", None)
        summary = getattr(item, "summary", None)
        encrypted_content = getattr(item, "encrypted_content", None)

        if content:
            print("Type: Raw Trace")
            print("Content:")
            # Format multi-line reasoning nicely
            for line in content.split("\n"):
                print(f"  {line}")

        elif summary:
            print("Type: Summary")
            print(f"Content: {summary}")

        elif encrypted_content:
            print("Type: Encrypted")
            print("Content: [ENCRYPTED - Not accessible]")
            print(f"Encrypted Length: {len(encrypted_content)} chars")

        else:
            print("Type: Unknown")
            print(f"Raw: {item}")

        print()


def create_agent_with_reasoning(
    model: str,
    input_text: str,
    reasoning_effort: str = "medium",
):
    """
    Create agent request with reasoning focus.

    Args:
        model: Model identifier with provider suffix
        input_text: The reasoning-heavy prompt
        reasoning_effort: Reasoning effort level (low, medium, high)

    Returns:
        Response object with reasoning items
    """
    response = client.responses.create(
        model=model,
        instructions="You are a helpful assistant. Show your step-by-step reasoning process.",
        input=input_text,
        reasoning={"effort": reasoning_effort},
    )

    return response


def demonstrate_reasoning_visibility() -> None:
    """Demonstrate reasoning visibility with a reasoning-heavy prompt."""
    # Model with provider suffix - using Groq for fast inference
    model = os.environ.get("MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq")
    reasoning_effort = os.environ.get("REASONING_EFFORT", "medium")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        exit(1)

    # Prompt designed to elicit multi-step reasoning
    reasoning_prompt = """
    Solve this step by step:

    A farmer has 15 chickens and 12 cows. Some of the animals get sick.
    After treating them, the farmer has 80% of the original chickens
    and 75% of the original cows healthy.

    1. How many chickens are healthy?
    2. How many cows are healthy?
    3. What is the total number of healthy animals?
    4. What percentage of all animals are healthy?

    Show all your work and reasoning.
    """

    print("\n" + "=" * 60)
    print("REASONING VISIBILITY DEMONSTRATION")
    print("=" * 60)
    print(f"Endpoint: https://router.huggingface.co/v1/responses")
    print(f"Model: {model}")
    print(f"Reasoning Effort: {reasoning_effort}")
    print("=" * 60)
    print("\nPrompt:")
    print(reasoning_prompt)

    try:
        print("\nSending request...\n")
        response = create_agent_with_reasoning(model, reasoning_prompt, reasoning_effort)

        # Analyze reasoning visibility
        analysis = analyze_reasoning_visibility(response)

        print("=" * 60)
        print("REASONING ANALYSIS")
        print("=" * 60)
        print(f"Response ID: {response.id}")
        print(f"Model: {response.model}")
        print(f"Visibility Level: {analysis['level'].value}")
        print(f"Reasoning Items: {len(analysis['reasoning_items'])}")
        print(f"Est. Reasoning Tokens: ~{analysis['total_reasoning_tokens']}")
        print(f"Details: {analysis['details']}")

        # Display reasoning traces
        if analysis["reasoning_items"]:
            display_reasoning(analysis["reasoning_items"], analysis["level"])

        # Display final answer using convenience helper
        print("-" * 60)
        print("FINAL ANSWER (output_text)")
        print("-" * 60)
        print(response.output_text)

        # Also show individual output items
        print("\n" + "-" * 60)
        print("ALL OUTPUT ITEMS")
        print("-" * 60)
        for i, item in enumerate(response.output):
            print(f"[{i + 1}/{len(response.output)}] Type: {item.type}")

            match item.type:
                case "reasoning":
                    text = getattr(item, "content", None) or getattr(item, "summary", None) or "[encrypted]"
                    print(f"    {text[:150]}{'...' if len(text) > 150 else ''}")
                case "message":
                    content = getattr(item, "content", "") or ""
                    print(f"    {content[:150]}...")
                case _:
                    print(f"    {str(item)[:150]}...")

        # Token usage
        print("\n" + "-" * 60)
        print("TOKEN USAGE")
        print("-" * 60)
        in_tokens = response.usage.input_tokens if response.usage else 0
        out_tokens = response.usage.output_tokens if response.usage else 0
        print(f"Input: {in_tokens}")
        print(f"Output: {out_tokens}")
        print(f"Total: {in_tokens + out_tokens}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def compare_reasoning_across_providers() -> None:
    """Compare reasoning visibility across providers."""
    print("\n" + "=" * 70)
    print("REASONING VISIBILITY COMPARISON")
    print("=" * 70)
    print(f"\nEndpoint: https://router.huggingface.co/v1 (unified)")
    print(f"Provider is specified via MODEL SUFFIX (e.g., :groq, :together)")

    provider_info = [
        (":groq", "Groq", "RAW - Full reasoning traces (open weight models)"),
        (":together", "Together AI", "RAW - Full reasoning traces (open weight models)"),
        (":nebius", "Nebius", "RAW - Full reasoning traces (European infrastructure)"),
        (":auto", "Auto", "Varies by model - automatic provider selection"),
    ]

    print("\nExpected Reasoning Visibility by Provider Suffix:")
    print("-" * 60)
    for suffix, name, expected in provider_info:
        print(f"{suffix:12} | {name:12} | {expected}")
    print("-" * 60)

    print("\nReasoning Effort Levels:")
    print("-" * 60)
    print("  low    - Minimal reasoning, faster responses")
    print("  medium - Balanced reasoning depth (default)")
    print("  high   - Deep reasoning, more thorough analysis")
    print("-" * 60)

    print("\nRecommendation:")
    print("  For maximum reasoning visibility, use open weight models")
    print("  via HuggingFace router with :groq, :together, or :nebius suffix.")
    print("  These providers expose raw reasoning traces.")
    print("  Use reasoning={'effort': 'high'} for maximum reasoning depth.")


def demonstrate_reasoning_efforts() -> None:
    """Demonstrate different reasoning effort levels."""
    model = os.environ.get("MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq")
    prompt = "What is 17 * 23? Show your work."

    print("\n" + "=" * 70)
    print("REASONING EFFORT LEVELS COMPARISON")
    print("=" * 70)
    print(f"Model: {model}")
    print(f'Prompt: "{prompt}"')
    print("=" * 70)

    efforts = ["low", "medium", "high"]

    for effort in efforts:
        print(f"\n--- Testing effort: {effort.upper()} ---")

        try:
            import time
            start_time = time.time()
            response = create_agent_with_reasoning(model, prompt, effort)
            duration = int((time.time() - start_time) * 1000)

            analysis = analyze_reasoning_visibility(response)

            print(f"Duration: {duration}ms")
            print(f"Reasoning Items: {len(analysis['reasoning_items'])}")
            in_tokens = response.usage.input_tokens if response.usage else 0
            out_tokens = response.usage.output_tokens if response.usage else 0
            print(f"Tokens: {in_tokens} in / {out_tokens} out")
            output_text = response.output_text or ""
            print(f"Answer: {output_text[:100]}...")

        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    """Main execution."""
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        exit(1)

    mode = os.environ.get("MODE", "demo")

    match mode:
        case "compare":
            compare_reasoning_across_providers()
        case "efforts":
            demonstrate_reasoning_efforts()
        case _:
            demonstrate_reasoning_visibility()


if __name__ == "__main__":
    main()
