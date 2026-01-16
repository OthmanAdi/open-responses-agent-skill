"""
Basic Agent Example - Open Responses API

Demonstrates the simplest form of an agent using Open Responses
via the HuggingFace Inference Providers router.

Usage:
    pip install openai
    export HF_TOKEN=your-token
    python basic_agent.py
"""

import os
from openai import OpenAI


# Configure client with HuggingFace router endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)


def create_basic_agent(model: str, input_text: str, instructions: str | None = None):
    """
    Create a basic agent request to Open Responses API.

    Args:
        model: Model identifier with provider suffix (e.g., "model:groq")
        input_text: The user's request
        instructions: Optional system prompt

    Returns:
        Response with all output items
    """
    response = client.responses.create(
        model=model,
        instructions=instructions or "You are a helpful assistant.",
        input=input_text,
    )

    return response


def display_response(response) -> None:
    """Display the response in a readable format."""
    print(f"\n{'='*60}")
    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.input_tokens if response.usage else 0} in / {response.usage.output_tokens if response.usage else 0} out")
    print(f"{'='*60}\n")

    # Use the convenience helper for simple text output
    print("--- Output Text (convenience helper) ---")
    print(response.output_text)

    # Or iterate through all output items for more detail
    print("\n--- All Output Items ---")
    for item in response.output:
        match item.type:
            case "reasoning":
                # Open weight models provide raw content
                # Proprietary models may provide summary or encrypted_content
                text = getattr(item, "content", None) or getattr(item, "summary", None) or "[encrypted]"
                print(f"[REASONING] {text}")

            case "message":
                print(f"[MESSAGE] {getattr(item, 'content', '')}")

            case _:
                print(f"[{item.type.upper()}] {item}")


def main() -> None:
    """Main execution."""
    # Model with provider suffix - using Groq for fast inference
    model = os.environ.get("MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        exit(1)

    print(f"Using model: {model}")
    print(f"Endpoint: https://router.huggingface.co/v1/responses")

    try:
        result = create_basic_agent(
            model=model,
            input_text="Explain the difference between TCP and UDP in simple terms.",
        )

        display_response(result)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
