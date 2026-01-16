"""
Open Responses Agent Template - Python

A production-ready template for building autonomous agents using
the Open Responses API via HuggingFace Inference Providers router.

Usage:
    pip install openai
    export HF_TOKEN=your-token
    python agent_template.py
"""

import os
from enum import Enum
from openai import OpenAI


# =============================================================================
# CONFIGURATION - Customize for your use case
# =============================================================================

CONFIG = {
    # HuggingFace token (required)
    "api_key": os.environ.get("HF_TOKEN"),

    # Model with provider suffix (model:provider format)
    # Examples:
    #   - "moonshotai/Kimi-K2-Instruct-0905:groq" (Groq - fast)
    #   - "meta-llama/Llama-3.1-70B-Instruct:together" (Together AI)
    #   - "meta-llama/Llama-3.1-70B-Instruct:nebius" (Nebius - EU)
    #   - "meta-llama/Llama-3.1-70B-Instruct:auto" (Auto selection)
    "model": os.environ.get("MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq"),

    # Reasoning configuration
    "reasoning_effort": os.environ.get("REASONING_EFFORT", "medium"),

    # Timeout
    "timeout": 120.0,
}

# Configure client with HuggingFace router endpoint
# This SINGLE endpoint routes to different providers via model suffix
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=CONFIG["api_key"],
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ReasoningLevel(Enum):
    """Reasoning visibility levels."""
    RAW = "raw"           # Full thinking traces (open weight)
    SUMMARY = "summary"   # Sanitized summary (some proprietary)
    ENCRYPTED = "encrypted"  # No visibility (most proprietary)
    NONE = "none"         # No reasoning at all


# =============================================================================
# TOOLS - Define your agent's capabilities
# =============================================================================

# NOTE: Tools are defined at TOP LEVEL (name, description, parameters)
# NOT nested inside a "function" object
TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "example_tool",  # AT TOP LEVEL
        "description": "An example tool - replace with your own",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to process",
                },
            },
            "required": ["input"],
        },
    },
    # Add more tools here...
]


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool and return the result.

    Args:
        name: The tool name
        arguments: The tool arguments

    Returns:
        The tool result as a string
    """
    match name:
        case "example_tool":
            # Replace with your implementation
            return f"Processed: {arguments.get('input', '')}"

        case _:
            return f"Unknown tool: {name}"


# =============================================================================
# AGENT CORE
# =============================================================================

def create_agent(
    task: str,
    instructions: str | None = None,
    tools: list[dict] | None = None,
    reasoning_effort: str = CONFIG["reasoning_effort"],
):
    """
    Create and run an agent.

    Args:
        task: The task to complete
        instructions: Optional system prompt
        tools: Optional list of tools
        reasoning_effort: Reasoning effort level (low, medium, high)

    Returns:
        Response object
    """
    request_params = {
        "model": CONFIG["model"],
        "instructions": instructions or "You are a helpful assistant that completes tasks step by step.",
        "input": task,
        "reasoning": {"effort": reasoning_effort},
    }

    if tools:
        request_params["tools"] = tools
        request_params["tool_choice"] = "auto"

    response = client.responses.create(**request_params)

    return response


def get_reasoning(item) -> tuple[ReasoningLevel, str]:
    """
    Extract reasoning from an item.

    Args:
        item: A response output item

    Returns:
        Tuple of (level, text)
    """
    if item.type != "reasoning":
        return ReasoningLevel.NONE, ""

    content = getattr(item, "content", None)
    summary = getattr(item, "summary", None)
    encrypted_content = getattr(item, "encrypted_content", None)

    if content and not encrypted_content:
        return ReasoningLevel.RAW, content

    if summary:
        return ReasoningLevel.SUMMARY, summary

    if encrypted_content:
        return ReasoningLevel.ENCRYPTED, "[encrypted]"

    return ReasoningLevel.NONE, ""


# =============================================================================
# EXECUTION HELPERS
# =============================================================================

def display_response(response) -> None:
    """Display the response in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    in_tokens = response.usage.input_tokens if response.usage else 0
    out_tokens = response.usage.output_tokens if response.usage else 0
    print(f"Tokens: {in_tokens} in / {out_tokens} out")
    print(f"{'=' * 60}\n")

    tool_call_count = 0

    for item in response.output:
        match item.type:
            case "reasoning":
                text = getattr(item, "content", None) or getattr(item, "summary", None) or "[encrypted]"
                print(f"[REASONING] {text[:200]}{'...' if len(text) > 200 else ''}")

            case "function_call":
                tool_call_count += 1
                name = getattr(item, "name", "unknown")
                args = getattr(item, "arguments", "{}")
                print(f"[TOOL CALL #{tool_call_count}] {name}")
                print(f"  Arguments: {args}")

            case "function_call_output":
                output = getattr(item, "output", "") or ""
                print(f"[TOOL RESULT] {output[:150]}{'...' if len(output) > 150 else ''}")

            case "message":
                print(f"[RESPONSE] {getattr(item, 'content', '')}")

            case _:
                print(f"[{item.type.upper()}] {item}")

        print()

    # Also show convenience helper output
    print("─" * 60)
    print("OUTPUT TEXT (convenience helper):")
    print("─" * 60)
    print(response.output_text)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_agent(task: str, use_tools: bool = False):
    """
    Run the agent with a task.

    Args:
        task: The task description
        use_tools: Whether to enable tools

    Returns:
        The agent response
    """
    print(f"\n{'=' * 60}")
    print("OPEN RESPONSES AGENT")
    print(f"{'=' * 60}")
    print(f"Endpoint: https://router.huggingface.co/v1/responses")
    print(f"Model: {CONFIG['model']}")
    print(f"Tools: {'Enabled' if use_tools else 'Disabled'}")
    print(f"Reasoning Effort: {CONFIG['reasoning_effort']}")
    print(f"{'=' * 60}")
    print(f"\nTask: {task}\n")
    print("Processing...\n")

    tools = TOOLS if use_tools else None
    response = create_agent(task, tools=tools)

    display_response(response)

    return response


def main() -> None:
    """Main execution."""
    if not CONFIG["api_key"]:
        print("Error: HF_TOKEN environment variable required")
        exit(1)

    # Example task - customize for your use case
    task = """
    Explain the key benefits of using the Open Responses API
    for building autonomous agents.
    """

    try:
        # Run without tools
        run_agent(task, use_tools=False)

        # Or run with tools:
        # run_agent("Process this input using the example tool", use_tools=True)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
