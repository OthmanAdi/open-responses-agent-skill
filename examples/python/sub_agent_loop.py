"""
Sub-Agent Loop Example - Open Responses API

Demonstrates multi-step autonomous workflows with tools.
The API automatically handles the agentic loop.

Usage:
    pip install openai
    export HF_TOKEN=your-token
    python sub_agent_loop.py
"""

import os
from openai import OpenAI


# Configure client with HuggingFace router endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)


# Define tools for the agent
# NOTE: Tools are defined at TOP LEVEL (name, description, parameters)
# NOT nested inside a "function" object
tools = [
    {
        "type": "function",
        "name": "search_documents",  # AT TOP LEVEL
        "description": "Search company documents and knowledge base for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents",
                },
                "department": {
                    "type": "string",
                    "description": "Optional department filter (sales, engineering, hr)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "analyze_data",
        "description": "Analyze numerical data and return insights",
        "parameters": {
            "type": "object",
            "properties": {
                "data_source": {
                    "type": "string",
                    "description": "The data source to analyze (e.g., 'q3_sales', 'user_metrics')",
                },
                "metric": {
                    "type": "string",
                    "description": "The specific metric to analyze (e.g., 'revenue', 'growth', 'churn')",
                },
            },
            "required": ["data_source", "metric"],
        },
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email to specified recipients",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Email recipient address",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Email body content",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "type": "function",
        "name": "create_report",
        "description": "Create a formatted report document",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Report title",
                },
                "sections": {
                    "type": "string",
                    "description": "JSON array of section objects with title and content",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: 'markdown', 'html', or 'pdf'",
                },
            },
            "required": ["title", "sections"],
        },
    },
]


def create_agent_with_tools(model: str, input_text: str, instructions: str | None = None):
    """
    Create an agent with sub-agent loop capability.

    Args:
        model: Model identifier with provider suffix
        input_text: The user's task
        instructions: Optional system prompt

    Returns:
        Response with all execution items
    """
    print(f"\n[REQUEST] Sending to HuggingFace router...")
    print(f"[MODEL] {model}")
    print(f"[INPUT] {input_text[:100]}...")

    response = client.responses.create(
        model=model,
        instructions=instructions or "You are a helpful assistant that completes tasks step by step.",
        input=input_text,
        tools=tools,
        tool_choice="auto",
    )

    return response


def display_execution_trace(response) -> None:
    """Display the complete execution trace."""
    print(f"\n{'='*60}")
    print(f"EXECUTION TRACE - {response.id}")
    print(f"{'='*60}")
    print(f"Model: {response.model}")
    print(f"Total output items: {len(response.output)}")
    print(f"Tokens: {response.usage.input_tokens if response.usage else 0} in / {response.usage.output_tokens if response.usage else 0} out")
    print(f"{'='*60}\n")

    tool_call_count = 0

    for i, item in enumerate(response.output):
        prefix = f"[{i + 1}/{len(response.output)}]"

        match item.type:
            case "reasoning":
                text = getattr(item, "summary", None) or getattr(item, "content", None) or "[encrypted reasoning]"
                print(f"{prefix} [REASONING]")
                print(f"    {text[:200]}{'...' if len(text) > 200 else ''}")

            case "function_call":
                tool_call_count += 1
                print(f"{prefix} [TOOL CALL #{tool_call_count}]")
                print(f"    Function: {getattr(item, 'name', 'unknown')}")
                print(f"    Arguments: {getattr(item, 'arguments', '{}')}")

            case "function_call_output":
                output = getattr(item, "output", "") or ""
                print(f"{prefix} [TOOL RESULT]")
                print(f"    Result: {output[:200]}{'...' if len(output) > 200 else ''}")

            case "message":
                print(f"{prefix} [FINAL RESPONSE]")
                print(f"    {getattr(item, 'content', '')}")

            case _:
                print(f"{prefix} [{item.type.upper()}]")
                print(f"    {item}")

        print()

    print(f"{'='*60}")
    print(f"SUMMARY: {tool_call_count} tool calls made")
    print(f"{'='*60}")

    # Also show the convenience output_text
    print(f"\n--- Final Output Text ---")
    print(response.output_text)


def main() -> None:
    """Main execution."""
    # Model with provider suffix
    model = os.environ.get("MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        exit(1)

    # Complex multi-step task that requires multiple tool calls
    task = """
    I need you to complete the following multi-step task:

    1. Search for Q3 2024 sales data in the sales department
    2. Analyze the revenue and growth metrics from the sales data
    3. Create a summary report with the key findings
    4. Email the report to the team at team@company.com

    Please complete all steps and provide a final summary.
    """

    try:
        result = create_agent_with_tools(model, task)
        display_execution_trace(result)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
