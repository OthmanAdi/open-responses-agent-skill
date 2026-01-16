/**
 * Sub-Agent Loop Example - Open Responses API
 *
 * Demonstrates multi-step autonomous workflows with tools.
 * The API automatically handles the agentic loop.
 *
 * Usage:
 *     npm install openai
 *     export HF_TOKEN=your-token
 *     npx ts-node sub-agent-loop.ts
 */

import OpenAI from "openai";

// Configure client with HuggingFace router endpoint
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

/**
 * Define tools for the agent
 * NOTE: Tools are defined at TOP LEVEL (name, description, parameters)
 * NOT nested inside a "function" object
 */
const tools: OpenAI.Responses.Tool[] = [
  {
    type: "function",
    name: "search_documents",
    description: "Search company documents and knowledge base for information",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query to find relevant documents",
        },
        department: {
          type: "string",
          description: "Optional department filter (sales, engineering, hr)",
        },
      },
      required: ["query"],
    },
  },
  {
    type: "function",
    name: "analyze_data",
    description: "Analyze numerical data and return insights",
    parameters: {
      type: "object",
      properties: {
        data_source: {
          type: "string",
          description: "The data source to analyze (e.g., 'q3_sales', 'user_metrics')",
        },
        metric: {
          type: "string",
          description: "The specific metric to analyze (e.g., 'revenue', 'growth', 'churn')",
        },
      },
      required: ["data_source", "metric"],
    },
  },
  {
    type: "function",
    name: "send_email",
    description: "Send an email to specified recipients",
    parameters: {
      type: "object",
      properties: {
        to: {
          type: "string",
          description: "Email recipient address",
        },
        subject: {
          type: "string",
          description: "Email subject line",
        },
        body: {
          type: "string",
          description: "Email body content",
        },
      },
      required: ["to", "subject", "body"],
    },
  },
  {
    type: "function",
    name: "create_report",
    description: "Create a formatted report document",
    parameters: {
      type: "object",
      properties: {
        title: {
          type: "string",
          description: "Report title",
        },
        sections: {
          type: "string",
          description: "JSON array of section objects with title and content",
        },
        format: {
          type: "string",
          description: "Output format: 'markdown', 'html', or 'pdf'",
        },
      },
      required: ["title", "sections"],
    },
  },
];

/**
 * Create an agent with sub-agent loop capability
 */
async function createAgentWithTools(
  model: string,
  input: string,
  instructions?: string
) {
  console.log(`\n[REQUEST] Sending to HuggingFace router...`);
  console.log(`[MODEL] ${model}`);
  console.log(`[INPUT] ${input.substring(0, 100)}...`);

  const response = await client.responses.create({
    model,
    instructions: instructions || "You are a helpful assistant that completes tasks step by step.",
    input,
    tools,
    tool_choice: "auto",
  });

  return response;
}

/**
 * Display the complete execution trace
 */
function displayExecutionTrace(response: OpenAI.Responses.Response): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`EXECUTION TRACE - ${response.id}`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Model: ${response.model}`);
  console.log(`Total output items: ${response.output.length}`);
  console.log(`Tokens: ${response.usage?.input_tokens || 0} in / ${response.usage?.output_tokens || 0} out`);
  console.log(`${"=".repeat(60)}\n`);

  let toolCallCount = 0;

  for (let i = 0; i < response.output.length; i++) {
    const item = response.output[i];
    const prefix = `[${i + 1}/${response.output.length}]`;

    switch (item.type) {
      case "reasoning":
        // @ts-ignore
        const reasoningText = item.content || item.summary || "[no content]";
        console.log(`${prefix} [REASONING]`);
        console.log(`    ${reasoningText.substring(0, 200)}${reasoningText.length > 200 ? "..." : ""}`);
        break;

      case "function_call":
        toolCallCount++;
        // @ts-ignore
        console.log(`${prefix} [TOOL CALL #${toolCallCount}]`);
        // @ts-ignore
        console.log(`    Function: ${item.name}`);
        // @ts-ignore
        console.log(`    Arguments: ${JSON.stringify(item.arguments)}`);
        break;

      case "function_call_output":
        console.log(`${prefix} [TOOL RESULT]`);
        // @ts-ignore
        const output = item.output || "";
        console.log(`    Result: ${output.substring(0, 200)}${output.length > 200 ? "..." : ""}`);
        break;

      case "message":
        console.log(`${prefix} [FINAL RESPONSE]`);
        // @ts-ignore
        console.log(`    ${item.content}`);
        break;

      default:
        console.log(`${prefix} [${item.type.toUpperCase()}]`);
        console.log(`    ${JSON.stringify(item)}`);
    }
    console.log();
  }

  console.log(`${"=".repeat(60)}`);
  console.log(`SUMMARY: ${toolCallCount} tool calls made`);
  console.log(`${"=".repeat(60)}`);

  // Also show the convenience output_text
  console.log(`\n--- Final Output Text ---`);
  console.log(response.output_text);
}

// Main execution
async function main(): Promise<void> {
  // Model with provider suffix
  const model = process.env.MODEL || "moonshotai/Kimi-K2-Instruct-0905:groq";

  if (!process.env.HF_TOKEN) {
    console.error("Error: HF_TOKEN environment variable required");
    process.exit(1);
  }

  // Complex multi-step task that requires multiple tool calls
  const task = `
    I need you to complete the following multi-step task:

    1. Search for Q3 2024 sales data in the sales department
    2. Analyze the revenue and growth metrics from the sales data
    3. Create a summary report with the key findings
    4. Email the report to the team at team@company.com

    Please complete all steps and provide a final summary.
  `;

  try {
    const result = await createAgentWithTools(model, task);
    displayExecutionTrace(result);
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
