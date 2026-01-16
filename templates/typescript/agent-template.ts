/**
 * Open Responses Agent Template - TypeScript
 *
 * A production-ready template for building autonomous agents using
 * the Open Responses API via HuggingFace Inference Providers router.
 *
 * Usage:
 *   1. npm install openai
 *   2. Copy this file to your project
 *   3. Customize the tools array
 *   4. Update the task description
 *   5. Set HF_TOKEN environment variable
 *   6. Run with: npx ts-node agent-template.ts
 */

import OpenAI from "openai";

// ============================================================
// CONFIGURATION
// ============================================================

const CONFIG = {
  // HuggingFace token (required)
  apiKey: process.env.HF_TOKEN || "",

  // Model with provider suffix (model:provider format)
  // Examples:
  //   - "moonshotai/Kimi-K2-Instruct-0905:groq" (Groq - fast)
  //   - "meta-llama/Llama-3.1-70B-Instruct:together" (Together AI)
  //   - "meta-llama/Llama-3.1-70B-Instruct:nebius" (Nebius - EU)
  //   - "meta-llama/Llama-3.1-70B-Instruct:auto" (Auto selection)
  model: process.env.MODEL || "moonshotai/Kimi-K2-Instruct-0905:groq",

  // Agent configuration
  timeout: parseInt(process.env.TIMEOUT || "60000", 10),

  // Reasoning configuration
  reasoningEffort: (process.env.REASONING_EFFORT || "medium") as "low" | "medium" | "high",

  // Logging
  verbose: process.env.VERBOSE === "true",
};

// ============================================================
// CLIENT SETUP
// ============================================================

// Configure client with HuggingFace router endpoint
// This SINGLE endpoint routes to different providers via model suffix
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: CONFIG.apiKey,
});

// ============================================================
// TOOLS - CUSTOMIZE THESE FOR YOUR USE CASE
// ============================================================

// NOTE: Tools are defined at TOP LEVEL (name, description, parameters)
// NOT nested inside a "function" object
const tools: OpenAI.Responses.Tool[] = [
  // TODO: Add your tools here
  {
    type: "function",
    name: "example_tool", // AT TOP LEVEL
    description: "An example tool - replace with your actual tools",
    parameters: {
      type: "object",
      properties: {
        input: {
          type: "string",
          description: "Input parameter for the tool",
        },
      },
      required: ["input"],
    },
  },
  // Add more tools as needed...
];

// ============================================================
// TOOL EXECUTION - IMPLEMENT YOUR TOOL LOGIC HERE
// ============================================================

async function executeTool(name: string, args: string): Promise<string> {
  const parsedArgs = JSON.parse(args);

  switch (name) {
    case "example_tool":
      // TODO: Implement your tool logic
      return `Executed example_tool with input: ${parsedArgs.input}`;

    // Add more tool implementations...

    default:
      return `Unknown tool: ${name}`;
  }
}

// ============================================================
// AGENT CORE
// ============================================================

/**
 * Create and run an agent
 */
async function runAgent(
  task: string,
  instructions?: string
): Promise<OpenAI.Responses.Response> {
  if (!CONFIG.apiKey) {
    throw new Error("HF_TOKEN environment variable is required");
  }

  if (CONFIG.verbose) {
    console.log(`[CONFIG] Endpoint: https://router.huggingface.co/v1/responses`);
    console.log(`[CONFIG] Model: ${CONFIG.model}`);
    console.log(`[CONFIG] Reasoning Effort: ${CONFIG.reasoningEffort}`);
    console.log(`[TASK] ${task}`);
  }

  const response = await client.responses.create({
    model: CONFIG.model,
    instructions: instructions || "You are a helpful assistant that completes tasks step by step.",
    input: task,
    tools: tools.length > 0 ? tools : undefined,
    tool_choice: tools.length > 0 ? "auto" : undefined,
    reasoning: { effort: CONFIG.reasoningEffort },
  });

  return response;
}

// ============================================================
// RESPONSE PROCESSING
// ============================================================

/**
 * Process and display agent response
 */
function processResponse(response: OpenAI.Responses.Response): void {
  console.log("\n" + "=".repeat(60));
  console.log("AGENT RESPONSE");
  console.log("=".repeat(60));
  console.log(`ID: ${response.id}`);
  console.log(`Model: ${response.model}`);
  console.log(`Output Items: ${response.output.length}`);
  console.log("=".repeat(60) + "\n");

  let toolCallCount = 0;

  for (const item of response.output) {
    switch (item.type) {
      case "reasoning":
        if (CONFIG.verbose) {
          // @ts-ignore
          const text = item.summary || item.content || "[encrypted]";
          console.log(`[REASONING] ${text.substring(0, 200)}${text.length > 200 ? "..." : ""}`);
        }
        break;

      case "function_call":
        toolCallCount++;
        // @ts-ignore
        console.log(`[TOOL CALL #${toolCallCount}] ${item.name}`);
        if (CONFIG.verbose) {
          // @ts-ignore
          console.log(`  Arguments: ${JSON.stringify(item.arguments)}`);
        }
        break;

      case "function_call_output":
        if (CONFIG.verbose) {
          // @ts-ignore
          console.log(`[TOOL RESULT] ${item.output?.substring(0, 200)}...`);
        }
        break;

      case "message":
        // @ts-ignore
        console.log(`\n[FINAL RESPONSE]\n${item.content}`);
        break;

      default:
        console.log(`[${item.type.toUpperCase()}]`, item);
    }
  }

  // Also show convenience helper output
  console.log("\n" + "─".repeat(60));
  console.log("OUTPUT TEXT (convenience helper):");
  console.log("─".repeat(60));
  console.log(response.output_text);

  // Token usage
  console.log("\n" + "─".repeat(60));
  console.log(`Tool Calls: ${toolCallCount}`);
  console.log(`Tokens: ${response.usage?.input_tokens || 0} in / ${response.usage?.output_tokens || 0} out`);
}

// ============================================================
// MAIN - CUSTOMIZE YOUR TASK HERE
// ============================================================

async function main(): Promise<void> {
  // Check required environment variable
  if (!CONFIG.apiKey) {
    console.error("Error: HF_TOKEN environment variable is required");
    process.exit(1);
  }

  // TODO: Define your task here
  const task = `
    Your task description goes here.
    Be specific about what you want the agent to accomplish.
  `;

  // TODO: Customize your system prompt (optional)
  const instructions = "You are a helpful assistant that completes tasks step by step.";

  try {
    const response = await runAgent(task, instructions);
    processResponse(response);
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

// Run if this is the main module
main();

// Export for use as a module
export { runAgent, processResponse, CONFIG, tools, executeTool, client };
