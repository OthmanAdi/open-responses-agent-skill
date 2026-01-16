/**
 * Basic Agent Example - Open Responses API
 *
 * Demonstrates the simplest form of an agent using Open Responses
 * via the HuggingFace Inference Providers router.
 *
 * Usage:
 *     npm install openai
 *     export HF_TOKEN=your-token
 *     npx ts-node basic-agent.ts
 */

import OpenAI from "openai";

// Configure client with HuggingFace router endpoint
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

/**
 * Create a basic agent request to Open Responses API
 */
async function createBasicAgent(
  model: string,
  input: string,
  instructions?: string
) {
  console.log("=".repeat(60));
  console.log("BASIC AGENT EXAMPLE");
  console.log("=".repeat(60));
  console.log(`Model: ${model}`);
  console.log(`Input: ${input}`);
  console.log("=".repeat(60));

  const response = await client.responses.create({
    model,
    instructions: instructions || "You are a helpful assistant.",
    input,
  });

  console.log(`\nResponse ID: ${response.id}`);
  console.log(`Model: ${response.model}`);

  // Use the convenience helper for simple text output
  console.log("\n--- Output Text (convenience helper) ---");
  console.log(response.output_text);

  // Or iterate through all output items for more detail
  console.log("\n--- All Output Items ---");
  for (const item of response.output) {
    switch (item.type) {
      case "reasoning":
        // @ts-ignore - content may exist on reasoning items
        console.log(`[REASONING] ${item.content || "[no content]"}`);
        break;
      case "message":
        // @ts-ignore - content exists on message items
        console.log(`[MESSAGE] ${item.content}`);
        break;
      default:
        console.log(`[${item.type.toUpperCase()}]`, item);
    }
  }

  // Token usage
  console.log("\n--- Usage ---");
  console.log(`Input tokens: ${response.usage?.input_tokens}`);
  console.log(`Output tokens: ${response.usage?.output_tokens}`);

  return response;
}

// Main execution
async function main(): Promise<void> {
  // Model with provider suffix - using Groq for fast inference
  const model = process.env.MODEL || "moonshotai/Kimi-K2-Instruct-0905:groq";

  if (!process.env.HF_TOKEN) {
    console.error("Error: HF_TOKEN environment variable required");
    process.exit(1);
  }

  console.log(`Using model: ${model}`);
  console.log(`Endpoint: https://router.huggingface.co/v1/responses`);

  try {
    await createBasicAgent(
      model,
      "Explain the difference between TCP and UDP in simple terms."
    );
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
