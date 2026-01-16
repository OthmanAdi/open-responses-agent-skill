/**
 * Multi-Provider Example - Open Responses API
 *
 * Demonstrates provider routing via model suffixes using the unified
 * HuggingFace router endpoint. Single codebase, multiple providers.
 *
 * Usage:
 *     npm install openai
 *     export HF_TOKEN=your-token
 *     npx ts-node multi-provider.ts
 */

import OpenAI from "openai";

// Configure client with HuggingFace router endpoint
// This SINGLE endpoint routes to different providers via model suffix
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

/**
 * Provider information - providers are specified via model SUFFIX
 * e.g., "model-name:groq" or "model-name:together"
 */
interface ProviderInfo {
  suffix: string;
  name: string;
  description: string;
  exampleModel: string;
}

const PROVIDERS: ProviderInfo[] = [
  {
    suffix: ":groq",
    name: "Groq",
    description: "Fast inference provider",
    exampleModel: "moonshotai/Kimi-K2-Instruct-0905:groq",
  },
  {
    suffix: ":together",
    name: "Together AI",
    description: "Open weight model specialist",
    exampleModel: "meta-llama/Llama-3.1-70B-Instruct:together",
  },
  {
    suffix: ":nebius",
    name: "Nebius AI",
    description: "European infrastructure",
    exampleModel: "meta-llama/Llama-3.1-70B-Instruct:nebius",
  },
  {
    suffix: ":auto",
    name: "Auto",
    description: "Automatic provider selection",
    exampleModel: "meta-llama/Llama-3.1-70B-Instruct:auto",
  },
];

/**
 * Create an agent with a specific model (provider specified via suffix)
 */
async function createAgent(model: string, input: string, instructions?: string) {
  const response = await client.responses.create({
    model,
    instructions: instructions || "You are a helpful assistant.",
    input,
  });

  return response;
}

/**
 * Compare the same prompt across different providers
 */
async function compareProviders(prompt: string, models: string[]): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("MULTI-PROVIDER COMPARISON");
  console.log("=".repeat(70));
  console.log(`Endpoint: https://router.huggingface.co/v1/responses`);
  console.log(`Prompt: "${prompt}"`);
  console.log(`Models: ${models.join(", ")}`);
  console.log("=".repeat(70) + "\n");

  const results: Array<{
    model: string;
    response?: OpenAI.Responses.Response;
    duration?: number;
    error?: string;
  }> = [];

  for (const model of models) {
    // Extract provider suffix for display
    const suffix = model.includes(":") ? model.split(":").pop() : "default";
    console.log(`\n--- Testing ${suffix?.toUpperCase()} (${model}) ---`);

    try {
      console.log(`Sending request...`);

      const startTime = Date.now();
      const response = await createAgent(model, prompt);
      const duration = Date.now() - startTime;

      console.log(`Response received in ${duration}ms`);
      console.log(`Tokens: ${response.usage?.input_tokens || 0} in / ${response.usage?.output_tokens || 0} out`);

      // Display reasoning (if available)
      const reasoningItems = response.output.filter((i) => i.type === "reasoning");
      if (reasoningItems.length > 0) {
        console.log(`\nReasoning (${reasoningItems.length} items):`);
        for (const item of reasoningItems) {
          // @ts-ignore
          const text = item.content || item.summary || "[no content]";
          console.log(`  - ${text.substring(0, 150)}${text.length > 150 ? "..." : ""}`);
        }
      }

      // Display final response using convenience helper
      console.log(`\nResponse:`);
      const outputText = response.output_text || "";
      console.log(`  ${outputText.substring(0, 300)}${outputText.length > 300 ? "..." : ""}`);

      results.push({ model, response, duration });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log(`Error: ${errorMessage}`);
      results.push({ model, error: errorMessage });
    }
  }

  // Summary
  console.log("\n" + "=".repeat(70));
  console.log("COMPARISON SUMMARY");
  console.log("=".repeat(70));

  for (const result of results) {
    const suffix = result.model.includes(":") ? result.model.split(":").pop() : "default";
    if (result.response) {
      const tokens = (result.response.usage?.input_tokens || 0) + (result.response.usage?.output_tokens || 0);
      console.log(
        `${(suffix || "").padEnd(12)} | SUCCESS | ${result.duration}ms | ${tokens} tokens`
      );
    } else {
      console.log(`${(suffix || "").padEnd(12)} | FAILED  | ${result.error}`);
    }
  }
}

/**
 * Demonstrate provider switching via model suffix
 */
async function demonstrateProviderSwitching(): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("PROVIDER SWITCHING DEMONSTRATION");
  console.log("=".repeat(70));
  console.log("\nKey concept: Provider is specified via MODEL SUFFIX");
  console.log("Endpoint is ALWAYS: https://router.huggingface.co/v1");
  console.log("=".repeat(70));

  // Default model - uses Groq provider
  const model = process.env.MODEL || "moonshotai/Kimi-K2-Instruct-0905:groq";

  console.log(`\nUsing model: ${model}`);
  const suffix = model.includes(":") ? model.split(":").pop() : "default";
  console.log(`Provider (from suffix): ${suffix}`);

  console.log(`\nSending request...`);
  const response = await createAgent(model, "What is 2 + 2? Explain your reasoning.");

  console.log(`\nResponse ID: ${response.id}`);
  console.log(`Model: ${response.model}`);

  // Show output items
  console.log(`\nOutput items:`);
  for (const item of response.output) {
    switch (item.type) {
      case "reasoning":
        // @ts-ignore
        const text = item.content || item.summary || "[no content]";
        console.log(`  [REASONING] ${text.substring(0, 100)}...`);
        break;
      case "message":
        // @ts-ignore
        console.log(`  [MESSAGE] ${item.content}`);
        break;
      default:
        console.log(`  [${item.type.toUpperCase()}]`, item);
    }
  }

  // Show convenience helper
  console.log(`\n--- Output Text (convenience helper) ---`);
  console.log(response.output_text);

  // Show available providers
  console.log("\n" + "=".repeat(70));
  console.log("AVAILABLE PROVIDERS (via model suffix)");
  console.log("=".repeat(70));
  for (const provider of PROVIDERS) {
    console.log(`  ${provider.suffix.padEnd(12)} - ${provider.name}: ${provider.description}`);
    console.log(`               Example: ${provider.exampleModel}`);
  }
}

// Main execution
async function main(): Promise<void> {
  if (!process.env.HF_TOKEN) {
    console.error("Error: HF_TOKEN environment variable required");
    process.exit(1);
  }

  const mode = process.env.MODE || "switch";

  try {
    if (mode === "compare") {
      // Compare multiple providers
      const models = [
        "moonshotai/Kimi-K2-Instruct-0905:groq",
        // Add more models with different provider suffixes to compare
        // "meta-llama/Llama-3.1-70B-Instruct:together",
        // "meta-llama/Llama-3.1-70B-Instruct:nebius",
      ];
      await compareProviders("Explain quantum entanglement in one paragraph.", models);
    } else {
      // Demonstrate provider switching
      await demonstrateProviderSwitching();
    }
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
