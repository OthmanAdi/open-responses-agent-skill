/**
 * Reasoning Visibility Example - Open Responses API
 *
 * Demonstrates how to access and display agent reasoning.
 * Shows differences between open weight models (raw traces) and
 * proprietary models (summaries/encrypted).
 *
 * Usage:
 *     npm install openai
 *     export HF_TOKEN=your-token
 *     npx ts-node reasoning-visibility.ts
 */

import OpenAI from "openai";

// Configure client with HuggingFace router endpoint
const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

/**
 * Reasoning visibility levels
 */
enum ReasoningLevel {
  RAW = "raw", // Full thinking traces (open weight)
  SUMMARY = "summary", // Sanitized summary (some proprietary)
  ENCRYPTED = "encrypted", // No visibility (most proprietary)
  NONE = "none", // No reasoning at all
}

/**
 * Reasoning effort levels for the API
 */
type ReasoningEffort = "low" | "medium" | "high";

/**
 * Analyze reasoning visibility for a response
 */
function analyzeReasoningVisibility(response: OpenAI.Responses.Response): {
  level: ReasoningLevel;
  reasoningItems: Array<{ type: string; content?: string; summary?: string; encrypted_content?: string }>;
  totalReasoningTokens: number;
  details: string;
} {
  const reasoningItems = response.output.filter((item) => item.type === "reasoning");

  if (reasoningItems.length === 0) {
    return {
      level: ReasoningLevel.NONE,
      reasoningItems: [],
      totalReasoningTokens: 0,
      details: "No reasoning items found in response.",
    };
  }

  // Check what type of reasoning is available
  // @ts-ignore - content may exist on reasoning items
  const hasRawContent = reasoningItems.some((item) => item.content && !item.encrypted_content);
  // @ts-ignore
  const hasEncrypted = reasoningItems.some((item) => item.encrypted_content);
  // @ts-ignore
  const hasSummary = reasoningItems.some((item) => item.summary);

  // Estimate tokens (rough approximation)
  const totalReasoningTokens = reasoningItems.reduce((sum, item) => {
    // @ts-ignore
    const text = item.content || item.summary || "";
    return sum + Math.ceil(text.length / 4);
  }, 0);

  if (hasRawContent) {
    return {
      level: ReasoningLevel.RAW,
      // @ts-ignore
      reasoningItems,
      totalReasoningTokens,
      details: "Full raw reasoning traces available. This model provides complete transparency.",
    };
  }

  if (hasSummary) {
    return {
      level: ReasoningLevel.SUMMARY,
      // @ts-ignore
      reasoningItems,
      totalReasoningTokens,
      details: "Summarized reasoning available. Raw traces are not exposed.",
    };
  }

  if (hasEncrypted) {
    return {
      level: ReasoningLevel.ENCRYPTED,
      // @ts-ignore
      reasoningItems,
      totalReasoningTokens: 0,
      details: "Reasoning is encrypted and not accessible to the client.",
    };
  }

  return {
    level: ReasoningLevel.NONE,
    // @ts-ignore
    reasoningItems,
    totalReasoningTokens: 0,
    details: "Unknown reasoning format.",
  };
}

/**
 * Pretty print reasoning items
 */
function displayReasoning(
  reasoningItems: Array<{ type: string; content?: string; summary?: string; encrypted_content?: string }>,
  level: ReasoningLevel
): void {
  console.log("\n" + "─".repeat(60));
  console.log("REASONING TRACE");
  console.log("─".repeat(60));
  console.log(`Visibility Level: ${level.toUpperCase()}`);
  console.log(`Total Items: ${reasoningItems.length}`);
  console.log("─".repeat(60) + "\n");

  for (let i = 0; i < reasoningItems.length; i++) {
    const item = reasoningItems[i];
    console.log(`[Step ${i + 1}]`);

    if (item.content) {
      console.log("Type: Raw Trace");
      console.log("Content:");
      // Format multi-line reasoning nicely
      const lines = item.content.split("\n");
      for (const line of lines) {
        console.log(`  ${line}`);
      }
    } else if (item.summary) {
      console.log("Type: Summary");
      console.log(`Content: ${item.summary}`);
    } else if (item.encrypted_content) {
      console.log("Type: Encrypted");
      console.log("Content: [ENCRYPTED - Not accessible]");
      console.log(`Encrypted Length: ${item.encrypted_content.length} chars`);
    } else {
      console.log("Type: Unknown");
      console.log(`Raw: ${JSON.stringify(item)}`);
    }

    console.log();
  }
}

/**
 * Create agent request with reasoning focus
 */
async function createAgentWithReasoning(
  model: string,
  input: string,
  reasoningEffort: ReasoningEffort = "medium"
): Promise<OpenAI.Responses.Response> {
  const response = await client.responses.create({
    model,
    instructions: "You are a helpful assistant. Show your step-by-step reasoning process.",
    input,
    reasoning: { effort: reasoningEffort },
  });

  return response;
}

/**
 * Demonstrate reasoning visibility with a reasoning-heavy prompt
 */
async function demonstrateReasoningVisibility(): Promise<void> {
  // Model with provider suffix - using Groq for fast inference
  const model = process.env.MODEL || "moonshotai/Kimi-K2-Instruct-0905:groq";
  const reasoningEffort = (process.env.REASONING_EFFORT || "medium") as ReasoningEffort;

  if (!process.env.HF_TOKEN) {
    console.error("Error: HF_TOKEN environment variable required");
    process.exit(1);
  }

  // Prompt designed to elicit multi-step reasoning
  const reasoningPrompt = `
    Solve this step by step:

    A farmer has 15 chickens and 12 cows. Some of the animals get sick.
    After treating them, the farmer has 80% of the original chickens
    and 75% of the original cows healthy.

    1. How many chickens are healthy?
    2. How many cows are healthy?
    3. What is the total number of healthy animals?
    4. What percentage of all animals are healthy?

    Show all your work and reasoning.
  `;

  console.log("\n" + "=".repeat(60));
  console.log("REASONING VISIBILITY DEMONSTRATION");
  console.log("=".repeat(60));
  console.log(`Endpoint: https://router.huggingface.co/v1/responses`);
  console.log(`Model: ${model}`);
  console.log(`Reasoning Effort: ${reasoningEffort}`);
  console.log("=".repeat(60));
  console.log("\nPrompt:");
  console.log(reasoningPrompt);

  try {
    console.log("\nSending request...\n");
    const response = await createAgentWithReasoning(model, reasoningPrompt, reasoningEffort);

    // Analyze reasoning visibility
    const analysis = analyzeReasoningVisibility(response);

    console.log("=".repeat(60));
    console.log("REASONING ANALYSIS");
    console.log("=".repeat(60));
    console.log(`Response ID: ${response.id}`);
    console.log(`Model: ${response.model}`);
    console.log(`Visibility Level: ${analysis.level}`);
    console.log(`Reasoning Items: ${analysis.reasoningItems.length}`);
    console.log(`Est. Reasoning Tokens: ~${analysis.totalReasoningTokens}`);
    console.log(`Details: ${analysis.details}`);

    // Display reasoning traces
    if (analysis.reasoningItems.length > 0) {
      displayReasoning(analysis.reasoningItems, analysis.level);
    }

    // Display final answer using convenience helper
    console.log("─".repeat(60));
    console.log("FINAL ANSWER (output_text)");
    console.log("─".repeat(60));
    console.log(response.output_text);

    // Also show individual output items
    console.log("\n" + "─".repeat(60));
    console.log("ALL OUTPUT ITEMS");
    console.log("─".repeat(60));
    for (let i = 0; i < response.output.length; i++) {
      const item = response.output[i];
      console.log(`[${i + 1}/${response.output.length}] Type: ${item.type}`);

      switch (item.type) {
        case "reasoning":
          // @ts-ignore
          const reasoningText = item.content || item.summary || "[encrypted]";
          console.log(`    ${reasoningText.substring(0, 150)}${reasoningText.length > 150 ? "..." : ""}`);
          break;
        case "message":
          // @ts-ignore
          console.log(`    ${item.content?.substring(0, 150)}...`);
          break;
        default:
          console.log(`    ${JSON.stringify(item).substring(0, 150)}...`);
      }
    }

    // Token usage
    console.log("\n" + "─".repeat(60));
    console.log("TOKEN USAGE");
    console.log("─".repeat(60));
    console.log(`Input: ${response.usage?.input_tokens || 0}`);
    console.log(`Output: ${response.usage?.output_tokens || 0}`);
    console.log(`Total: ${(response.usage?.input_tokens || 0) + (response.usage?.output_tokens || 0)}`);
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

/**
 * Compare reasoning visibility across providers
 */
async function compareReasoningAcrossProviders(): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("REASONING VISIBILITY COMPARISON");
  console.log("=".repeat(70));
  console.log(`\nEndpoint: https://router.huggingface.co/v1 (unified)`);
  console.log(`Provider is specified via MODEL SUFFIX (e.g., :groq, :together)`);

  const providerInfo = [
    { suffix: ":groq", name: "Groq", expected: "RAW - Full reasoning traces (open weight models)" },
    { suffix: ":together", name: "Together AI", expected: "RAW - Full reasoning traces (open weight models)" },
    { suffix: ":nebius", name: "Nebius", expected: "RAW - Full reasoning traces (European infrastructure)" },
    { suffix: ":auto", name: "Auto", expected: "Varies by model - automatic provider selection" },
  ];

  console.log("\nExpected Reasoning Visibility by Provider Suffix:");
  console.log("─".repeat(60));
  for (const { suffix, name, expected } of providerInfo) {
    console.log(`${suffix.padEnd(12)} | ${name.padEnd(12)} | ${expected}`);
  }
  console.log("─".repeat(60));

  console.log("\nReasoning Effort Levels:");
  console.log("─".repeat(60));
  console.log("  low    - Minimal reasoning, faster responses");
  console.log("  medium - Balanced reasoning depth (default)");
  console.log("  high   - Deep reasoning, more thorough analysis");
  console.log("─".repeat(60));

  console.log("\nRecommendation:");
  console.log("  For maximum reasoning visibility, use open weight models");
  console.log("  via HuggingFace router with :groq, :together, or :nebius suffix.");
  console.log("  These providers expose raw reasoning traces.");
  console.log("  Use reasoning: { effort: 'high' } for maximum reasoning depth.");
}

/**
 * Demonstrate different reasoning effort levels
 */
async function demonstrateReasoningEfforts(): Promise<void> {
  const model = process.env.MODEL || "moonshotai/Kimi-K2-Instruct-0905:groq";
  const prompt = "What is 17 * 23? Show your work.";

  console.log("\n" + "=".repeat(70));
  console.log("REASONING EFFORT LEVELS COMPARISON");
  console.log("=".repeat(70));
  console.log(`Model: ${model}`);
  console.log(`Prompt: "${prompt}"`);
  console.log("=".repeat(70));

  const efforts: ReasoningEffort[] = ["low", "medium", "high"];

  for (const effort of efforts) {
    console.log(`\n--- Testing effort: ${effort.toUpperCase()} ---`);

    try {
      const startTime = Date.now();
      const response = await createAgentWithReasoning(model, prompt, effort);
      const duration = Date.now() - startTime;

      const analysis = analyzeReasoningVisibility(response);

      console.log(`Duration: ${duration}ms`);
      console.log(`Reasoning Items: ${analysis.reasoningItems.length}`);
      console.log(`Tokens: ${response.usage?.input_tokens || 0} in / ${response.usage?.output_tokens || 0} out`);
      console.log(`Answer: ${response.output_text?.substring(0, 100)}...`);
    } catch (error) {
      console.log(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

// Main execution
async function main(): Promise<void> {
  if (!process.env.HF_TOKEN) {
    console.error("Error: HF_TOKEN environment variable required");
    process.exit(1);
  }

  const mode = process.env.MODE || "demo";

  switch (mode) {
    case "compare":
      await compareReasoningAcrossProviders();
      break;
    case "efforts":
      await demonstrateReasoningEfforts();
      break;
    default:
      await demonstrateReasoningVisibility();
  }
}

main().catch(console.error);
