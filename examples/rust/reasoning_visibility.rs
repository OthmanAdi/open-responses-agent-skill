//! Reasoning Visibility Example - Open Responses API
//!
//! Demonstrates how to access and display agent reasoning.
//! Shows differences between open weight models (raw traces) and
//! proprietary models (summaries/encrypted).
//!
//! Usage:
//!     export HF_TOKEN=your-token
//!     cargo run --bin reasoning_visibility

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::env;

/// Single unified endpoint - routes to different providers via model suffix
const ENDPOINT: &str = "https://router.huggingface.co/v1/responses";

/// Reasoning visibility levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningLevel {
    Raw,       // Full thinking traces (open weight)
    Summary,   // Sanitized summary (some proprietary)
    Encrypted, // No visibility (most proprietary)
    None,      // No reasoning at all
}

impl ReasoningLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReasoningLevel::Raw => "RAW",
            ReasoningLevel::Summary => "SUMMARY",
            ReasoningLevel::Encrypted => "ENCRYPTED",
            ReasoningLevel::None => "NONE",
        }
    }
}

/// A single item in the response output
#[derive(Debug, Clone, Deserialize)]
pub struct OutputItem {
    #[serde(rename = "type")]
    pub item_type: String,
    pub content: Option<String>,
    pub summary: Option<String>,
    pub encrypted_content: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Response from Open Responses API
#[derive(Debug, Clone, Deserialize)]
pub struct OpenResponsesResponse {
    pub id: String,
    pub model: String,
    /// Output items (NOT "items" - the field is "output")
    pub output: Vec<OutputItem>,
    /// Convenience helper for text output
    pub output_text: Option<String>,
    pub usage: Option<Usage>,
}

/// Analysis of reasoning visibility in a response
#[derive(Debug)]
pub struct ReasoningAnalysis {
    pub level: ReasoningLevel,
    pub reasoning_items: Vec<OutputItem>,
    pub total_reasoning_tokens: u32,
    pub details: String,
}

/// Analyze reasoning visibility for a response
fn analyze_reasoning_visibility(response: &OpenResponsesResponse) -> ReasoningAnalysis {
    let reasoning_items: Vec<_> = response
        .output
        .iter()
        .filter(|item| item.item_type == "reasoning")
        .cloned()
        .collect();

    if reasoning_items.is_empty() {
        return ReasoningAnalysis {
            level: ReasoningLevel::None,
            reasoning_items: vec![],
            total_reasoning_tokens: 0,
            details: "No reasoning items found in response.".to_string(),
        };
    }

    // Check what type of reasoning is available
    let has_raw_content = reasoning_items
        .iter()
        .any(|item| item.content.is_some() && item.encrypted_content.is_none());
    let has_encrypted = reasoning_items
        .iter()
        .any(|item| item.encrypted_content.is_some());
    let has_summary = reasoning_items.iter().any(|item| item.summary.is_some());

    // Estimate tokens (rough approximation)
    let total_reasoning_tokens: u32 = reasoning_items
        .iter()
        .map(|item| {
            let text = item
                .content
                .as_ref()
                .or(item.summary.as_ref())
                .map(|s| s.len())
                .unwrap_or(0);
            (text / 4) as u32
        })
        .sum();

    if has_raw_content {
        return ReasoningAnalysis {
            level: ReasoningLevel::Raw,
            reasoning_items,
            total_reasoning_tokens,
            details: "Full raw reasoning traces available. This model provides complete transparency.".to_string(),
        };
    }

    if has_summary {
        return ReasoningAnalysis {
            level: ReasoningLevel::Summary,
            reasoning_items,
            total_reasoning_tokens,
            details: "Summarized reasoning available. Raw traces are not exposed.".to_string(),
        };
    }

    if has_encrypted {
        return ReasoningAnalysis {
            level: ReasoningLevel::Encrypted,
            reasoning_items,
            total_reasoning_tokens: 0,
            details: "Reasoning is encrypted and not accessible to the client.".to_string(),
        };
    }

    ReasoningAnalysis {
        level: ReasoningLevel::None,
        reasoning_items,
        total_reasoning_tokens: 0,
        details: "Unknown reasoning format.".to_string(),
    }
}

/// Pretty print reasoning items
fn display_reasoning(reasoning_items: &[OutputItem], level: ReasoningLevel) {
    println!("\n{}", "-".repeat(60));
    println!("REASONING TRACE");
    println!("{}", "-".repeat(60));
    println!("Visibility Level: {}", level.as_str());
    println!("Total Items: {}", reasoning_items.len());
    println!("{}\n", "-".repeat(60));

    for (i, item) in reasoning_items.iter().enumerate() {
        println!("[Step {}]", i + 1);

        if let Some(content) = &item.content {
            println!("Type: Raw Trace");
            println!("Content:");
            for line in content.lines() {
                println!("  {}", line);
            }
        } else if let Some(summary) = &item.summary {
            println!("Type: Summary");
            println!("Content: {}", summary);
        } else if let Some(encrypted) = &item.encrypted_content {
            println!("Type: Encrypted");
            println!("Content: [ENCRYPTED - Not accessible]");
            println!("Encrypted Length: {} chars", encrypted.len());
        } else {
            println!("Type: Unknown");
            println!("Raw: {:?}", item);
        }

        println!();
    }
}

/// Create agent request with reasoning focus
async fn create_agent_with_reasoning(
    model: &str,
    input_text: &str,
    reasoning_effort: &str,
) -> Result<OpenResponsesResponse, Box<dyn std::error::Error>> {
    let client = Client::new();
    let response = client
        .post(ENDPOINT)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", env::var("HF_TOKEN")?))
        .json(&json!({
            "model": model,
            "instructions": "You are a helpful assistant. Show your step-by-step reasoning process.",
            "input": input_text,
            "reasoning": { "effort": reasoning_effort }
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await?;
        return Err(format!("HTTP error: {} - {}", status, text).into());
    }

    let data: OpenResponsesResponse = response.json().await?;
    Ok(data)
}

/// Demonstrate reasoning visibility with a reasoning-heavy prompt
async fn demonstrate_reasoning_visibility() {
    // Model with provider suffix - using Groq for fast inference
    let model = env::var("MODEL").unwrap_or_else(|_| "moonshotai/Kimi-K2-Instruct-0905:groq".to_string());
    let reasoning_effort = env::var("REASONING_EFFORT").unwrap_or_else(|_| "medium".to_string());

    // Prompt designed to elicit multi-step reasoning
    let reasoning_prompt = r#"
    Solve this step by step:

    A farmer has 15 chickens and 12 cows. Some of the animals get sick.
    After treating them, the farmer has 80% of the original chickens
    and 75% of the original cows healthy.

    1. How many chickens are healthy?
    2. How many cows are healthy?
    3. What is the total number of healthy animals?
    4. What percentage of all animals are healthy?

    Show all your work and reasoning.
    "#;

    println!("\n{}", "=".repeat(60));
    println!("REASONING VISIBILITY DEMONSTRATION");
    println!("{}", "=".repeat(60));
    println!("Endpoint: {}", ENDPOINT);
    println!("Model: {}", model);
    println!("Reasoning Effort: {}", reasoning_effort);
    println!("{}", "=".repeat(60));
    println!("\nPrompt:{}", reasoning_prompt);

    println!("\nSending request...\n");

    match create_agent_with_reasoning(&model, reasoning_prompt, &reasoning_effort).await {
        Ok(response) => {
            // Analyze reasoning visibility
            let analysis = analyze_reasoning_visibility(&response);

            println!("{}", "=".repeat(60));
            println!("REASONING ANALYSIS");
            println!("{}", "=".repeat(60));
            println!("Response ID: {}", response.id);
            println!("Model: {}", response.model);
            println!("Visibility Level: {}", analysis.level.as_str());
            println!("Reasoning Items: {}", analysis.reasoning_items.len());
            println!("Est. Reasoning Tokens: ~{}", analysis.total_reasoning_tokens);
            println!("Details: {}", analysis.details);

            // Display reasoning traces
            if !analysis.reasoning_items.is_empty() {
                display_reasoning(&analysis.reasoning_items, analysis.level);
            }

            // Display final answer using convenience helper
            println!("{}", "-".repeat(60));
            println!("FINAL ANSWER (output_text)");
            println!("{}", "-".repeat(60));
            println!("{}", response.output_text.as_deref().unwrap_or("[no output]"));

            // Also show individual output items
            println!("\n{}", "-".repeat(60));
            println!("ALL OUTPUT ITEMS");
            println!("{}", "-".repeat(60));
            for (i, item) in response.output.iter().enumerate() {
                println!("[{}/{}] Type: {}", i + 1, response.output.len(), item.item_type);

                match item.item_type.as_str() {
                    "reasoning" => {
                        let text = item
                            .content
                            .as_ref()
                            .or(item.summary.as_ref())
                            .map(|s| s.as_str())
                            .unwrap_or("[encrypted]");
                        let display = if text.len() > 150 {
                            format!("{}...", &text[..150])
                        } else {
                            text.to_string()
                        };
                        println!("    {}", display);
                    }
                    "message" => {
                        if let Some(content) = &item.content {
                            let display = if content.len() > 150 {
                                format!("{}...", &content[..150])
                            } else {
                                content.clone()
                            };
                            println!("    {}", display);
                        }
                    }
                    _ => {
                        println!("    {:?}", item);
                    }
                }
            }

            // Token usage
            if let Some(usage) = &response.usage {
                println!("\n{}", "-".repeat(60));
                println!("TOKEN USAGE");
                println!("{}", "-".repeat(60));
                println!("Input: {}", usage.input_tokens);
                println!("Output: {}", usage.output_tokens);
                println!("Total: {}", usage.input_tokens + usage.output_tokens);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Compare reasoning visibility across providers
fn compare_reasoning_across_providers() {
    println!("\n{}", "=".repeat(70));
    println!("REASONING VISIBILITY COMPARISON");
    println!("{}", "=".repeat(70));
    println!("\nEndpoint: {} (unified)", ENDPOINT);
    println!("Provider is specified via MODEL SUFFIX (e.g., :groq, :together)");

    let provider_info = [
        (":groq", "Groq", "RAW - Full reasoning traces (open weight models)"),
        (":together", "Together AI", "RAW - Full reasoning traces (open weight models)"),
        (":nebius", "Nebius", "RAW - Full reasoning traces (European infrastructure)"),
        (":auto", "Auto", "Varies by model - automatic provider selection"),
    ];

    println!("\nExpected Reasoning Visibility by Provider Suffix:");
    println!("{}", "-".repeat(60));
    for (suffix, name, expected) in provider_info {
        println!("{:12} | {:12} | {}", suffix, name, expected);
    }
    println!("{}", "-".repeat(60));

    println!("\nReasoning Effort Levels:");
    println!("{}", "-".repeat(60));
    println!("  low    - Minimal reasoning, faster responses");
    println!("  medium - Balanced reasoning depth (default)");
    println!("  high   - Deep reasoning, more thorough analysis");
    println!("{}", "-".repeat(60));

    println!("\nRecommendation:");
    println!("  For maximum reasoning visibility, use open weight models");
    println!("  via HuggingFace router with :groq, :together, or :nebius suffix.");
    println!("  These providers expose raw reasoning traces.");
    println!("  Use reasoning: {{ effort: \"high\" }} for maximum reasoning depth.");
}

#[tokio::main]
async fn main() {
    if env::var("HF_TOKEN").is_err() {
        eprintln!("Error: HF_TOKEN environment variable required");
        std::process::exit(1);
    }

    let mode = env::var("MODE").unwrap_or_else(|_| "demo".to_string());

    if mode == "compare" {
        compare_reasoning_across_providers();
    } else {
        demonstrate_reasoning_visibility().await;
    }
}
