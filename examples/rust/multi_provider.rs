//! Multi-Provider Example - Open Responses API
//!
//! Demonstrates provider routing via model suffixes using the unified
//! HuggingFace router endpoint. Single codebase, multiple providers.
//!
//! Usage:
//!     export HF_TOKEN=your-token
//!     cargo run --bin multi_provider

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::time::Instant;

/// Single unified endpoint - routes to different providers via model suffix
const ENDPOINT: &str = "https://router.huggingface.co/v1/responses";

/// Provider information - providers are specified via model SUFFIX
/// e.g., "model-name:groq" or "model-name:together"
struct ProviderInfo {
    suffix: &'static str,
    name: &'static str,
    description: &'static str,
    example_model: &'static str,
}

fn get_providers() -> Vec<ProviderInfo> {
    vec![
        ProviderInfo {
            suffix: ":groq",
            name: "Groq",
            description: "Fast inference provider",
            example_model: "moonshotai/Kimi-K2-Instruct-0905:groq",
        },
        ProviderInfo {
            suffix: ":together",
            name: "Together AI",
            description: "Open weight model specialist",
            example_model: "meta-llama/Llama-3.1-70B-Instruct:together",
        },
        ProviderInfo {
            suffix: ":nebius",
            name: "Nebius AI",
            description: "European infrastructure",
            example_model: "meta-llama/Llama-3.1-70B-Instruct:nebius",
        },
        ProviderInfo {
            suffix: ":auto",
            name: "Auto",
            description: "Automatic provider selection",
            example_model: "meta-llama/Llama-3.1-70B-Instruct:auto",
        },
    ]
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

/// Create an agent with a specific model (provider specified via suffix)
async fn create_agent(
    model: &str,
    input_text: &str,
    instructions: Option<&str>,
) -> Result<OpenResponsesResponse, Box<dyn std::error::Error>> {
    let client = Client::new();
    let response = client
        .post(ENDPOINT)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", env::var("HF_TOKEN")?))
        .json(&json!({
            "model": model,
            "instructions": instructions.unwrap_or("You are a helpful assistant."),
            "input": input_text
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

/// Compare the same prompt across different providers
async fn compare_providers(prompt: &str, models: &[&str]) {
    println!("\n{}", "=".repeat(70));
    println!("MULTI-PROVIDER COMPARISON");
    println!("{}", "=".repeat(70));
    println!("Endpoint: {}", ENDPOINT);
    println!("Prompt: \"{}\"", prompt);
    println!("Models: {:?}", models);
    println!("{}\n", "=".repeat(70));

    let mut results: Vec<(&str, Option<OpenResponsesResponse>, Option<String>, Option<u128>)> = Vec::new();

    for model in models {
        // Extract provider suffix for display
        let suffix = model.split(':').last().unwrap_or("default");
        println!("\n--- Testing {} ({}) ---", suffix.to_uppercase(), model);

        println!("Sending request...");

        let start = Instant::now();
        match create_agent(model, prompt, None).await {
            Ok(response) => {
                let duration = start.elapsed().as_millis();
                println!("Response received in {}ms", duration);

                if let Some(usage) = &response.usage {
                    println!("Tokens: {} in / {} out", usage.input_tokens, usage.output_tokens);
                }

                // Display reasoning (if available)
                let reasoning_items: Vec<_> = response
                    .output
                    .iter()
                    .filter(|i| i.item_type == "reasoning")
                    .collect();

                if !reasoning_items.is_empty() {
                    println!("\nReasoning ({} items):", reasoning_items.len());
                    for item in reasoning_items {
                        let text = item
                            .content
                            .as_ref()
                            .or(item.summary.as_ref())
                            .map(|s| s.as_str())
                            .unwrap_or("[no content]");
                        let display = if text.len() > 150 {
                            format!("{}...", &text[..150])
                        } else {
                            text.to_string()
                        };
                        println!("  - {}", display);
                    }
                }

                // Display final response using convenience helper
                println!("\nResponse:");
                let output_text = response.output_text.as_deref().unwrap_or("");
                let display = if output_text.len() > 300 {
                    format!("{}...", &output_text[..300])
                } else {
                    output_text.to_string()
                };
                println!("  {}", display);

                results.push((model, Some(response), None, Some(duration)));
            }
            Err(e) => {
                println!("Error: {}", e);
                results.push((model, None, Some(e.to_string()), None));
            }
        }
    }

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("COMPARISON SUMMARY");
    println!("{}", "=".repeat(70));

    for (model, response, error, duration) in results {
        let suffix = model.split(':').last().unwrap_or("default");
        if let Some(resp) = response {
            let total = resp.usage.map(|u| u.input_tokens + u.output_tokens).unwrap_or(0);
            println!(
                "{:12} | SUCCESS | {}ms | {} tokens",
                suffix,
                duration.unwrap_or(0),
                total
            );
        } else if let Some(err) = error {
            println!("{:12} | FAILED  | {}", suffix, err);
        }
    }
}

/// Demonstrate provider switching via model suffix
async fn demonstrate_provider_switching() {
    println!("\n{}", "=".repeat(70));
    println!("PROVIDER SWITCHING DEMONSTRATION");
    println!("{}", "=".repeat(70));
    println!("\nKey concept: Provider is specified via MODEL SUFFIX");
    println!("Endpoint is ALWAYS: {}", ENDPOINT);
    println!("{}", "=".repeat(70));

    // Default model - uses Groq provider
    let model = env::var("MODEL").unwrap_or_else(|_| "moonshotai/Kimi-K2-Instruct-0905:groq".to_string());

    println!("\nUsing model: {}", model);
    let suffix = model.split(':').last().unwrap_or("default");
    println!("Provider (from suffix): {}", suffix);

    println!("\nSending request...");

    match create_agent(&model, "What is 2 + 2? Explain your reasoning.", None).await {
        Ok(response) => {
            println!("\nResponse ID: {}", response.id);
            println!("Model: {}", response.model);

            // Show output items
            println!("\nOutput items:");
            for item in &response.output {
                match item.item_type.as_str() {
                    "reasoning" => {
                        let text = item
                            .content
                            .as_ref()
                            .or(item.summary.as_ref())
                            .map(|s| s.as_str())
                            .unwrap_or("[no content]");
                        let display = if text.len() > 100 {
                            format!("{}...", &text[..100])
                        } else {
                            text.to_string()
                        };
                        println!("  [REASONING] {}", display);
                    }
                    "message" => {
                        if let Some(content) = &item.content {
                            println!("  [MESSAGE] {}", content);
                        }
                    }
                    _ => {
                        println!("  [{}] {:?}", item.item_type.to_uppercase(), item);
                    }
                }
            }

            // Show convenience helper
            println!("\n--- Output Text (convenience helper) ---");
            println!("{}", response.output_text.as_deref().unwrap_or("[no output]"));
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // Show available providers
    println!("\n{}", "=".repeat(70));
    println!("AVAILABLE PROVIDERS (via model suffix)");
    println!("{}", "=".repeat(70));
    for provider in get_providers() {
        println!(
            "  {:12} - {}: {}",
            provider.suffix, provider.name, provider.description
        );
        println!("               Example: {}", provider.example_model);
    }
}

#[tokio::main]
async fn main() {
    if env::var("HF_TOKEN").is_err() {
        eprintln!("Error: HF_TOKEN environment variable required");
        std::process::exit(1);
    }

    let mode = env::var("MODE").unwrap_or_else(|_| "switch".to_string());

    if mode == "compare" {
        // Compare multiple providers
        let models = vec![
            "moonshotai/Kimi-K2-Instruct-0905:groq",
            // Add more models with different provider suffixes to compare
            // "meta-llama/Llama-3.1-70B-Instruct:together",
            // "meta-llama/Llama-3.1-70B-Instruct:nebius",
        ];
        compare_providers("Explain quantum entanglement in one paragraph.", &models).await;
    } else {
        // Demonstrate provider switching
        demonstrate_provider_switching().await;
    }
}
