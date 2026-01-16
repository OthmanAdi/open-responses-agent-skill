//! Basic Agent Example - Open Responses API
//!
//! Demonstrates the simplest form of an agent using Open Responses
//! via the HuggingFace Inference Providers router.
//!
//! Usage:
//!     export HF_TOKEN=your-token
//!     cargo run --bin basic_agent

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

/// Single unified endpoint - routes to different providers via model suffix
const ENDPOINT: &str = "https://router.huggingface.co/v1/responses";

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

/// Request body
#[derive(Debug, Serialize)]
struct RequestBody {
    model: String,
    instructions: String,
    input: String,
}

/// Create a basic agent request to Open Responses API
async fn create_basic_agent(
    model: &str,
    input_text: &str,
    instructions: Option<&str>,
) -> Result<OpenResponsesResponse, Box<dyn std::error::Error>> {
    let client = Client::new();
    let response = client
        .post(ENDPOINT)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", env::var("HF_TOKEN")?))
        .json(&RequestBody {
            model: model.to_string(),
            instructions: instructions.unwrap_or("You are a helpful assistant.").to_string(),
            input: input_text.to_string(),
        })
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

/// Display the response in a readable format
fn display_response(response: &OpenResponsesResponse) {
    println!("\n{}", "=".repeat(60));
    println!("Response ID: {}", response.id);
    println!("Model: {}", response.model);
    if let Some(usage) = &response.usage {
        println!(
            "Tokens: {} in / {} out",
            usage.input_tokens, usage.output_tokens
        );
    }
    println!("{}\n", "=".repeat(60));

    // Use the convenience helper for simple text output
    println!("--- Output Text (convenience helper) ---");
    println!("{}", response.output_text.as_deref().unwrap_or("[no output]"));

    // Or iterate through all output items for more detail
    println!("\n--- All Output Items ---");
    for item in &response.output {
        match item.item_type.as_str() {
            "reasoning" => {
                // Open weight models provide raw content
                // Proprietary models may provide summary or encrypted_content
                let text = item
                    .content
                    .as_ref()
                    .or(item.summary.as_ref())
                    .map(|s| s.as_str())
                    .unwrap_or("[encrypted]");
                println!("[REASONING] {}", text);
            }
            "message" => {
                if let Some(content) = &item.content {
                    println!("[MESSAGE] {}", content);
                }
            }
            _ => {
                println!("[{}] {:?}", item.item_type.to_uppercase(), item);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    // Model with provider suffix - using Groq for fast inference
    // Format: model-id:provider (e.g., :groq, :together, :nebius, :auto)
    let model = env::var("MODEL").unwrap_or_else(|_| "moonshotai/Kimi-K2-Instruct-0905:groq".to_string());

    if env::var("HF_TOKEN").is_err() {
        eprintln!("Error: HF_TOKEN environment variable required");
        std::process::exit(1);
    }

    println!("Using model: {}", model);
    println!("Endpoint: {}", ENDPOINT);

    match create_basic_agent(
        &model,
        "Explain the difference between TCP and UDP in simple terms.",
        None,
    )
    .await
    {
        Ok(result) => display_response(&result),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
