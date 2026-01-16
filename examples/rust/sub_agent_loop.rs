//! Sub-Agent Loop Example - Open Responses API
//!
//! Demonstrates multi-step autonomous workflows with tools.
//! The API automatically handles the agentic loop.
//!
//! Usage:
//!     export HF_TOKEN=your-token
//!     cargo run --bin sub_agent_loop

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
    pub name: Option<String>,
    pub arguments: Option<Value>,
    pub output: Option<String>,
    pub call_id: Option<String>,
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

/// Create tool definitions
/// NOTE: Tools are defined at TOP LEVEL (name, description, parameters)
/// NOT nested inside a "function" object
fn create_tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "name": "search_documents",  // AT TOP LEVEL
            "description": "Search company documents and knowledge base for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "department": {
                        "type": "string",
                        "description": "Optional department filter (sales, engineering, hr)"
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "type": "function",
            "name": "analyze_data",
            "description": "Analyze numerical data and return insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "description": "The data source to analyze (e.g., 'q3_sales', 'user_metrics')"
                    },
                    "metric": {
                        "type": "string",
                        "description": "The specific metric to analyze (e.g., 'revenue', 'growth', 'churn')"
                    }
                },
                "required": ["data_source", "metric"]
            }
        }),
        json!({
            "type": "function",
            "name": "send_email",
            "description": "Send an email to specified recipients",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Email recipient address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }),
        json!({
            "type": "function",
            "name": "create_report",
            "description": "Create a formatted report document",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Report title"
                    },
                    "sections": {
                        "type": "string",
                        "description": "JSON array of section objects with title and content"
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'html', or 'pdf'"
                    }
                },
                "required": ["title", "sections"]
            }
        }),
    ]
}

/// Create an agent with sub-agent loop capability
async fn create_agent_with_tools(
    model: &str,
    input_text: &str,
    instructions: Option<&str>,
) -> Result<OpenResponsesResponse, Box<dyn std::error::Error>> {
    let tools = create_tools();

    println!("\n[REQUEST] Sending to HuggingFace router...");
    println!("[MODEL] {}", model);
    println!("[INPUT] {}...", &input_text[..input_text.len().min(100)]);

    let client = Client::new();
    let response = client
        .post(ENDPOINT)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", env::var("HF_TOKEN")?))
        .json(&json!({
            "model": model,
            "instructions": instructions.unwrap_or("You are a helpful assistant that completes tasks step by step."),
            "input": input_text,
            "tools": tools,
            "tool_choice": "auto"
        }))
        .timeout(std::time::Duration::from_secs(120))
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

/// Display the complete execution trace
fn display_execution_trace(response: &OpenResponsesResponse) {
    println!("\n{}", "=".repeat(60));
    println!("EXECUTION TRACE - {}", response.id);
    println!("{}", "=".repeat(60));
    println!("Model: {}", response.model);
    println!("Total output items: {}", response.output.len());
    if let Some(usage) = &response.usage {
        println!(
            "Tokens: {} in / {} out",
            usage.input_tokens, usage.output_tokens
        );
    }
    println!("{}\n", "=".repeat(60));

    let mut tool_call_count = 0;

    for (i, item) in response.output.iter().enumerate() {
        let prefix = format!("[{}/{}]", i + 1, response.output.len());

        match item.item_type.as_str() {
            "reasoning" => {
                let text = item
                    .summary
                    .as_ref()
                    .or(item.content.as_ref())
                    .map(|s| s.as_str())
                    .unwrap_or("[encrypted reasoning]");
                println!("{} [REASONING]", prefix);
                let display_text = if text.len() > 200 {
                    format!("{}...", &text[..200])
                } else {
                    text.to_string()
                };
                println!("    {}", display_text);
            }
            "function_call" => {
                tool_call_count += 1;
                println!("{} [TOOL CALL #{}]", prefix, tool_call_count);
                println!("    Function: {}", item.name.as_deref().unwrap_or("unknown"));
                println!(
                    "    Arguments: {}",
                    item.arguments
                        .as_ref()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string())
                );
            }
            "function_call_output" => {
                let output = item.output.as_deref().unwrap_or("");
                println!("{} [TOOL RESULT]", prefix);
                let display_output = if output.len() > 200 {
                    format!("{}...", &output[..200])
                } else {
                    output.to_string()
                };
                println!("    Result: {}", display_output);
            }
            "message" => {
                println!("{} [FINAL RESPONSE]", prefix);
                if let Some(content) = &item.content {
                    println!("    {}", content);
                }
            }
            _ => {
                println!("{} [{}]", prefix, item.item_type.to_uppercase());
                println!("    {:?}", item);
            }
        }
        println!();
    }

    println!("{}", "=".repeat(60));
    println!("SUMMARY: {} tool calls made", tool_call_count);
    println!("{}", "=".repeat(60));

    // Also show the convenience output_text
    println!("\n--- Final Output Text ---");
    println!("{}", response.output_text.as_deref().unwrap_or("[no output]"));
}

#[tokio::main]
async fn main() {
    // Model with provider suffix
    let model = env::var("MODEL").unwrap_or_else(|_| "moonshotai/Kimi-K2-Instruct-0905:groq".to_string());

    if env::var("HF_TOKEN").is_err() {
        eprintln!("Error: HF_TOKEN environment variable required");
        std::process::exit(1);
    }

    // Complex multi-step task that requires multiple tool calls
    let task = r#"
    I need you to complete the following multi-step task:

    1. Search for Q3 2024 sales data in the sales department
    2. Analyze the revenue and growth metrics from the sales data
    3. Create a summary report with the key findings
    4. Email the report to the team at team@company.com

    Please complete all steps and provide a final summary.
    "#;

    match create_agent_with_tools(&model, task, None).await {
        Ok(result) => display_execution_trace(&result),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
