//! Open Responses Agent Template - Rust
//!
//! A production-ready template for building autonomous agents.
//! Customize the CONFIG, tools, and execute_tool function for your use case.
//!
//! Usage:
//!     export PROVIDER=huggingface
//!     export API_KEY=your-key
//!     cargo run

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;

// =============================================================================
// CONFIGURATION - Customize for your use case
// =============================================================================

/// Agent configuration
struct Config {
    provider: String,
    api_key: String,
    model: String,
    max_tool_calls: u32,
    timeout_secs: u64,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let provider = env::var("PROVIDER").unwrap_or_else(|_| "huggingface".to_string());
        let api_key = env::var("API_KEY")
            .or_else(|_| env::var("HF_TOKEN"))
            .map_err(|_| "API_KEY or HF_TOKEN environment variable required")?;
        let model =
            env::var("MODEL").unwrap_or_else(|_| "meta-llama/Llama-3.1-70B-Instruct".to_string());

        Ok(Self {
            provider,
            api_key,
            model,
            max_tool_calls: 10,
            timeout_secs: 120,
        })
    }
}

/// Provider endpoints
fn get_endpoints() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("openai", "https://api.openai.com/v1/responses"),
        ("anthropic", "https://api.anthropic.com/v1/responses"),
        ("huggingface", "https://api-inference.huggingface.co/v1/responses"),
        ("together", "https://api.together.xyz/v1/responses"),
        ("nebius", "https://api.nebius.ai/v1/responses"),
    ])
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// Reasoning visibility levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningLevel {
    Raw,
    Summary,
    Encrypted,
    None,
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

/// A single item in the response
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseItem {
    #[serde(rename = "type")]
    pub item_type: String,
    pub content: Option<String>,
    pub summary: Option<String>,
    pub encrypted_content: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    pub id: Option<String>,
    pub tool_call_id: Option<String>,
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
    pub items: Vec<ResponseItem>,
    pub usage: Usage,
}

// =============================================================================
// TOOLS - Define your agent's capabilities
// =============================================================================

/// Create tool definitions
fn create_tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "example_tool",
                "description": "An example tool - replace with your own",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input to process"
                        }
                    },
                    "required": ["input"]
                }
            }
        }),
        // Add more tools here...
    ]
}

/// Execute a tool and return the result
fn execute_tool(name: &str, arguments: &Value) -> String {
    match name {
        "example_tool" => {
            // Replace with your implementation
            let input = arguments
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            format!("Processed: {}", input)
        }
        _ => format!("Unknown tool: {}", name),
    }
}

// =============================================================================
// OPEN RESPONSES CLIENT
// =============================================================================

/// Provider-agnostic Open Responses agent
pub struct OpenResponsesAgent {
    provider: String,
    endpoint: String,
    api_key: String,
    model: String,
    max_tool_calls: u32,
    timeout_secs: u64,
}

impl OpenResponsesAgent {
    /// Create a new agent
    pub fn new(config: Config) -> Result<Self, String> {
        let endpoints = get_endpoints();
        let endpoint = endpoints
            .get(config.provider.as_str())
            .ok_or_else(|| {
                let available: Vec<_> = endpoints.keys().collect();
                format!(
                    "Unknown provider: {}. Available: {:?}",
                    config.provider, available
                )
            })?
            .to_string();

        Ok(Self {
            provider: config.provider,
            endpoint,
            api_key: config.api_key,
            model: config.model,
            max_tool_calls: config.max_tool_calls,
            timeout_secs: config.timeout_secs,
        })
    }

    /// Send a request to the Open Responses API
    pub async fn create(
        &self,
        input_text: &str,
        tools: Option<Vec<Value>>,
    ) -> Result<OpenResponsesResponse, Box<dyn std::error::Error>> {
        let mut request_body = json!({
            "model": &self.model,
            "input": input_text
        });

        if let Some(tools) = tools {
            request_body["tools"] = json!(tools);
            request_body["max_tool_calls"] = json!(self.max_tool_calls);
            request_body["tool_choice"] = json!("auto");
        }

        let client = Client::new();
        let response = client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("OpenResponses-Version", "latest")
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
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

    /// Get reasoning from an item
    pub fn get_reasoning(&self, item: &ResponseItem) -> (ReasoningLevel, String) {
        if item.item_type != "reasoning" {
            return (ReasoningLevel::None, String::new());
        }

        if item.content.is_some() && item.encrypted_content.is_none() {
            return (
                ReasoningLevel::Raw,
                item.content.clone().unwrap_or_default(),
            );
        }

        if let Some(summary) = &item.summary {
            return (ReasoningLevel::Summary, summary.clone());
        }

        if item.encrypted_content.is_some() {
            return (ReasoningLevel::Encrypted, "[encrypted]".to_string());
        }

        (ReasoningLevel::None, String::new())
    }

    /// Get provider info
    pub fn info(&self) -> String {
        format!("Provider: {}, Model: {}", self.provider, self.model)
    }
}

// =============================================================================
// EXECUTION HELPERS
// =============================================================================

/// Display the response in a readable format
fn display_response(agent: &OpenResponsesAgent, response: &OpenResponsesResponse) {
    println!("\n{}", "=".repeat(60));
    println!("Response ID: {}", response.id);
    println!("Model: {}", response.model);
    println!(
        "Tokens: {} in / {} out",
        response.usage.input_tokens, response.usage.output_tokens
    );
    println!("{}\n", "=".repeat(60));

    let mut tool_call_count = 0;

    for item in &response.items {
        match item.item_type.as_str() {
            "reasoning" => {
                let (level, text) = agent.get_reasoning(item);
                let display = if text.len() > 200 {
                    format!("{}...", &text[..200])
                } else {
                    text
                };
                println!("[REASONING ({})] {}", level.as_str(), display);
            }
            "tool_call" => {
                tool_call_count += 1;
                println!(
                    "[TOOL CALL #{}] {}",
                    tool_call_count,
                    item.name.as_deref().unwrap_or("unknown")
                );
                println!("  Arguments: {}", item.arguments.as_deref().unwrap_or("{}"));
            }
            "tool_result" => {
                let content = item.content.as_deref().unwrap_or("");
                let display = if content.len() > 150 {
                    format!("{}...", &content[..150])
                } else {
                    content.to_string()
                };
                println!("[TOOL RESULT] {}", display);
            }
            "message" => {
                if let Some(content) = &item.content {
                    println!("[RESPONSE] {}", content);
                }
            }
            _ => {
                println!("[{}] {:?}", item.item_type.to_uppercase(), item);
            }
        }
        println!();
    }
}

// =============================================================================
// MAIN EXECUTION
// =============================================================================

/// Run the agent with a task
async fn run_agent(task: &str, use_tools: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env().map_err(|e| e)?;
    let agent = OpenResponsesAgent::new(config)?;

    println!("\n{}", "=".repeat(60));
    println!("OPEN RESPONSES AGENT");
    println!("{}", "=".repeat(60));
    println!("{}", agent.info());
    println!("Tools: {}", if use_tools { "Enabled" } else { "Disabled" });
    println!("{}", "=".repeat(60));
    println!("\nTask: {}\n", task);
    println!("Processing...\n");

    let tools = if use_tools {
        Some(create_tools())
    } else {
        None
    };

    let response = agent.create(task, tools).await?;
    display_response(&agent, &response);

    Ok(())
}

#[tokio::main]
async fn main() {
    // Example task - customize for your use case
    let task = r#"
    Explain the key benefits of using the Open Responses API
    for building autonomous agents.
    "#;

    match run_agent(task, false).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    // Or run with tools:
    // run_agent("Process this input using the example tool", true).await;
}
