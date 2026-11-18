//! What the app says about itself in an export's metadata (about.py): its name and where its source lives. The name
//! is set only in native/web/brand.json, read here at compile time, so the pages and this can't disagree.
use std::sync::OnceLock;

const BRAND_JSON: &str = include_str!("../../../web/brand.json");
pub const URL: &str = "https://github.com/acgourley/mojimask";

/// brand.json.
pub fn brand() -> &'static serde_json::Value {
    static B: OnceLock<serde_json::Value> = OnceLock::new();
    B.get_or_init(|| serde_json::from_str(BRAND_JSON).expect("native/web/brand.json"))
}

/// the app's name, as people see it
pub fn name() -> String {
    brand()["name"].as_str().unwrap_or("mojimask").to_string()
}

/// an export's description: "Made with <name>: <url>"
pub fn made_with() -> String {
    format!("Made with {}: {}", name(), URL)
}
