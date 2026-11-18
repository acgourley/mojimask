//! What the app says about itself in an export's metadata (about.py): its name and where its source lives. The name
//! is set only in native/web/brand.js (its BRAND object is plain JSON), read here at compile time, so the pages, the
//! Python side and this can't disagree.
use std::sync::OnceLock;

const BRAND_JS: &str = include_str!("../../../web/brand.js");
pub const URL: &str = "https://github.com/acgourley/mojimask";

/// brand.js's BRAND object.
pub fn brand() -> &'static serde_json::Value {
    static B: OnceLock<serde_json::Value> = OnceLock::new();
    B.get_or_init(|| {
        let at = BRAND_JS.find("const BRAND").expect("brand.js: no BRAND");
        let open = at + BRAND_JS[at..].find('{').expect("brand.js: no object");
        let mut de = serde_json::Deserializer::from_str(&BRAND_JS[open..]).into_iter::<serde_json::Value>();
        de.next().expect("brand.js: empty").expect("brand.js: BRAND is not JSON")
    })
}

/// the app's name, as people see it
pub fn name() -> String {
    brand()["name"].as_str().unwrap_or("mojimask").to_string()
}

/// an export's description: "Made with <name>: <url>"
pub fn made_with() -> String {
    format!("Made with {}: {}", name(), URL)
}
