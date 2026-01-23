use wasm_bindgen::prelude::*;
use tokenizers::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

/// Initialize panic hook for better error messages in the browser console
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Result of encoding text
#[derive(Serialize, Deserialize)]
pub struct EncodingResult {
    pub ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_mask: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub special_tokens_mask: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offsets: Option<Vec<(usize, usize)>>,
}

/// Wrapper around the HuggingFace tokenizer for WASM
#[wasm_bindgen]
pub struct WasmTokenizer {
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl WasmTokenizer {
    /// Create a new tokenizer from JSON bytes (tokenizer.json content)
    #[wasm_bindgen(constructor)]
    pub fn new(json_bytes: &[u8]) -> Result<WasmTokenizer, JsError> {
        let tokenizer = Tokenizer::from_bytes(json_bytes)
            .map_err(|e| JsError::new(&format!("Failed to load tokenizer: {}", e)))?;
        Ok(WasmTokenizer { tokenizer })
    }

    /// Create a tokenizer from a JSON string
    #[wasm_bindgen(js_name = fromString)]
    pub fn from_string(json_str: &str) -> Result<WasmTokenizer, JsError> {
        let tokenizer = Tokenizer::from_bytes(json_str.as_bytes())
            .map_err(|e| JsError::new(&format!("Failed to load tokenizer: {}", e)))?;
        Ok(WasmTokenizer { tokenizer })
    }

    /// Encode text to token IDs
    /// Returns a Uint32Array of token IDs
    #[wasm_bindgen]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, JsError> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| JsError::new(&format!("Encoding error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text and return detailed encoding information as a JS object
    #[wasm_bindgen(js_name = encodeWithDetails)]
    pub fn encode_with_details(
        &self,
        text: &str,
        add_special_tokens: bool,
        return_tokens: bool,
        return_type_ids: bool,
        return_attention_mask: bool,
        return_special_tokens_mask: bool,
        return_offsets: bool,
    ) -> Result<JsValue, JsError> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| JsError::new(&format!("Encoding error: {}", e)))?;

        let result = EncodingResult {
            ids: encoding.get_ids().to_vec(),
            tokens: if return_tokens {
                Some(encoding.get_tokens().to_vec())
            } else {
                None
            },
            type_ids: if return_type_ids {
                Some(encoding.get_type_ids().to_vec())
            } else {
                None
            },
            attention_mask: if return_attention_mask {
                Some(encoding.get_attention_mask().to_vec())
            } else {
                None
            },
            special_tokens_mask: if return_special_tokens_mask {
                Some(encoding.get_special_tokens_mask().to_vec())
            } else {
                None
            },
            offsets: if return_offsets {
                Some(encoding.get_offsets().to_vec())
            } else {
                None
            },
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }

    /// Decode token IDs back to text
    #[wasm_bindgen]
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, JsError> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| JsError::new(&format!("Decoding error: {}", e)))
    }

    /// Get the vocabulary size
    #[wasm_bindgen(js_name = vocabSize)]
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get a token by its ID
    #[wasm_bindgen(js_name = idToToken)]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id).map(|s| s.to_string())
    }

    /// Get the ID for a token
    #[wasm_bindgen(js_name = tokenToId)]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}
