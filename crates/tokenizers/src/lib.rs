use std::{collections::HashSet, ffi::CStr};
use std::path::PathBuf;
use std::ptr;
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use tiktoken_rs;

// Version-specific symbol that will cause link failure if version doesn't match
// Bump minor.patch version every time we bump tokenizers dependency version.
// Can't bump major version because Go doesn't like major version >= 2.
#[no_mangle]
pub extern "C" fn tokenizers_version_1_23_0() {
    // This function exists purely as a link-time version check
}

/// Truncation direction for tokenizer truncation
#[repr(u8)]
pub enum TruncationDirection {
    Left = 0,
    Right = 1,
}

impl TruncationDirection {
    /// Create from a u8 value, returns None if invalid
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(TruncationDirection::Left),
            1 => Some(TruncationDirection::Right),
            _ => None,
        }
    }
    
    /// Convert to the tokenizers crate's TruncationDirection
    pub fn to_tokenizers_direction(&self) -> tokenizers::tokenizer::TruncationDirection {
        match self {
            TruncationDirection::Left => tokenizers::tokenizer::TruncationDirection::Left,
            TruncationDirection::Right => tokenizers::tokenizer::TruncationDirection::Right,
        }
    }
}

// Unified tokenizer interface
pub enum UnifiedTokenizer {
    HuggingFace(Tokenizer),
    Tiktoken(tiktoken_rs::CoreBPE, u32, HashSet<String>, HashSet<u32>),
}

impl UnifiedTokenizer {
    /// Creates a HashSet of special token references for tiktoken encoding.
    /// This avoids recreating the HashSet on every encode call.
    fn get_special_tokens_refs<'a>(special_tokens: &'a HashSet<String>, add_special_tokens: bool) -> HashSet<&'a str> {
        if add_special_tokens {
            special_tokens.iter().map(String::as_str).collect()
        } else {
            HashSet::new()
        }
    }
    
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        match self {
            UnifiedTokenizer::HuggingFace(tokenizer) => {
                let encoding = tokenizer.encode(text, add_special_tokens)
                    .map_err(|e| format!("Encoding error: {}", e))?;
                Ok(encoding.get_ids().to_vec())
            }
            UnifiedTokenizer::Tiktoken(bpe, _vocab_size, special_tokens, _special_token_ids) => {
                let special_tokens_refs = Self::get_special_tokens_refs(special_tokens, add_special_tokens);
                let (tokens, _) = bpe.encode(text, &special_tokens_refs);
                Ok(tokens)
            }
        }
    }

    pub fn encode_with_details(&self, text: &str, add_special_tokens: bool) -> Result<EncodingDetails, Box<dyn std::error::Error>> {
        match self {
            UnifiedTokenizer::HuggingFace(tokenizer) => {
                let encoding = tokenizer.encode(text, add_special_tokens)
                    .map_err(|e| format!("Encoding error: {}", e))?;
                Ok(EncodingDetails {
                    ids: encoding.get_ids().to_vec(),
                    type_ids: Some(encoding.get_type_ids().to_vec()),
                    tokens: Some(encoding.get_tokens().iter().map(|s| s.to_string()).collect()),
                    special_tokens_mask: Some(encoding.get_special_tokens_mask().to_vec()),
                    attention_mask: Some(encoding.get_attention_mask().to_vec()),
                    offsets: Some(encoding.get_offsets().to_vec()),
                })
            }
            UnifiedTokenizer::Tiktoken(bpe, _vocab_size, special_tokens, _special_token_ids) => {
                let special_tokens_refs = Self::get_special_tokens_refs(special_tokens, add_special_tokens);
                let (tokens, _) = bpe.encode(text, &special_tokens_refs);
                // Tiktoken doesn't provide the same level of detail as HuggingFace
                Ok(EncodingDetails {
                    ids: tokens,
                    type_ids: None,
                    tokens: None,
                    special_tokens_mask: None,
                    attention_mask: None,
                    offsets: None,
                })
            }
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            UnifiedTokenizer::HuggingFace(tokenizer) => {
                tokenizer.decode(ids, skip_special_tokens)
                    .map_err(|e| format!("Decoding error: {}", e).into())
            }
            UnifiedTokenizer::Tiktoken(bpe, _vocab_size, _special_tokens, special_token_ids) => {
                let tokens_to_decode = if skip_special_tokens {
                    ids.iter()
                        .filter(|id| !special_token_ids.contains(id))
                        .copied()
                        .collect::<Vec<u32>>()
                } else {
                    ids.to_vec()
                };
                // Handle decode errors by trying to decode progressively fewer tokens
                // until we find a valid UTF-8 sequence, then add replacement characters
                // for the remaining tokens
                match bpe.decode(tokens_to_decode.clone()) {
                    Ok(decoded) => Ok(decoded),
                    Err(_) if tokens_to_decode.is_empty() => Ok(String::new()),
                    Err(_) => {
                        // Try decoding progressively fewer tokens from the end
                        let mut valid_prefix_len = tokens_to_decode.len() - 1;
                        let mut decoded_prefix = String::new();
                        
                        while valid_prefix_len > 0 {
                            if let Ok(decoded) = bpe.decode(tokens_to_decode[..valid_prefix_len].to_vec()) {
                                decoded_prefix = decoded;
                                break;
                            }
                            valid_prefix_len -= 1;
                        }
                        
                        let remaining_tokens = tokens_to_decode.len() - valid_prefix_len;
                        if remaining_tokens > 0 {
                            decoded_prefix.push_str(&"\u{FFFD}".repeat(remaining_tokens));
                        }
                        
                        Ok(decoded_prefix)
                    }
                }
            }
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            UnifiedTokenizer::HuggingFace(tokenizer) => tokenizer.get_vocab_size(true) as u32,
            UnifiedTokenizer::Tiktoken(_bpe, vocab_size, _special_tokens, _special_token_ids) => *vocab_size
        }
    }

    pub fn set_encode_special_tokens(&mut self, encode_special_tokens: bool) {
        match self {
            UnifiedTokenizer::HuggingFace(ref mut tokenizer) => {
                tokenizer.set_encode_special_tokens(encode_special_tokens);
            }
            UnifiedTokenizer::Tiktoken(_, _, _, _) => {
                // Silently ignore for Tiktoken since it doesn't support this operation
                // This is safer than panicking in a library
            }
        }
    }
    
    pub fn supports_encode_special_tokens(&self) -> bool {
        matches!(self, UnifiedTokenizer::HuggingFace(_))
    }


}

pub struct EncodingDetails {
    pub ids: Vec<u32>,
    pub type_ids: Option<Vec<u32>>,
    pub tokens: Option<Vec<String>>,
    pub special_tokens_mask: Option<Vec<u32>>,
    pub attention_mask: Option<Vec<u32>>,
    pub offsets: Option<Vec<(usize, usize)>>,
}

#[repr(C)]
pub struct tokenizers_options {
    encode_special_tokens: bool,
}

#[repr(C)]
pub struct tokenizers_buffer {
    ids: *mut u32,
    type_ids: *mut u32,
    special_tokens_mask: *mut u32,
    attention_mask: *mut u32,
    tokens: *mut *mut libc::c_char,
    offsets: *mut usize,
    len: usize,
}

#[no_mangle]
pub extern "C" fn tokenizers_from_bytes(bytes: *const u8, len: u32, opts: &tokenizers_options, error: *mut *mut libc::c_char) -> *mut libc::c_void {
    if bytes.is_null() {
        if !error.is_null() {
            let err_msg = std::ffi::CString::new("Bytes pointer is null").unwrap();
            unsafe { *error = err_msg.into_raw(); }
        }
        return ptr::null_mut();
    }
    
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    match Tokenizer::from_bytes(bytes_slice) {
        Ok(mut tokenizer) => {
            tokenizer.set_encode_special_tokens(opts.encode_special_tokens);
            let unified = UnifiedTokenizer::HuggingFace(tokenizer);
            Box::into_raw(Box::new(unified)).cast()
        }
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Failed to create tokenizer from bytes: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            ptr::null_mut()
        }
    }
}

// TODO merge with from_bytes and pass truncation params as an argument to TokenizerOptions
#[no_mangle]
pub extern "C" fn tokenizers_from_bytes_with_truncation(bytes: *const u8, len: u32, max_len: usize, dir: u8, error: *mut *mut libc::c_char) -> *mut libc::c_void {
    if bytes.is_null() {
        if !error.is_null() {
            let err_msg = std::ffi::CString::new("Bytes pointer is null").unwrap();
            unsafe { *error = err_msg.into_raw(); }
        }
        return ptr::null_mut();
    }
    
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let direction = match TruncationDirection::from_u8(dir) {
        Some(d) => d.to_tokenizers_direction(),
        None => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Invalid truncation direction: {}", dir)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            return ptr::null_mut();
        }
    };
    
    match Tokenizer::from_bytes(bytes_slice) {
        Ok(mut tokenizer) => {
            let truncation_params = tokenizers::tokenizer::TruncationParams {
                max_length: max_len,
                direction,
                ..Default::default()
            };
            match tokenizer.with_truncation(Some(truncation_params)) {
                Ok(tokenizer_with_truncation) => {
                    let unified = UnifiedTokenizer::HuggingFace(tokenizer_with_truncation.to_owned().into());
                    Box::into_raw(Box::new(unified)).cast()
                }
                Err(e) => {
                    if !error.is_null() {
                        let err_msg = std::ffi::CString::new(format!("Failed to set truncation parameters: {}", e)).unwrap();
                        unsafe { *error = err_msg.into_raw(); }
                    }
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Failed to create tokenizer from bytes: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_from_file(config: *const libc::c_char, error: *mut *mut libc::c_char) -> *mut libc::c_void {
    if config.is_null() {
        if !error.is_null() {
            let err_msg = std::ffi::CString::new("Config path is null").unwrap();
            unsafe { *error = err_msg.into_raw(); }
        }
        return ptr::null_mut();
    }
    
    let config_cstr = unsafe { CStr::from_ptr(config) };
    let config_str = match config_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Invalid UTF-8 in config path: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            return ptr::null_mut();
        }
    };
    
    let config_path = PathBuf::from(config_str);
    match Tokenizer::from_file(&config_path) {
        Ok(tokenizer) => {
            let unified = UnifiedTokenizer::HuggingFace(tokenizer);
            let ptr = Box::into_raw(Box::new(unified));
            ptr.cast()
        }
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Failed to load tokenizer from file '{}': {}", config_str, e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_from_tiktoken(
    model_file: *const libc::c_char,
    config_file: *const libc::c_char,
    pattern: *const libc::c_char,
    error: *mut *mut libc::c_char,
) -> *mut libc::c_void {
    if model_file.is_null() || config_file.is_null() || pattern.is_null() {
        if !error.is_null() {
            let err_msg = std::ffi::CString::new("One or more required parameters are null").unwrap();
            unsafe { *error = err_msg.into_raw(); }
        }
        return ptr::null_mut();
    }
    
    let model_file_cstr = unsafe { CStr::from_ptr(model_file) };
    let model_file_str = match model_file_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Invalid UTF-8 in model file path: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            return ptr::null_mut();
        }
    };
    
    let config_file_cstr = unsafe { CStr::from_ptr(config_file) };
    let config_file_str = match config_file_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Invalid UTF-8 in config file path: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            return ptr::null_mut();
        }
    };
    
    let pattern_cstr = unsafe { CStr::from_ptr(pattern) };
    let pattern_str = match pattern_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Invalid UTF-8 in pattern: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            return ptr::null_mut();
        }
    };
    
    match create_tiktoken_encoder(model_file_str, config_file_str, pattern_str) {
        Ok((bpe, vocab_size, special_tokens, special_token_ids)) => {
            let unified = UnifiedTokenizer::Tiktoken(bpe, vocab_size, special_tokens, special_token_ids);
            Box::into_raw(Box::new(unified)).cast()
        }
        Err(e) => {
            if !error.is_null() {
                let err_msg = std::ffi::CString::new(format!("Failed to create tiktoken tokenizer: {}", e)).unwrap();
                unsafe { *error = err_msg.into_raw(); }
            }
            ptr::null_mut()
        }
    }
}

#[repr(C)]
pub struct tokenizers_encode_options {
    add_special_tokens: bool,

    return_type_ids: bool,
    return_tokens: bool,
    return_special_tokens_mask: bool,
    return_attention_mask: bool,
    return_offsets: bool,
}

#[no_mangle]
pub extern "C" fn tokenizers_encode(ptr: *mut libc::c_void, message: *const libc::c_char, options: &tokenizers_encode_options) -> tokenizers_buffer {
    if ptr.is_null() || message.is_null() {
        return tokenizers_buffer { 
            ids: ptr::null_mut(), 
            tokens: ptr::null_mut(), 
            len: 0, 
            type_ids: ptr::null_mut(), 
            special_tokens_mask: ptr::null_mut(), 
            attention_mask: ptr::null_mut(), 
            offsets: ptr::null_mut()
        };
    }
    
    let unified_tokenizer = unsafe {
        match ptr.cast::<UnifiedTokenizer>().as_ref() {
            Some(tokenizer) => tokenizer,
            None => return tokenizers_buffer { 
                ids: ptr::null_mut(), 
                tokens: ptr::null_mut(), 
                len: 0, 
                type_ids: ptr::null_mut(), 
                special_tokens_mask: ptr::null_mut(), 
                attention_mask: ptr::null_mut(), 
                offsets: ptr::null_mut()
            }
        }
    };
    
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message_bytes = message_cstr.to_bytes();
    
    // Use from_utf8_lossy to handle invalid UTF-8 gracefully
    // This will replace invalid sequences with the replacement character (U+FFFD)
    let message_cow = String::from_utf8_lossy(message_bytes);
    let message = message_cow.as_ref();

    let encoding_details = match std::panic::catch_unwind(|| { unified_tokenizer.encode_with_details(message, options.add_special_tokens) }) {
        Ok(Ok(details)) => details,
        Ok(Err(_)) | Err(_) => return tokenizers_buffer {
            ids: ptr::null_mut(),
            tokens: ptr::null_mut(),
            len: 0,
            type_ids: ptr::null_mut(),
            special_tokens_mask: ptr::null_mut(),
            attention_mask: ptr::null_mut(),
            offsets: ptr::null_mut()
        }
    };
    
    let mut vec_ids = encoding_details.ids;
    vec_ids.shrink_to_fit();
    let ids = vec_ids.as_mut_ptr();
    let len = vec_ids.len();
    std::mem::forget(vec_ids);

    let mut type_ids: *mut u32 = ptr::null_mut();
    if options.return_type_ids {
        if let Some(mut vec_type_ids) = encoding_details.type_ids {
            vec_type_ids.shrink_to_fit();
            type_ids = vec_type_ids.as_mut_ptr();
            std::mem::forget(vec_type_ids);
        }
    }

    let mut tokens: *mut *mut libc::c_char = ptr::null_mut();
    if options.return_tokens {
        if let Some(token_strings) = encoding_details.tokens {
            let mut vec_tokens = token_strings.into_iter()
                .map(|s| std::ffi::CString::new(s).unwrap().into_raw())
                .collect::<Vec<_>>();
            vec_tokens.shrink_to_fit();
            tokens = vec_tokens.as_mut_ptr();
            std::mem::forget(vec_tokens);
        }
    }

    let mut special_tokens_mask: *mut u32 = ptr::null_mut();
    if options.return_special_tokens_mask {
        if let Some(mut vec_special_tokens_mask) = encoding_details.special_tokens_mask {
            vec_special_tokens_mask.shrink_to_fit();
            special_tokens_mask = vec_special_tokens_mask.as_mut_ptr();
            std::mem::forget(vec_special_tokens_mask);
        }
    }

    let mut attention_mask: *mut u32 = ptr::null_mut();
    if options.return_attention_mask {
        if let Some(mut vec_attention_mask) = encoding_details.attention_mask {
            vec_attention_mask.shrink_to_fit();
            attention_mask = vec_attention_mask.as_mut_ptr();
            std::mem::forget(vec_attention_mask);
        }
    }

    let mut offsets: *mut usize = ptr::null_mut();
    if options.return_offsets {
        if let Some(vec_offsets_tuples) = encoding_details.offsets {
            let mut vec_offsets = Vec::with_capacity(vec_offsets_tuples.len() * 2);
            for i in vec_offsets_tuples {
                vec_offsets.push(i.0);
                vec_offsets.push(i.1);
            }
            vec_offsets.shrink_to_fit();
            offsets = vec_offsets.as_mut_ptr();
            std::mem::forget(vec_offsets);
        }
    }

    tokenizers_buffer { ids, type_ids, special_tokens_mask, attention_mask, tokens, offsets, len }
}

#[no_mangle]
pub extern "C" fn tokenizers_decode(ptr: *mut libc::c_void, ids: *const u32, len: u32, skip_special_tokens: bool) -> *mut libc::c_char {
    if ptr.is_null() || ids.is_null() {
        return ptr::null_mut();
    }
    
    let unified_tokenizer = unsafe {
        match ptr.cast::<UnifiedTokenizer>().as_ref() {
            Some(tokenizer) => tokenizer,
            None => return ptr::null_mut()
        }
    };
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };

    match unified_tokenizer.decode(ids_slice, skip_special_tokens) {
        Ok(string) => match std::ffi::CString::new(string) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_vocab_size(ptr: *mut libc::c_void) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    
    let unified_tokenizer = unsafe {
        match ptr.cast::<UnifiedTokenizer>().as_ref() {
            Some(tokenizer) => tokenizer,
            None => return 0
        }
    };
    unified_tokenizer.vocab_size()
}

#[no_mangle]
pub extern "C" fn tokenizers_free_tokenizer(ptr: *mut ::libc::c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr.cast::<UnifiedTokenizer>()));
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free_buffer(buf: tokenizers_buffer) {
    if !buf.ids.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.ids, buf.len, buf.len);
        }
    }
    if !buf.type_ids.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.type_ids, buf.len, buf.len);
        }
    }
    if !buf.special_tokens_mask.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.special_tokens_mask, buf.len, buf.len);
        }
    }
    if !buf.attention_mask.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.attention_mask, buf.len, buf.len);
        }
    }
    if !buf.offsets.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.offsets, buf.len*2, buf.len*2);
        }
    }
    if !buf.tokens.is_null() {
        unsafe {
            let strings = Vec::from_raw_parts(buf.tokens, buf.len, buf.len);
            for s in strings {
                drop(std::ffi::CString::from_raw(s.cast::<libc::c_char>()));
            }   
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free_string(ptr: *mut libc::c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(std::ffi::CString::from_raw(ptr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Common tiktoken pattern for models like Kimi
    const TIKTOKEN_PATTERN_KIMI: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// Common tiktoken pattern for models like GPT-4 (cl100k_base)
    const TIKTOKEN_PATTERN_CL100K_BASE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    #[test]
    fn test_tiktoken() -> Result<(), Box<dyn std::error::Error>> {
        // Use the constant pattern for tokenization
        let (bpe, _vocab_size, _special_tokens, _special_token_ids) = crate::create_tiktoken_encoder(
            "test/data/kimi-k2-instruct/tiktoken.model",
            "test/data/kimi-k2-instruct/tokenizer_config.json",
            TIKTOKEN_PATTERN_KIMI
        )?;

        // Test encoding and decoding
        let tokens = bpe.encode("Hello, world! 你好，世界！", &HashSet::new()).0;
        assert_eq!(tokens, vec![19180, 11, 2695, 0, 220, 33845, 378, 2243, 856]);

        let decoded = bpe.decode(tokens)?;
        assert_eq!(decoded, "Hello, world! 你好，世界！");

        Ok(())
    }

    /// Create a test tiktoken unified tokenizer
    fn create_test_tiktoken_tokenizer() -> Result<UnifiedTokenizer, Box<dyn std::error::Error>> {
        let (bpe, vocab_size, special_tokens, special_token_ids) = create_tiktoken_encoder(
            "test/data/kimi-k2-instruct/tiktoken.model",
            "test/data/kimi-k2-instruct/tokenizer_config.json",
            TIKTOKEN_PATTERN_KIMI
        )?;
        Ok(UnifiedTokenizer::Tiktoken(bpe, vocab_size, special_tokens, special_token_ids))
    }

    /// Create a test Llama 3 tiktoken unified tokenizer
    fn create_test_llama_tokenizer() -> Result<UnifiedTokenizer, Box<dyn std::error::Error>> {
        let (bpe, vocab_size, special_tokens, special_token_ids) = create_tiktoken_encoder(
            "test/data/meta-llama-3-8b-instruct/tiktoken.model",
            "test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
            TIKTOKEN_PATTERN_CL100K_BASE
        )?;
        Ok(UnifiedTokenizer::Tiktoken(bpe, vocab_size, special_tokens, special_token_ids))
    }

    #[test]
    fn test_unified_huggingface() -> Result<(), Box<dyn std::error::Error>> {
        // Test with HuggingFace tokenizer
        let tokenizer = Tokenizer::from_file("test/data/bert-base-uncased.json").map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        let unified = UnifiedTokenizer::HuggingFace(tokenizer);
        
        let text = "Hello, world!";
        let ids = unified.encode(text, false)?;
        assert!(!ids.is_empty());
        
        let decoded = unified.decode(&ids, false)?;
        assert_eq!(decoded.to_lowercase(), text.to_lowercase());
        
        let vocab_size = unified.vocab_size();
        assert!(vocab_size > 0);
        
        Ok(())
    }

    #[test]
    fn test_unified_tiktoken() -> Result<(), Box<dyn std::error::Error>> {
        // Test basic tiktoken functionality, including multilingual support and vocab size
        let unified = create_test_tiktoken_tokenizer()?;
        
        // Test English encoding/decoding
        let text_en = "Hello, world!";
        let ids_en = unified.encode(text_en, false)?;
        assert!(!ids_en.is_empty());
        let decoded_en = unified.decode(&ids_en, false)?;
        assert_eq!(decoded_en, text_en);
        
        // Test multilingual encoding/decoding
        let text_multi = "Hello, world! 你好，世界！";
        let ids_multi = unified.encode(text_multi, false)?;
        assert!(!ids_multi.is_empty());
        assert_eq!(ids_multi, vec![19180, 11, 2695, 0, 220, 33845, 378, 2243, 856]);
        let decoded_multi = unified.decode(&ids_multi, false)?;
        assert_eq!(decoded_multi, text_multi);
        
        // Test vocab size
        let vocab_size = unified.vocab_size();
        assert_eq!(vocab_size, 163840);
        
        Ok(())
    }

    #[test]
    fn test_tiktoken_replacement_character() -> Result<(), Box<dyn std::error::Error>> {
        // Test that tiktoken can handle the replacement character (U+FFFD)
        let unified = create_test_llama_tokenizer()?;
        
        // Test cases with replacement character
        let test_cases = vec![
            ("Hello world", "Normal text should work"),
            ("Hello �world", "Text with replacement character should work"),
            ("Test � multiple � chars", "Multiple replacement characters should work"),
            ("�", "Just replacement character should work"),
            ("\u{FFFD}", "Unicode escape replacement character should work"),
            (std::str::from_utf8(&[0xEF, 0xBF, 0xBD]).unwrap(), "UTF-8 bytes of replacement character should work"),
            ("歪", "Multi token UTF-8 should work"),
        ];
        
        for (text, description) in test_cases {
            let ids = unified.encode(text, false)?;
            assert!(!ids.is_empty(), "{}: Expected non-empty token IDs for text: {:?}", description, text);

            let decoded = unified.decode(&ids, false)?;
            assert_eq!(decoded, text, "{}: Decoded text should match original", description);
            }
        
        Ok(())
    }

    #[test]
    fn test_partial_decode() -> Result<(), Box<dyn std::error::Error>> {
        // Test that tiktoken can handle incomplete UTF-8 by returning replacement characters
        let unified = create_test_llama_tokenizer()?;
        
        // Test cases for partial and complete UTF-8 sequences
        let test_cases = vec![
            (vec![15722], "\u{FFFD}", "Partial decode should return replacement character"),
            (vec![15722, 103], "歪", "Complete UTF-8 should decode correctly"),
            (vec![15722, 103, 15722], "歪\u{FFFD}", "Partial decode should return replacement character"),
            (vec![15722, 103, 15722, 103], "歪歪", "Complete UTF-8 should decode correctly"),
            (vec![15722, 103, 15722, 15722], "歪\u{FFFD}\u{FFFD}", "Multiple non-decoded tokens should return replacement characters"),
        ];

        for (ids, expected, description) in test_cases {
            let decoded = unified.decode(&ids, false)?;
            assert_eq!(decoded, expected, "{}: Decoded text should match expected", description);
        }
        
        Ok(())
    }

    #[test]
    fn test_partial_decode_multiple_tokens() -> Result<(), Box<dyn std::error::Error>> {
        // Test the new approach with various scenarios
        let unified = create_test_llama_tokenizer()?;
        
        // Encode "Hello 歪 world" and test various partial decodes
        let full_text = "Hello 歪 world";
        let ids = unified.encode(full_text, false)?;
        
        // Full decode should work
        let decoded = unified.decode(&ids, false)?;
        assert_eq!(decoded, full_text, "Full decode should work correctly");
        
        // Test decoding with the last token removed (might break UTF-8)
        if ids.len() > 1 {
            let partial_ids = &ids[..ids.len()-1];
            let decoded_partial = unified.decode(partial_ids, false)?;
            // This should either decode correctly or have replacement chars at the end
            assert!(decoded_partial.contains("Hello"), "Partial decode should preserve valid prefix");
        }
        
        // Test with just the incomplete UTF-8 tokens
        let incomplete_utf8_ids = vec![15722]; // Just the first part of "歪"
        let decoded_incomplete = unified.decode(&incomplete_utf8_ids, false)?;
        assert_eq!(decoded_incomplete, "\u{FFFD}", "Single incomplete token should be replacement char");
        
        // Test with multiple tokens where the last few form incomplete UTF-8
        let text_with_emoji = "Test 👋";
        let emoji_ids = unified.encode(text_with_emoji, false)?;
        if emoji_ids.len() > 2 {
            // Remove last token to potentially break the emoji
            let partial_emoji_ids = &emoji_ids[..emoji_ids.len()-1];
            let decoded_partial_emoji = unified.decode(partial_emoji_ids, false)?;
            assert!(decoded_partial_emoji.starts_with("Test"), "Should preserve 'Test' prefix");
            // The emoji part should either decode correctly or be replaced
        }
        
        Ok(())
    }

    #[test]
    fn test_unified_llama() -> Result<(), Box<dyn std::error::Error>> {
        // Test Llama 3 tiktoken functionality
        let unified = create_test_llama_tokenizer()?;
        
        // Test basic English encoding/decoding
        let text_en = "Hello, world!";
        let ids_en = unified.encode(text_en, false)?;
        assert!(!ids_en.is_empty());
        let decoded_en = unified.decode(&ids_en, false)?;
        assert_eq!(decoded_en, text_en);
        
        // Test contractions and punctuation
        let text_contractions = "I'm here, you're there. It's great!";
        let ids_contractions = unified.encode(text_contractions, false)?;
        assert!(!ids_contractions.is_empty());
        let decoded_contractions = unified.decode(&ids_contractions, false)?;
        assert_eq!(decoded_contractions, text_contractions);
        
        // Test multilingual support with various languages
        let text_multilingual = "Hello world! 你好世界！ مرحبا بالعالم Bonjour le monde! 🌍";
        let ids_multilingual = unified.encode(text_multilingual, false)?;
        assert!(!ids_multilingual.is_empty());
        let decoded_multilingual = unified.decode(&ids_multilingual, false)?;
        assert_eq!(decoded_multilingual, text_multilingual);
        
        // Test numbers and mixed content
        let text_mixed = "Test123: The price is $99.99 USD (≈€92.50 EUR)";
        let ids_mixed = unified.encode(text_mixed, false)?;
        assert!(!ids_mixed.is_empty());
        let decoded_mixed = unified.decode(&ids_mixed, false)?;
        assert_eq!(decoded_mixed, text_mixed);
        
        // Test vocab size (Llama 3 has 128,256 base tokens)
        let vocab_size = unified.vocab_size();
        assert!(vocab_size >= 128256, "Llama 3 vocab size should be at least 128256, got {}", vocab_size);
        
        Ok(())
    }

    #[test]
    fn test_llama_special_tokens() -> Result<(), Box<dyn std::error::Error>> {
        // Test special token parsing and handling for Llama 3
        let (_bpe, _vocab_size, special_tokens, special_token_ids) = create_tiktoken_encoder(
            "test/data/meta-llama-3-8b-instruct/tiktoken.model",
            "test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
            TIKTOKEN_PATTERN_CL100K_BASE
        )?;
        
        // Verify special tokens were parsed
        assert!(!special_tokens.is_empty(), "Special tokens should have been parsed from config");
        assert!(!special_token_ids.is_empty(), "Special token IDs should have been parsed");
        
        // Check for expected Llama special tokens
        let has_begin_text = special_tokens.iter().any(|t| t == "<|begin_of_text|>");
        let has_end_text = special_tokens.iter().any(|t| t == "<|end_of_text|>");
        let has_start_header = special_tokens.iter().any(|t| t == "<|start_header_id|>");
        let has_end_header = special_tokens.iter().any(|t| t == "<|end_header_id|>");
        let has_eot = special_tokens.iter().any(|t| t == "<|eot_id|>");
        
        assert!(has_begin_text, "Expected to find <|begin_of_text|> special token");
        assert!(has_end_text, "Expected to find <|end_of_text|> special token");
        assert!(has_start_header, "Expected to find <|start_header_id|> special token");
        assert!(has_end_header, "Expected to find <|end_header_id|> special token");
        assert!(has_eot, "Expected to find <|eot_id|> special token");
        
        // Verify special token IDs are in the expected range (128000-128255 for Llama 3)
        assert!(special_token_ids.contains(&128000), "Should contain begin_of_text token ID");
        assert!(special_token_ids.contains(&128001), "Should contain end_of_text token ID");
        assert!(special_token_ids.contains(&128006), "Should contain start_header_id token ID");
        assert!(special_token_ids.contains(&128007), "Should contain end_header_id token ID");
        assert!(special_token_ids.contains(&128009), "Should contain eot_id token ID");
        
        Ok(())
    }

    #[test]
    fn test_llama_skip_special_tokens_decoding() -> Result<(), Box<dyn std::error::Error>> {
        // Test that skip_special_tokens correctly filters out special token IDs during decoding
        let unified = create_test_llama_tokenizer()?;
        
        // Create a sequence with special token IDs
        // Example: <|begin_of_text|>Hello, world!<|end_of_text|>
        let ids_with_special = vec![128000, 9906, 11, 1917, 0, 128001]; // [begin_of_text] Hello, world! [end_of_text]
        
        // Decode without skipping special tokens
        let decoded_with_special = unified.decode(&ids_with_special, false)?;
        assert!(decoded_with_special.contains("<|begin_of_text|>") || decoded_with_special.contains("<|end_of_text|>"), 
                "Decoded text should contain special tokens when skip_special_tokens is false");
        
        // Decode with skipping special tokens
        let decoded_skip_special = unified.decode(&ids_with_special, true)?;
        assert!(!decoded_skip_special.contains("<|begin_of_text|>"), 
                "Decoded text should NOT contain <|begin_of_text|> when skip_special_tokens is true");
        assert!(!decoded_skip_special.contains("<|end_of_text|>"), 
                "Decoded text should NOT contain <|end_of_text|> when skip_special_tokens is true");
        assert_eq!(decoded_skip_special, "Hello, world!", 
                "Decoded text should only contain the regular tokens");
        
        Ok(())
    }

    #[test]
    fn test_llama_chat_template_tokens() -> Result<(), Box<dyn std::error::Error>> {
        // Test encoding/decoding of chat template patterns
        let unified = create_test_llama_tokenizer()?;
        
        // Test typical chat template markers
        let chat_markers = vec![
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ];
        
        for marker in chat_markers {
            let ids = unified.encode(marker, true)?;
            assert!(!ids.is_empty(), "Chat marker '{}' should be encoded", marker);
            
            let decoded = unified.decode(&ids, false)?;
            assert_eq!(decoded, marker, "Chat marker should decode correctly");
        }
        
        // Test a full chat template sequence
        let chat_sequence = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|>";
        let ids = unified.encode(chat_sequence, true)?;
        assert!(!ids.is_empty(), "Chat sequence should be encoded");
        
        let decoded = unified.decode(&ids, false)?;
        assert_eq!(decoded, chat_sequence, "Chat sequence should decode correctly");
        
        Ok(())
    }

    #[test]
    fn test_tiktoken_special_tokens() -> Result<(), Box<dyn std::error::Error>> {
        // Test special token parsing and handling
        let (_bpe, _vocab_size, special_tokens, special_token_ids) = create_tiktoken_encoder(
            "test/data/kimi-k2-instruct/tiktoken.model",
            "test/data/kimi-k2-instruct/tokenizer_config.json",
            TIKTOKEN_PATTERN_KIMI
        )?;
        
        // Verify special tokens were parsed
        assert!(!special_tokens.is_empty(), "Special tokens should have been parsed from config");
        assert!(!special_token_ids.is_empty(), "Special token IDs should have been parsed");
        
        // Check for expected special tokens
        let has_bos = special_tokens.iter().any(|t| t == "[BOS]");
        let has_eos = special_tokens.iter().any(|t| t == "[EOS]");
        let has_im_end = special_tokens.iter().any(|t| t == "<|im_end|>");
        assert!(has_bos || has_eos || has_im_end, "Expected to find at least one special token");
        
        // Verify special token IDs are in the expected range
        assert!(special_token_ids.contains(&163584), "Should contain BOS token ID");
        assert!(special_token_ids.contains(&163585), "Should contain EOS token ID");
        
        Ok(())
    }

    #[test]
    fn test_tiktoken_skip_special_tokens_decoding() -> Result<(), Box<dyn std::error::Error>> {
        // Test that skip_special_tokens correctly filters out special token IDs during decoding
        let unified = create_test_tiktoken_tokenizer()?;
        
        // Create a sequence with special token IDs
        let ids_with_special = vec![163584, 19180, 11, 2695, 0, 163585]; // [BOS] Hello, world! [EOS]
        
        // Decode without skipping special tokens
        let decoded_with_special = unified.decode(&ids_with_special, false)?;
        assert!(decoded_with_special.contains("[BOS]") || decoded_with_special.contains("[EOS]"), 
                "Decoded text should contain special tokens when skip_special_tokens is false");
        
        // Decode with skipping special tokens
        let decoded_skip_special = unified.decode(&ids_with_special, true)?;
        assert!(!decoded_skip_special.contains("[BOS]"), 
                "Decoded text should NOT contain [BOS] when skip_special_tokens is true");
        assert!(!decoded_skip_special.contains("[EOS]"), 
                "Decoded text should NOT contain [EOS] when skip_special_tokens is true");
        assert_eq!(decoded_skip_special, "Hello, world!", 
                "Decoded text should only contain the regular tokens");
        
        Ok(())
    }
}

/// Creates a CoreBPE encoder from a model file, tokenizer config file, and pattern string.
/// 
/// # Arguments
/// * `model_file_path` - Path to the .model file containing base64 encoded tokens and ranks
/// * `config_file_path` - Path to the tokenizer_config.json file containing special tokens
/// * `pattern` - Regex pattern string for tokenization
/// 
/// # Returns
/// A tuple containing:
/// * `CoreBPE` - The tiktoken encoder instance
/// * `u32` - The vocabulary size
/// * `HashSet<String>` - Set of special token strings
/// * `HashSet<u32>` - Set of special token IDs
/// 
/// # Errors
/// Returns an error if:
/// * File reading fails
/// * Model file format is invalid
/// * Base64 decoding fails
/// * JSON parsing fails
pub fn create_tiktoken_encoder(
    model_file_path: &str,
    config_file_path: &str,
    pattern: &str,
) -> Result<(tiktoken_rs::CoreBPE, u32, std::collections::HashSet<String>, std::collections::HashSet<u32>), Box<dyn std::error::Error>> {
    use std::collections::{HashMap, HashSet};
    use tiktoken_rs::{CoreBPE, Rank};
    use base64::{Engine as _, engine::general_purpose};

    let mut encoder: HashMap<Vec<u8>, Rank, std::hash::BuildHasherDefault<rustc_hash::FxHasher>> =
        HashMap::default();

    // Parse the model file
    let file = std::fs::read_to_string(model_file_path)
        .map_err(|e| format!("Failed to read model file: {}", e))?;
    
    for (line_num, line) in file.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        
        let mut parts = line.split(' ');
        let raw = parts.next()
            .ok_or_else(|| format!("Invalid model file format at line {}: missing token", line_num + 1))?;
        let token = general_purpose::STANDARD.decode(raw)
            .map_err(|e| format!("Failed to decode base64 at line {}: {}", line_num + 1, e))?;
        let rank: Rank = parts.next()
            .ok_or_else(|| format!("Invalid model file format at line {}: missing rank", line_num + 1))?
            .parse()
            .map_err(|e| format!("Failed to parse rank at line {}: {}", line_num + 1, e))?;
        encoder.insert(token, rank);
    }

    // Parse special tokens from config
    let mut special_tokens: HashMap<String, u32, std::hash::BuildHasherDefault<rustc_hash::FxHasher>> = 
        HashMap::default();
    let mut special_tokens_set = HashSet::new();
    let mut special_token_ids = HashSet::new();
    {
        let config_file = std::fs::File::open(config_file_path)
            .map_err(|e| format!("Failed to open config file: {}", e))?;
        let tokenizer_config: TokenizerConfig = serde_json::from_reader(config_file)
            .map_err(|e| format!("Failed to parse config JSON: {}", e))?;
        
        for (token_id, added_token) in tokenizer_config.added_tokens_decoder {
            let id: u32 = token_id.parse()
                .map_err(|e| format!("Failed to parse token ID '{}': {}", token_id, e))?;
            special_tokens.insert(added_token.content.clone(), id);
            special_tokens_set.insert(added_token.content);
            special_token_ids.insert(id);
        }
    }

    // Calculate vocab size more robustly
    // The vocab size should be the maximum of:
    // 1. The highest rank in the base vocabulary
    // 2. The highest special token ID
    // Plus 1 to account for 0-based indexing
    let max_rank = encoder.values().copied().max().unwrap_or(0);
    let max_special_token = special_tokens.values().copied().max().unwrap_or(0);
    let vocab_size = std::cmp::max(max_rank, max_special_token) + 1;

    // Fill in missing ranks with reserved special tokens
    // This ensures the encoder has entries for all ranks from 0 to max_rank
    let existing_ranks: HashSet<Rank> = encoder.values().copied().collect();
    let mut reserved_token_count: u32 = 0;
    for rank in 0..max_rank {
        if !existing_ranks.contains(&rank) {
            let reserved_token = format!("<|reserved_special_token_{}|>", reserved_token_count);
            reserved_token_count += 1;
            encoder.insert(reserved_token.into_bytes(), rank);
        }
    }

    let bpe = CoreBPE::new(encoder, special_tokens, pattern)?;
    Ok((bpe, vocab_size, special_tokens_set, special_token_ids))
}


// Structures for deserializing tokenizer_config.json
#[derive(Debug, Deserialize, Serialize)]
pub struct TokenizerConfig {
    added_tokens_decoder: HashMap<String, AddedToken>,
    additional_special_tokens: Option<Vec<String>>,
    bos_token: String,
    eos_token: String,
    unk_token: Option<String>,
    pad_token: Option<String>,
    model_max_length: f64,  // Changed from u32 to handle very large values
    tokenizer_class: String,
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AddedToken {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: bool,
}
