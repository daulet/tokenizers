use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr::null_mut;
use tokenizers::Encoding;

use tokenizers::tokenizer::Tokenizer;

#[repr(C)]
pub struct Offset {
    start: usize,
    end: usize,
}

#[repr(C)]
pub struct Buffer {
    ids: *mut u32,
    type_ids: *mut u32,
    special_tokens_mask: *mut u32,
    attention_mask: *mut u32,
    tokens: *mut *mut libc::c_char,
    offsets: *mut Offset,
    len: usize,
}

#[repr(C)]
pub struct EncodeOptions {
    add_special_tokens: bool,

    return_type_ids: bool,
    return_special_tokens_mask: bool,
    return_attention_mask: bool,
    return_offsets: bool,

    with_offsets_char_mode: bool,
}

/// # Safety
///
/// This function is return Tokenizer object to Golang from
/// after reading tokenizer.json to bytes
#[no_mangle]
pub unsafe extern "C" fn from_bytes(bytes: *const u8, len: u32) -> *mut Tokenizer {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let tokenizer = Tokenizer::from_bytes(bytes_slice).expect("failed to create tokenizer");

    Box::into_raw(Box::new(tokenizer))
}

/// # Safety
///
/// This function is return Tokenizer(truncation mode) object to Golang from
/// after read tokenizer.json to bytes
#[no_mangle]
pub unsafe extern "C" fn from_bytes_with_truncation(
    bytes: *const u8,
    len: u32,
    max_len: usize,
    dir: u8,
) -> *mut Tokenizer {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let tokenizer: Tokenizer = Tokenizer::from_bytes(bytes_slice)
        .expect("failed to create tokenizer")
        .with_truncation(Some(tokenizers::tokenizer::TruncationParams {
            max_length: max_len,
            direction: match dir {
                0 => tokenizers::tokenizer::TruncationDirection::Left,
                1 => tokenizers::tokenizer::TruncationDirection::Right,
                _ => panic!("invalid truncation direction"),
            },
            ..Default::default()
        }))
        .unwrap()
        .to_owned()
        .into();

    Box::into_raw(Box::new(tokenizer))
}

/// # Safety
///
/// This function is return Tokenizer object to Golang from tokenizer.json
#[no_mangle]
pub unsafe extern "C" fn from_file(config: *const libc::c_char) -> *mut libc::c_void {
    let config_cstr = unsafe { CStr::from_ptr(config) };
    let config = config_cstr.to_str().unwrap();
    let config = PathBuf::from(config);
    match Tokenizer::from_file(config) {
        Ok(tokenizer) => {
            let ptr = Box::into_raw(Box::new(tokenizer));
            ptr.cast()
        }
        Err(_) => null_mut(),
    }
}

fn encode_process(encoding: Encoding, options: &EncodeOptions) -> Buffer {
    // ids, tokens
    let mut vec_ids = encoding.get_ids().to_vec();
    let mut vec_tokens = encoding
        .get_tokens()
        .iter()
        .cloned()
        .map(|s| std::ffi::CString::new(s).unwrap().into_raw())
        .collect::<Vec<_>>();
    vec_ids.shrink_to_fit();
    vec_tokens.shrink_to_fit();
    let ids = vec_ids.as_mut_ptr();
    let tokens = vec_tokens.as_mut_ptr();
    let len = vec_ids.len();
    std::mem::forget(vec_ids);
    std::mem::forget(vec_tokens);

    // type_ids
    let mut type_ids: *mut u32 = null_mut();
    if options.return_type_ids {
        let mut vec_type_ids = encoding.get_type_ids().to_vec();
        vec_type_ids.shrink_to_fit();
        type_ids = vec_type_ids.as_mut_ptr();
        std::mem::forget(vec_type_ids);
    }

    // special_tokens_mask
    let mut special_tokens_mask: *mut u32 = null_mut();
    if options.return_special_tokens_mask {
        let mut vec_special_tokens_mask = encoding.get_special_tokens_mask().to_vec();
        vec_special_tokens_mask.shrink_to_fit();
        special_tokens_mask = vec_special_tokens_mask.as_mut_ptr();
        std::mem::forget(vec_special_tokens_mask);
    }

    // attention mask
    let mut attention_mask: *mut u32 = null_mut();
    if options.return_attention_mask {
        let mut vec_attention_mask = encoding.get_attention_mask().to_vec();
        vec_attention_mask.shrink_to_fit();
        attention_mask = vec_attention_mask.as_mut_ptr();
        std::mem::forget(vec_attention_mask);
    }

    // offsets
    let mut offsets: *mut Offset = null_mut();
    if options.return_offsets {
        let mut vec_offsets = encoding
            .get_offsets()
            .iter()
            .map(|s| Offset {
                start: s.0,
                end: s.1,
            })
            .collect::<Vec<_>>();
        vec_offsets.shrink_to_fit();
        offsets = vec_offsets.as_mut_ptr();
        std::mem::forget(vec_offsets);
    }

    Buffer {
        ids,
        type_ids,
        special_tokens_mask,
        attention_mask,
        tokens,
        offsets,
        len,
    }
}

/// # Safety
///
/// This function is tokenizer single encode function
#[no_mangle]
pub unsafe extern "C" fn encode(
    ptr: *mut libc::c_void,
    message: *const libc::c_char,
    options: &EncodeOptions,
) -> Buffer {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr
            .cast::<Tokenizer>()
            .as_ref()
            .expect("failed to cast tokenizer");
    }
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

    let encoding = if options.with_offsets_char_mode {
        tokenizer
            .encode_char_offsets(message, options.add_special_tokens)
            .expect("failed to encode input")
    } else {
        tokenizer
            .encode(message, options.add_special_tokens)
            .expect("failed to encode input")
    };

    encode_process(encoding, options)
}

/// # Safety
///
/// This function is tokenizer batch encode function
#[no_mangle]
pub unsafe extern "C" fn encode_batch(
    ptr: *mut libc::c_void,
    messages: *const *const libc::c_char,
    options: &EncodeOptions,
) -> *mut Buffer {
    let tokenizer: &Tokenizer;
    let mut index = 0;
    let mut encode_messages: Vec<String> = Vec::new();

    unsafe {
        tokenizer = ptr
            .cast::<Tokenizer>()
            .as_ref()
            .expect("failed to cast tokenizer");
        // Iterate through the C string pointers until a NULL pointer is encountered
        while !(*messages.offset(index)).is_null() {
            let cstr_ptr = *messages.offset(index);
            let rust_string = CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned();
            encode_messages.push(rust_string);
            index += 1;
        }
    }

    let encoding = if options.with_offsets_char_mode {
        tokenizer
            .encode_batch_char_offsets(encode_messages, options.add_special_tokens)
            .expect("failed to encode input")
    } else {
        tokenizer
            .encode_batch(encode_messages, options.add_special_tokens)
            .expect("failed to encode input")
    };

    // batch process
    let mut vec_encode_results: Vec<Buffer> = encoding
        .iter()
        .cloned()
        .map(|s| encode_process(s, options))
        .collect::<Vec<Buffer>>();
    vec_encode_results.shrink_to_fit();

    let encode_results = vec_encode_results.as_mut_ptr();
    std::mem::forget(vec_encode_results);

    encode_results
}

/// # Safety
///
/// This function is tokenizer decode function
#[no_mangle]
pub unsafe extern "C" fn decode(
    ptr: *mut libc::c_void,
    ids: *const u32,
    len: u32,
    skip_special_tokens: bool,
) -> *mut libc::c_char {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr
            .cast::<Tokenizer>()
            .as_ref()
            .expect("failed to cast tokenizer");
    }
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };
    let string = tokenizer
        .decode(ids_slice, skip_special_tokens)
        .expect("failed to decode input");
    let c_string = std::ffi::CString::new(string).unwrap();
    c_string.into_raw()
}

/// # Safety
///
/// This function is return vocab size to Golang
#[no_mangle]
pub unsafe extern "C" fn vocab_size(ptr: *mut libc::c_void) -> u32 {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr
            .cast::<Tokenizer>()
            .as_ref()
            .expect("failed to cast tokenizer");
    }
    tokenizer.get_vocab_size(true) as u32
}

/// # Safety
///
/// This function is release Tokenizer from Rust return to Golang
#[no_mangle]
pub unsafe extern "C" fn free_tokenizer(ptr: *mut libc::c_void) {
    if ptr.is_null() {
        return;
    }
    ptr.cast::<Tokenizer>();
}

/// # Safety
///
/// This function is release Buffer struct from Rust return to Golang
#[no_mangle]
pub unsafe extern "C" fn free_buffer(buf: Buffer) {
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
    if !buf.tokens.is_null() {
        unsafe {
            let strings = Vec::from_raw_parts(buf.tokens, buf.len, buf.len);
            for s in strings {
                drop(std::ffi::CString::from_raw(s));
            }
        }
    }
    if !buf.offsets.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.offsets, buf.len, buf.len).clear();
        }
    }
}

/// # Safety
///
/// This function is release Vec<Buffer> from Rust return to Golang
#[no_mangle]
pub unsafe extern "C" fn free_batch_buffer(bufs: *mut Buffer) {
    unsafe {
        let box_buffer: Box<Buffer> = Box::from_raw(bufs);
        let vec_buf: Vec<Buffer> = vec![*box_buffer];
        for buf in vec_buf {
            free_buffer(buf);
        }
    }
}

/// # Safety
///
/// This function is release C.char from Rust return to Golang
#[no_mangle]
pub unsafe extern "C" fn free_string(ptr: *mut libc::c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(std::ffi::CString::from_raw(ptr));
    }
}
