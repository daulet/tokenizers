use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;
use tokenizers::tokenizer::Tokenizer;

#[repr(C)]
pub struct TokenizerOptions {
    encode_special_tokens: bool,
}

#[repr(C)]
pub struct Buffer {
    ids: *mut u32,
    type_ids: *mut u32,
    special_tokens_mask: *mut u32,
    attention_mask: *mut u32,
    tokens: *mut *mut libc::c_char,
    offsets: *mut (usize, usize),
    len: usize,
}

#[no_mangle]
pub extern "C" fn from_bytes(bytes: *const u8, len: u32, opts: &TokenizerOptions) -> *mut Tokenizer {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let mut tokenizer = Tokenizer::from_bytes(bytes_slice).expect("failed to create tokenizer");
    tokenizer.set_encode_special_tokens(opts.encode_special_tokens);
    Box::into_raw(Box::new(tokenizer))
}

// TODO merge with from_bytes and pass truncation params as an argument to TokenizerOptions
#[no_mangle]
pub extern "C" fn from_bytes_with_truncation(bytes: *const u8, len: u32, max_len: usize, dir: u8) -> *mut Tokenizer {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let tokenizer: Tokenizer = Tokenizer::from_bytes(bytes_slice)
        .expect("failed to create tokenizer")
        .with_truncation(Some(tokenizers::tokenizer::TruncationParams{
            max_length: max_len,
            direction: match dir {
                0 => tokenizers::tokenizer::TruncationDirection::Left,
                1 => tokenizers::tokenizer::TruncationDirection::Right,
                _ => panic!("invalid truncation direction"),
            },
            ..Default::default()
        })).unwrap().to_owned().into();
    Box::into_raw(Box::new(tokenizer))
}

#[no_mangle]
pub extern "C" fn from_file(config: *const libc::c_char) -> *mut libc::c_void {
    let config_cstr = unsafe { CStr::from_ptr(config) };
    let config = config_cstr.to_str().unwrap();
    let config = PathBuf::from(config);
    match Tokenizer::from_file(config) {
        Ok(tokenizer) => {
            let ptr = Box::into_raw(Box::new(tokenizer));
            ptr.cast()
        }
        Err(_) => {
            ptr::null_mut()
        }
    }
}

#[repr(C)]
pub struct EncodeOptions {
    add_special_tokens: bool,

    return_type_ids: bool,
    return_tokens: bool,
    return_special_tokens_mask: bool,
    return_attention_mask: bool,
    return_offsets: bool,
}

#[no_mangle]
pub extern "C" fn encode(ptr: *mut libc::c_void, message: *const libc::c_char, options: &EncodeOptions) -> Buffer {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str();
    if message.is_err() {
        return Buffer { ids: ptr::null_mut(), tokens: ptr::null_mut(), len: 0, type_ids: ptr::null_mut(), special_tokens_mask: ptr::null_mut(), attention_mask: ptr::null_mut() , offsets: ptr::null_mut()};
    }

    let encoding = tokenizer.encode(message.unwrap(), options.add_special_tokens).expect("failed to encode input");
    let mut vec_ids = encoding.get_ids().to_vec();
    vec_ids.shrink_to_fit();
    let ids = vec_ids.as_mut_ptr();
    let len = vec_ids.len();
    std::mem::forget(vec_ids);

    let mut type_ids: *mut u32 = ptr::null_mut();
    if options.return_type_ids {
        let mut vec_type_ids = encoding.get_type_ids().to_vec();
        vec_type_ids.shrink_to_fit();
        type_ids = vec_type_ids.as_mut_ptr();
        std::mem::forget(vec_type_ids);
    }

    let mut tokens: *mut *mut libc::c_char = ptr::null_mut();
    if options.return_tokens {
        let mut vec_tokens = encoding.get_tokens()
            .to_vec().into_iter()
            .map(|s| std::ffi::CString::new(s).unwrap().into_raw())
            .collect::<Vec<_>>();
        vec_tokens.shrink_to_fit();
        tokens = vec_tokens.as_mut_ptr();
        std::mem::forget(vec_tokens);
    }

    let mut special_tokens_mask: *mut u32 = ptr::null_mut();
    if options.return_special_tokens_mask {
        let mut vec_special_tokens_mask = encoding.get_special_tokens_mask().to_vec();
        vec_special_tokens_mask.shrink_to_fit();
        special_tokens_mask = vec_special_tokens_mask.as_mut_ptr();
        std::mem::forget(vec_special_tokens_mask);
    }

    let mut attention_mask: *mut u32 = ptr::null_mut();
    if options.return_attention_mask {
        let mut vec_attention_mask = encoding.get_attention_mask().to_vec();
        vec_attention_mask.shrink_to_fit();
        attention_mask = vec_attention_mask.as_mut_ptr();
        std::mem::forget(vec_attention_mask);
    }

    let mut offsets: *mut (usize, usize) = ptr::null_mut();
    if options.return_offsets {
        let mut vec_offsets = encoding.get_offsets().to_vec();
        vec_offsets.shrink_to_fit();
        offsets = vec_offsets.as_mut_ptr();
        std::mem::forget(vec_offsets);
    }

    Buffer { ids, type_ids, special_tokens_mask, attention_mask, tokens, offsets, len }
}

#[no_mangle]
pub extern "C" fn decode(ptr: *mut libc::c_void, ids: *const u32, len: u32, skip_special_tokens: bool) -> *mut libc::c_char {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };

    let string = tokenizer.decode(ids_slice, skip_special_tokens).expect("failed to decode input");
    match std::ffi::CString::new(string) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn vocab_size(ptr: *mut libc::c_void) -> u32 {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    tokenizer.get_vocab_size(true) as u32
}

#[no_mangle]
pub extern "C" fn free_tokenizer(ptr: *mut ::libc::c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr.cast::<Tokenizer>()));
    }
}

#[no_mangle]
pub extern "C" fn free_buffer(buf: Buffer) {
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
                drop(std::ffi::CString::from_raw(s.cast::<libc::c_char>()));
            }   
        }
    }
}

#[no_mangle]
pub extern "C" fn free_string(ptr: *mut libc::c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(std::ffi::CString::from_raw(ptr));
    }
}
