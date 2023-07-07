use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;
use tokenizers::tokenizer::Tokenizer;

#[repr(C)]
pub struct Buffer {
    ids: *mut u32,
    tokens: *mut *mut libc::c_char,
    len: usize,
}

#[no_mangle]
pub extern "C" fn from_bytes(bytes: *const u8, len: u32) -> *mut Tokenizer {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let tokenizer = Tokenizer::from_bytes(bytes_slice).expect("failed to create tokenizer");
    return Box::into_raw(Box::new(tokenizer));
}

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
        })).to_owned().into();
    return Box::into_raw(Box::new(tokenizer));
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

#[no_mangle]
pub extern "C" fn encode(ptr: *mut libc::c_void, message: *const libc::c_char, add_special_tokens: bool) -> Buffer {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

    let encoding = tokenizer.encode(message, add_special_tokens).expect("failed to encode input");
    let mut vec_ids = encoding.get_ids().to_vec();
    let mut vec_tokens = encoding.get_tokens()
        .to_vec().into_iter()
        .map(|s| std::ffi::CString::new(s).unwrap().into_raw())
        .collect::<Vec<_>>();

    vec_ids.shrink_to_fit();
    vec_tokens.shrink_to_fit();
    
    let ids = vec_ids.as_mut_ptr();
    let tokens = vec_tokens.as_mut_ptr();
    let len = vec_ids.len();

    std::mem::forget(vec_ids);
    std::mem::forget(vec_tokens);

    Buffer { ids, tokens, len }
}

#[no_mangle]
pub extern "C" fn decode(ptr: *mut libc::c_void, ids: *const u32, len: u32, skip_special_tokens: bool) -> *mut libc::c_char {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };

    let string = tokenizer.decode(ids_slice.to_vec(), skip_special_tokens).expect("failed to decode input");
    let c_string = std::ffi::CString::new(string).unwrap();
    c_string.into_raw()
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
    ptr.cast::<Tokenizer>();
}

#[no_mangle]
pub extern "C" fn free_buffer(buf: Buffer) {
    if buf.ids.is_null() {
        return;
    }
    unsafe {
        Vec::from_raw_parts(buf.ids, buf.len, buf.len);
        let strings = Vec::from_raw_parts(buf.tokens, buf.len, buf.len);
        for s in strings {
            drop(std::ffi::CString::from_raw(s));
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