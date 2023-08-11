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
    tokens: *mut *mut libc::c_char,
    offsets: *mut Offset,
    len: usize,
}

#[no_mangle]
pub extern "C" fn from_bytes(bytes: *const u8, len: u32) -> *mut Tokenizer {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let tokenizer = Tokenizer::from_bytes(bytes_slice).expect("failed to create tokenizer");
    return Box::into_raw(Box::new(tokenizer));
}

#[no_mangle]
pub extern "C" fn from_bytes_with_truncation(
    bytes: *const u8,
    len: u32,
    max_len: usize,
    dir: u8,
) -> *mut Tokenizer {
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
            null_mut()
        }
    }
}

fn encode_process(encoding: Encoding, return_offsets: bool) -> Buffer {
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

    // offsets
    if return_offsets {
        let mut vec_offsets = encoding.get_offsets()
            .to_vec().into_iter().
            map(|s| Offset { start: s.0, end: s.1 }).
            collect::<Vec<_>>();

        vec_offsets.shrink_to_fit();

        let offsets_ptr = vec_offsets.as_mut_ptr();

        std::mem::forget(vec_offsets);

        return Buffer { ids, tokens, offsets: offsets_ptr, len }
    }

    return Buffer { ids, tokens, offsets: null_mut(), len }
}

#[no_mangle]
pub extern "C" fn encode(
    ptr: *mut libc::c_void,
    message: *const libc::c_char,
    add_special_tokens: bool,
    return_offsets: bool,
    with_char_mode: bool,
) -> Buffer {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

    let encoding: Encoding;
    if with_char_mode {
        encoding = tokenizer.encode_char_offsets(message, add_special_tokens).expect("failed to encode input");
    } else {
        encoding = tokenizer.encode(message, add_special_tokens).expect("failed to encode input");
    }

    return encode_process(encoding, return_offsets);
}

#[no_mangle]
pub extern "C" fn encode_batch(
    ptr: *mut libc::c_void,
    messages: *const *const libc::c_char,
    add_special_tokens: bool,
    return_offsets: bool,
    with_char_mode: bool,
) -> *mut Buffer {
    let tokenizer: &Tokenizer;
    let mut index = 0;
    let mut encode_messages: Vec<String> = Vec::new();

    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
        // Iterate through the C string pointers until a NULL pointer is encountered
        while !(*messages.offset(index)).is_null() {
            let cstr_ptr = *messages.offset(index);
            let rust_string = CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned();
            encode_messages.push(rust_string);
            index += 1;
        }
    }

    let encoding: Vec<Encoding>;
    if with_char_mode {
        encoding = tokenizer.encode_batch_char_offsets(encode_messages, add_special_tokens).expect("failed to encode input");
    } else {
        encoding = tokenizer.encode_batch(encode_messages, add_special_tokens).expect("failed to encode input");
    }

    let mut vec_encode_results: Vec<Buffer> = encoding
        .to_vec().into_iter()
        .map(|s|encode_process(s, return_offsets))
        .collect::<Vec<Buffer>>();
    vec_encode_results.shrink_to_fit();

    let encode_results = vec_encode_results.as_mut_ptr();

    std::mem::forget(vec_encode_results);

    return encode_results;
}

#[no_mangle]
pub extern "C" fn decode(
    ptr: *mut libc::c_void,
    ids: *const u32,
    len: u32,
    skip_special_tokens: bool,
) -> *mut libc::c_char {
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
pub extern "C" fn free_tokenizer(ptr: *mut libc::c_void) {
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
        if buf.offsets != null_mut() {
            Vec::from_raw_parts(buf.offsets, buf.len, buf.len).clear();
        }
    }
}

#[no_mangle]
pub extern "C" fn free_batch_buffer(bufs: *mut Buffer) {
    unsafe {
        let box_buffer: Box<Buffer> = Box::from_raw(bufs);
        let vec_buf: Vec<Buffer> = vec![*box_buffer];
        for buf in vec_buf {
            free_buffer(buf);
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