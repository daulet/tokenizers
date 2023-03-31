use std::ffi::CStr;
use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;

#[no_mangle]
pub extern "C" fn encode(message: *const libc::c_char, len: *mut u32) -> *mut u32 {
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

    // TODO read once
    let config = PathBuf::from("./lib/tokenizer/data/bert-base-uncased.json");
    let tokenizer = Tokenizer::from_file(config).expect("failed to load tokenizer");
    let encoding = tokenizer.encode(message, false).expect("failed to encode input");
    let mut vec = encoding.get_ids().to_vec();
    unsafe {
        *len = vec.len() as u32;
    }
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[no_mangle]
pub extern "C" fn decode(ids: *const u32, len: u32) -> *mut libc::c_char {
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };
    let config = PathBuf::from("./lib/tokenizer/data/bert-base-uncased.json");
    let tokenizer = Tokenizer::from_file(config).expect("failed to load tokenizer");
    // TODO parameterize special tokens
    let string = tokenizer.decode(ids_slice.to_vec(), true).expect("failed to decode input");
    let c_string = std::ffi::CString::new(string).unwrap();
    c_string.into_raw()
}
