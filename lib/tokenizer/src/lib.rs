use std::ffi::CStr;
use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;

#[no_mangle]
pub extern "C" fn encode(message: *const libc::c_char, len: *mut u32) -> *mut u32 {
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

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
