use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;
use tokenizers::tokenizer::Tokenizer;

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
pub extern "C" fn free_tokenizer(ptr: *mut ::libc::c_void) {
    unsafe {
        if ptr.is_null() {
            return;
        }
        Box::from_raw(ptr.cast::<Tokenizer>());
    }
}

#[no_mangle]
pub extern "C" fn encode(ptr: *mut libc::c_void, message: *const libc::c_char, len: *mut u32) -> *mut u32 {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

    let encoding = tokenizer.encode(message, false).expect("failed to encode input");
    let mut vec = encoding.get_ids().to_vec();
    vec.shrink_to_fit();
    unsafe {
        *len = vec.len() as u32;
    }
    let vec_ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    vec_ptr
}

#[no_mangle]
pub extern "C" fn decode(ptr: *mut libc::c_void, ids: *const u32, len: u32) -> *mut libc::c_char {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr.cast::<Tokenizer>().as_ref().expect("failed to cast tokenizer");
    }
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };

    // TODO parameterize special tokens
    let string = tokenizer.decode(ids_slice.to_vec(), true).expect("failed to decode input");
    let c_string = std::ffi::CString::new(string).unwrap();
    c_string.into_raw()
}
