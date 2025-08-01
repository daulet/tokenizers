use crate::chat_template::ChatTemplate;
use crate::chat_template::{Message, TokenizerConfigToken, Tool};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn tokenizers_new_chat_template(
    template: *const c_char,
    bos_token: *const c_char,
    eos_token: *const c_char,
) -> *mut ChatTemplate {
    let template = unsafe { CStr::from_ptr(template).to_string_lossy().into_owned() };
    let bos_token = if bos_token.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(bos_token).to_string_lossy().into_owned() };
        Some(TokenizerConfigToken::String(s))
    };
    let eos_token = if eos_token.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(eos_token).to_string_lossy().into_owned() };
        Some(TokenizerConfigToken::String(s))
    };

    let chat_template = ChatTemplate::new(template, bos_token, eos_token);
    Box::into_raw(Box::new(chat_template))
}

#[no_mangle]
pub extern "C" fn tokenizers_apply_chat_template(
    ptr: *mut ChatTemplate,
    messages_json: *const c_char,
    tools_json: *const c_char,
    tool_prompt: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    let chat_template = unsafe { &*ptr };
    let messages_json = unsafe { CStr::from_ptr(messages_json).to_string_lossy() };
    let messages: Vec<Message> = serde_json::from_str(&messages_json).unwrap();

    let tools_and_prompt = if !tools_json.is_null() && !tool_prompt.is_null() {
        let tools_json = unsafe { CStr::from_ptr(tools_json).to_string_lossy() };
        if tools_json.is_empty() {
            None
        } else {
            // Some((tools_json, unsafe { CStr::from_ptr(tool_prompt).to_string_lossy() }))

            let tools: Vec<Tool> = serde_json::from_str(&tools_json).unwrap();
            let tool_prompt = unsafe { CStr::from_ptr(tool_prompt).to_string_lossy().into_owned() };
            Some((tools, tool_prompt))
        }
    } else {
        None
    };

    match chat_template.apply(messages, tools_and_prompt) {
        Ok(result) => {
            let c_str = CString::new(result).unwrap();
            c_str.into_raw()
            // CString::new(result).unwrap(),
        }
        Err(e) => {
            let error_message = CString::new(e.to_string()).unwrap();
            if !error.is_null() {
                unsafe {
                    *error = error_message.into_raw();
                }
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizers_free_chat_template(ptr: *mut ChatTemplate) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}
