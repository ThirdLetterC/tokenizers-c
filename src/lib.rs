//! C FFI bindings (`libtokenizers.a` / `tokenizers.lib` + `tokenizers.h`) for `huggingface/tokenizers`.
//!
//! Exposes an embeddable, panic-safe C99/C11 API around the official Rust
//! `tokenizers::Tokenizer` and `tokenizers::Encoding` types.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;
use std::slice;

use tokenizers::tokenizer::{
    AddedToken, EncodeInput, Encoding, PaddingDirection, PaddingParams, PaddingStrategy,
    Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

#[cfg(feature = "http")]
use tokenizers::FromPretrainedParameters;

/* ========================================================================== *
 * C ABI Types & Enums (matching include/tokenizers.h)
 * ========================================================================== */

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TkStatus {
    TkOk = 0,
    TkErrNullPtr = 1,
    TkErrInvalidUtf8 = 2,
    TkErrIo = 3,
    TkErrJson = 4,
    TkErrModel = 5,
    TkErrTruncation = 6,
    TkErrPadding = 7,
    TkErrOutOfRange = 8,
    TkErrUnsupported = 9,
    TkErrHttp = 10,
    TkErrPanic = 11,
    TkErrInternal = 255,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TkDirection {
    TkDirectionRight = 0,
    TkDirectionLeft = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TkPaddingStrategyTag {
    TkPaddingBatchLongest = 0,
    TkPaddingFixed = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TkTruncationStrategy {
    TkTruncationLongestFirst = 0,
    TkTruncationOnlyFirst = 1,
    TkTruncationOnlySecond = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TkOffsetType {
    TkOffsetByte = 0,
    TkOffsetChar = 1,
    TkOffsetNone = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TkPaddingParams {
    pub strategy: TkPaddingStrategyTag,
    pub fixed_length: usize,
    pub direction: TkDirection,
    pub pad_to_multiple_of: usize,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TkTruncationParams {
    pub max_length: usize,
    pub strategy: TkTruncationStrategy,
    pub stride: usize,
    pub direction: TkDirection,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TkAddedToken {
    pub content: *const c_char,
    pub special: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TkPretrainedParams {
    pub revision: *const c_char,
    pub auth_token: *const c_char,
    pub user_agent: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct TkOffset {
    pub start: usize,
    pub end: usize,
}

/// Opaque wrapper around `tokenizers::Tokenizer` + cached C-string for `pad_token`.
pub struct TkTokenizer {
    inner: Tokenizer,
    cached_pad_token: CString,
}

/// Opaque wrapper around `tokenizers::Encoding` + cached C-strings for zero-copy `tk_encoding_get_token`.
pub struct TkEncoding {
    inner: Encoding,
    c_tokens: Vec<CString>,
}

impl TkEncoding {
    fn from_encoding(inner: Encoding) -> Self {
        let c_tokens = Self::build_c_tokens(&inner);
        Self { inner, c_tokens }
    }

    fn build_c_tokens(inner: &Encoding) -> Vec<CString> {
        inner
            .get_tokens()
            .iter()
            .map(|s| CString::new(s.as_bytes()).unwrap_or_else(|_| CString::default()))
            .collect()
    }

    fn refresh_c_tokens(&mut self) {
        self.c_tokens = Self::build_c_tokens(&self.inner);
    }
}

/// Opaque stateful streaming decoder.
pub struct TkDecodeStream {
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

/* ========================================================================== *
 * Thread-Local Error State & Panic Boundary Helpers
 * ========================================================================== */

struct ErrorState {
    code: TkStatus,
    message: CString,
}

impl Default for ErrorState {
    fn default() -> Self {
        Self {
            code: TkStatus::TkOk,
            message: CString::default(),
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<ErrorState> = RefCell::new(ErrorState::default());
}

static DEFAULT_PAD_TOKEN: &[u8] = b"[PAD]\0";
static DEFAULT_REVISION: &[u8] = b"main\0";
static VERSION_STR: &[u8] = b"0.23.2\0";
static BACKEND_STR: &[u8] = b"rust-huggingface-tokenizers\0";

fn set_error(code: TkStatus, msg: impl Into<String>) -> TkStatus {
    let s = msg.into().replace('\0', " ");
    let cstr = CString::new(s).unwrap_or_else(|_| CString::default());
    LAST_ERROR.with(|cell| {
        let mut err = cell.borrow_mut();
        err.code = code;
        err.message = cstr;
    });
    code
}

fn clear_error_internal() {
    LAST_ERROR.with(|cell| {
        let mut err = cell.borrow_mut();
        err.code = TkStatus::TkOk;
        err.message = CString::default();
    });
}

fn with_ffi_guard<T, F>(on_err: T, f: F) -> T
where
    F: FnOnce() -> T,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            set_error(
                TkStatus::TkErrPanic,
                "Rust panic caught at FFI boundary in tokenizers-c",
            );
            on_err
        }
    }
}

unsafe fn cstr_to_str<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, TkStatus> {
    if ptr.is_null() {
        return Err(set_error(
            TkStatus::TkErrNullPtr,
            format!("Null pointer passed for `{}`", name),
        ));
    }
    CStr::from_ptr(ptr).to_str().map_err(|e| {
        set_error(
            TkStatus::TkErrInvalidUtf8,
            format!("Invalid UTF-8 in `{}`: {}", name, e),
        )
    })
}

fn string_to_c_heap(s: String) -> *mut c_char {
    let sanitized = if s.as_bytes().contains(&0) {
        s.replace('\0', "")
    } else {
        s
    };
    match CString::new(sanitized) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

impl TkTokenizer {
    fn new(inner: Tokenizer) -> Self {
        let pad_tok = inner
            .get_padding()
            .map(|p| p.pad_token.as_str())
            .unwrap_or("[PAD]");
        let cached_pad_token = CString::new(pad_tok).unwrap_or_default();
        Self {
            inner,
            cached_pad_token,
        }
    }
}

/* ========================================================================== *
 * Version & Error API
 * ========================================================================== */

#[no_mangle]
pub extern "C" fn tk_version() -> *const c_char {
    VERSION_STR.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn tk_backend_name() -> *const c_char {
    BACKEND_STR.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn tk_last_error_code() -> TkStatus {
    LAST_ERROR.with(|cell| cell.borrow().code)
}

#[no_mangle]
pub extern "C" fn tk_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().message.as_ptr())
}

#[no_mangle]
pub extern "C" fn tk_clear_error() {
    clear_error_internal();
}

/* ========================================================================== *
 * Default Parameter Constructors
 * ========================================================================== */

#[no_mangle]
pub extern "C" fn tk_padding_params_default() -> TkPaddingParams {
    TkPaddingParams {
        strategy: TkPaddingStrategyTag::TkPaddingBatchLongest,
        fixed_length: 0,
        direction: TkDirection::TkDirectionRight,
        pad_to_multiple_of: 0,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: DEFAULT_PAD_TOKEN.as_ptr() as *const c_char,
    }
}

#[no_mangle]
pub extern "C" fn tk_truncation_params_default(max_length: usize) -> TkTruncationParams {
    TkTruncationParams {
        max_length,
        strategy: TkTruncationStrategy::TkTruncationLongestFirst,
        stride: 0,
        direction: TkDirection::TkDirectionRight,
    }
}

#[no_mangle]
pub extern "C" fn tk_added_token_default(content: *const c_char, special: bool) -> TkAddedToken {
    TkAddedToken {
        content,
        special,
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: !special,
    }
}

#[no_mangle]
pub extern "C" fn tk_pretrained_params_default() -> TkPretrainedParams {
    TkPretrainedParams {
        revision: DEFAULT_REVISION.as_ptr() as *const c_char,
        auth_token: ptr::null(),
        user_agent: ptr::null(),
    }
}

/* ========================================================================== *
 * Tokenizer Lifecycle & Serialization
 * ========================================================================== */

#[no_mangle]
pub unsafe extern "C" fn tk_from_file(path: *const c_char) -> *mut TkTokenizer {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        let path_str = match cstr_to_str(path, "path") {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        match Tokenizer::from_file(Path::new(path_str)) {
            Ok(tk) => Box::into_raw(Box::new(TkTokenizer::new(tk))),
            Err(e) => {
                set_error(
                    TkStatus::TkErrIo,
                    format!("Failed to load tokenizer from '{}': {}", path_str, e),
                );
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_from_bytes(data: *const u8, len: usize) -> *mut TkTokenizer {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if data.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `data` pointer in tk_from_bytes");
            return ptr::null_mut();
        }
        let bytes = slice::from_raw_parts(data, len);
        match Tokenizer::from_bytes(bytes) {
            Ok(tk) => Box::into_raw(Box::new(TkTokenizer::new(tk))),
            Err(e) => {
                set_error(
                    TkStatus::TkErrJson,
                    format!("Failed to deserialize tokenizer bytes: {}", e),
                );
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_from_json(json: *const c_char) -> *mut TkTokenizer {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        let json_str = match cstr_to_str(json, "json") {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        match Tokenizer::from_bytes(json_str.as_bytes()) {
            Ok(tk) => Box::into_raw(Box::new(TkTokenizer::new(tk))),
            Err(e) => {
                set_error(
                    TkStatus::TkErrJson,
                    format!("Failed to parse tokenizer JSON: {}", e),
                );
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_from_pretrained(identifier: *const c_char) -> *mut TkTokenizer {
    tk_from_pretrained_ex(identifier, ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn tk_from_pretrained_ex(
    identifier: *const c_char,
    params: *const TkPretrainedParams,
) -> *mut TkTokenizer {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        let id_str = match cstr_to_str(identifier, "identifier") {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        #[cfg(feature = "http")]
        {
            let hf_params = if params.is_null() {
                None
            } else {
                let p = &*params;
                let mut fp = FromPretrainedParameters::default();
                if !p.revision.is_null() {
                    if let Ok(rev) = cstr_to_str(p.revision, "params.revision") {
                        if !rev.is_empty() {
                            fp.revision = rev.to_string();
                        }
                    } else {
                        return ptr::null_mut();
                    }
                }
                if !p.auth_token.is_null() {
                    if let Ok(tok) = cstr_to_str(p.auth_token, "params.auth_token") {
                        if !tok.is_empty() {
                            fp.token = Some(tok.to_string());
                        }
                    } else {
                        return ptr::null_mut();
                    }
                }
                if !p.user_agent.is_null() {
                    if let Ok(ua) = cstr_to_str(p.user_agent, "params.user_agent") {
                        if !ua.is_empty() {
                            fp.user_agent.insert("user-agent".to_string(), ua.to_string());
                        }
                    } else {
                        return ptr::null_mut();
                    }
                }
                Some(fp)
            };

            match Tokenizer::from_pretrained(id_str, hf_params) {
                Ok(tk) => Box::into_raw(Box::new(TkTokenizer::new(tk))),
                Err(e) => {
                    set_error(
                        TkStatus::TkErrHttp,
                        format!("from_pretrained('{}') failed: {}", id_str, e),
                    );
                    ptr::null_mut()
                }
            }
        }

        #[cfg(not(feature = "http"))]
        {
            let _ = (id_str, params);
            set_error(
                TkStatus::TkErrUnsupported,
                "tokenizers-c was built without the `http` Cargo feature",
            );
            ptr::null_mut()
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_clone(tokenizer: *const TkTokenizer) -> *mut TkTokenizer {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_clone");
            return ptr::null_mut();
        }
        let src = &*tokenizer;
        Box::into_raw(Box::new(TkTokenizer {
            inner: src.inner.clone(),
            cached_pad_token: src.cached_pad_token.clone(),
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_save(
    tokenizer: *const TkTokenizer,
    path: *const c_char,
    pretty: bool,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if tokenizer.is_null() {
            return set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_save");
        }
        let path_str = match cstr_to_str(path, "path") {
            Ok(s) => s,
            Err(code) => return code,
        };
        match (*tokenizer).inner.save(path_str, pretty) {
            Ok(()) => TkStatus::TkOk,
            Err(e) => set_error(
                TkStatus::TkErrIo,
                format!("Failed to save tokenizer to '{}': {}", path_str, e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_to_json(
    tokenizer: *const TkTokenizer,
    pretty: bool,
) -> *mut c_char {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_to_json");
            return ptr::null_mut();
        }
        match (*tokenizer).inner.to_string(pretty) {
            Ok(json) => string_to_c_heap(json),
            Err(e) => {
                set_error(
                    TkStatus::TkErrJson,
                    format!("Failed to serialize tokenizer to JSON: {}", e),
                );
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_free(tokenizer: *mut TkTokenizer) {
    if !tokenizer.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(tokenizer));
        }));
    }
}

/* ========================================================================== *
 * Configuration: Padding, Truncation & Added Tokens
 * ========================================================================== */

#[no_mangle]
pub unsafe extern "C" fn tk_enable_padding(
    tokenizer: *mut TkTokenizer,
    params: *const TkPaddingParams,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if tokenizer.is_null() || params.is_null() {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null `tokenizer` or `params` in tk_enable_padding",
            );
        }
        let tk = &mut *tokenizer;
        let p = &*params;
        let pad_token = if p.pad_token.is_null() {
            "[PAD]".to_string()
        } else {
            match cstr_to_str(p.pad_token, "params.pad_token") {
                Ok(s) => s.to_string(),
                Err(code) => return code,
            }
        };

        let strategy = match p.strategy {
            TkPaddingStrategyTag::TkPaddingBatchLongest => PaddingStrategy::BatchLongest,
            TkPaddingStrategyTag::TkPaddingFixed => PaddingStrategy::Fixed(p.fixed_length),
        };
        let direction = match p.direction {
            TkDirection::TkDirectionRight => PaddingDirection::Right,
            TkDirection::TkDirectionLeft => PaddingDirection::Left,
        };
        let pad_to_multiple_of = if p.pad_to_multiple_of > 0 {
            Some(p.pad_to_multiple_of)
        } else {
            None
        };

        tk.cached_pad_token = CString::new(pad_token.as_str()).unwrap_or_default();
        tk.inner.with_padding(Some(PaddingParams {
            strategy,
            direction,
            pad_to_multiple_of,
            pad_id: p.pad_id,
            pad_type_id: p.pad_type_id,
            pad_token,
        }));
        TkStatus::TkOk
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_disable_padding(tokenizer: *mut TkTokenizer) {
    let _ = with_ffi_guard((), || {
        if !tokenizer.is_null() {
            (*tokenizer).inner.with_padding(None);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn tk_get_padding(
    tokenizer: *const TkTokenizer,
    out_params: *mut TkPaddingParams,
) -> bool {
    with_ffi_guard(false, || {
        if tokenizer.is_null() {
            return false;
        }
        let tk = &*tokenizer;
        match tk.inner.get_padding() {
            Some(p) => {
                if !out_params.is_null() {
                    let (strategy, fixed_length) = match p.strategy {
                        PaddingStrategy::BatchLongest => {
                            (TkPaddingStrategyTag::TkPaddingBatchLongest, 0)
                        }
                        PaddingStrategy::Fixed(n) => (TkPaddingStrategyTag::TkPaddingFixed, n),
                    };
                    let direction = match p.direction {
                        PaddingDirection::Right => TkDirection::TkDirectionRight,
                        PaddingDirection::Left => TkDirection::TkDirectionLeft,
                    };
                    *out_params = TkPaddingParams {
                        strategy,
                        fixed_length,
                        direction,
                        pad_to_multiple_of: p.pad_to_multiple_of.unwrap_or(0),
                        pad_id: p.pad_id,
                        pad_type_id: p.pad_type_id,
                        pad_token: tk.cached_pad_token.as_ptr(),
                    };
                }
                true
            }
            None => false,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_enable_truncation(
    tokenizer: *mut TkTokenizer,
    params: *const TkTruncationParams,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if tokenizer.is_null() || params.is_null() {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null `tokenizer` or `params` in tk_enable_truncation",
            );
        }
        let tk = &mut *tokenizer;
        let p = &*params;
        if p.max_length == 0 || p.stride >= p.max_length {
            return set_error(
                TkStatus::TkErrTruncation,
                format!(
                    "Invalid truncation parameters: stride ({}) must be < max_length ({}) and max_length > 0",
                    p.stride, p.max_length
                ),
            );
        }

        let strategy = match p.strategy {
            TkTruncationStrategy::TkTruncationLongestFirst => TruncationStrategy::LongestFirst,
            TkTruncationStrategy::TkTruncationOnlyFirst => TruncationStrategy::OnlyFirst,
            TkTruncationStrategy::TkTruncationOnlySecond => TruncationStrategy::OnlySecond,
        };
        let direction = match p.direction {
            TkDirection::TkDirectionRight => TruncationDirection::Right,
            TkDirection::TkDirectionLeft => TruncationDirection::Left,
        };

        match tk.inner.with_truncation(Some(TruncationParams {
            max_length: p.max_length,
            strategy,
            stride: p.stride,
            direction,
        })) {
            Ok(_) => TkStatus::TkOk,
            Err(e) => set_error(
                TkStatus::TkErrTruncation,
                format!("Failed to enable truncation: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_disable_truncation(tokenizer: *mut TkTokenizer) {
    let _ = with_ffi_guard((), || {
        if !tokenizer.is_null() {
            let _ = (*tokenizer).inner.with_truncation(None);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn tk_get_truncation(
    tokenizer: *const TkTokenizer,
    out_params: *mut TkTruncationParams,
) -> bool {
    with_ffi_guard(false, || {
        if tokenizer.is_null() {
            return false;
        }
        match (*tokenizer).inner.get_truncation() {
            Some(t) => {
                if !out_params.is_null() {
                    let strategy = match t.strategy {
                        TruncationStrategy::LongestFirst => {
                            TkTruncationStrategy::TkTruncationLongestFirst
                        }
                        TruncationStrategy::OnlyFirst => {
                            TkTruncationStrategy::TkTruncationOnlyFirst
                        }
                        TruncationStrategy::OnlySecond => {
                            TkTruncationStrategy::TkTruncationOnlySecond
                        }
                    };
                    let direction = match t.direction {
                        TruncationDirection::Right => TkDirection::TkDirectionRight,
                        TruncationDirection::Left => TkDirection::TkDirectionLeft,
                    };
                    *out_params = TkTruncationParams {
                        max_length: t.max_length,
                        strategy,
                        stride: t.stride,
                        direction,
                    };
                }
                true
            }
            None => false,
        }
    })
}

unsafe fn build_added_tokens(
    tokens: *const TkAddedToken,
    num_tokens: usize,
    force_special: Option<bool>,
) -> Result<Vec<AddedToken>, TkStatus> {
    if num_tokens == 0 {
        return Ok(Vec::new());
    }
    if tokens.is_null() {
        return Err(set_error(
            TkStatus::TkErrNullPtr,
            "Null `tokens` array passed to tk_add_tokens",
        ));
    }
    let slice = slice::from_raw_parts(tokens, num_tokens);
    let mut vec = Vec::with_capacity(num_tokens);
    for (idx, item) in slice.iter().enumerate() {
        let content = cstr_to_str(item.content, &format!("tokens[{}].content", idx))?;
        let is_special = force_special.unwrap_or(item.special);
        let tok = AddedToken::from(content.to_string(), is_special)
            .single_word(item.single_word)
            .lstrip(item.lstrip)
            .rstrip(item.rstrip)
            .normalized(item.normalized);
        vec.push(tok);
    }
    Ok(vec)
}

#[no_mangle]
pub unsafe extern "C" fn tk_add_tokens(
    tokenizer: *mut TkTokenizer,
    tokens: *const TkAddedToken,
    num_tokens: usize,
    out_added: *mut usize,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if tokenizer.is_null() {
            return set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_add_tokens");
        }
        let added_vec = match build_added_tokens(tokens, num_tokens, None) {
            Ok(v) => v,
            Err(code) => return code,
        };
        match (*tokenizer).inner.add_tokens(added_vec) {
            Ok(n) => {
                if !out_added.is_null() {
                    *out_added = n;
                }
                TkStatus::TkOk
            }
            Err(e) => set_error(
                TkStatus::TkErrModel,
                format!("tk_add_tokens failed: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_add_special_tokens(
    tokenizer: *mut TkTokenizer,
    tokens: *const TkAddedToken,
    num_tokens: usize,
    out_added: *mut usize,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if tokenizer.is_null() {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null `tokenizer` in tk_add_special_tokens",
            );
        }
        let added_vec = match build_added_tokens(tokens, num_tokens, Some(true)) {
            Ok(v) => v,
            Err(code) => return code,
        };
        match (*tokenizer).inner.add_special_tokens(added_vec) {
            Ok(n) => {
                if !out_added.is_null() {
                    *out_added = n;
                }
                TkStatus::TkOk
            }
            Err(e) => set_error(
                TkStatus::TkErrModel,
                format!("tk_add_special_tokens failed: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_set_encode_special_tokens(tokenizer: *mut TkTokenizer, value: bool) {
    let _ = with_ffi_guard((), || {
        if !tokenizer.is_null() {
            (*tokenizer).inner.set_encode_special_tokens(value);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn tk_get_encode_special_tokens(tokenizer: *const TkTokenizer) -> bool {
    with_ffi_guard(false, || {
        if tokenizer.is_null() {
            return false;
        }
        (*tokenizer).inner.get_encode_special_tokens()
    })
}

/* ========================================================================== *
 * Vocabulary Inspection
 * ========================================================================== */

#[no_mangle]
pub unsafe extern "C" fn tk_vocab_size(
    tokenizer: *const TkTokenizer,
    with_added_tokens: bool,
) -> usize {
    with_ffi_guard(0, || {
        if tokenizer.is_null() {
            return 0;
        }
        (*tokenizer).inner.get_vocab_size(with_added_tokens)
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_token_to_id(
    tokenizer: *const TkTokenizer,
    token: *const c_char,
    out_id: *mut u32,
) -> bool {
    with_ffi_guard(false, || {
        if tokenizer.is_null() || token.is_null() {
            return false;
        }
        let tok_str = match CStr::from_ptr(token).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };
        match (*tokenizer).inner.token_to_id(tok_str) {
            Some(id) => {
                if !out_id.is_null() {
                    *out_id = id;
                }
                true
            }
            None => false,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_id_to_token(
    tokenizer: *const TkTokenizer,
    id: u32,
) -> *mut c_char {
    with_ffi_guard(ptr::null_mut(), || {
        if tokenizer.is_null() {
            return ptr::null_mut();
        }
        match (*tokenizer).inner.id_to_token(id) {
            Some(tok) => string_to_c_heap(tok),
            None => ptr::null_mut(),
        }
    })
}

/* ========================================================================== *
 * Encoding (Single, Pair, Pre-tokenized, and Parallel Batch)
 * ========================================================================== */

fn run_encode<'s, E: Into<EncodeInput<'s>>>(
    tk: &Tokenizer,
    input: E,
    add_special_tokens: bool,
    offset_type: TkOffsetType,
) -> *mut TkEncoding {
    let res = match offset_type {
        TkOffsetType::TkOffsetByte => tk.encode(input, add_special_tokens),
        TkOffsetType::TkOffsetChar => tk.encode_char_offsets(input, add_special_tokens),
        TkOffsetType::TkOffsetNone => tk.encode_fast(input, add_special_tokens),
    };
    match res {
        Ok(enc) => Box::into_raw(Box::new(TkEncoding::from_encoding(enc))),
        Err(e) => {
            set_error(TkStatus::TkErrModel, format!("Encode failed: {}", e));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode(
    tokenizer: *const TkTokenizer,
    text: *const c_char,
    add_special_tokens: bool,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_encode");
            return ptr::null_mut();
        }
        let s = match cstr_to_str(text, "text") {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };
        run_encode(&(*tokenizer).inner, s, add_special_tokens, TkOffsetType::TkOffsetByte)
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_ex(
    tokenizer: *const TkTokenizer,
    text: *const c_char,
    text_len: usize,
    add_special_tokens: bool,
    offset_type: TkOffsetType,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() || text.is_null() {
            set_error(
                TkStatus::TkErrNullPtr,
                "Null `tokenizer` or `text` in tk_encode_ex",
            );
            return ptr::null_mut();
        }
        let bytes = slice::from_raw_parts(text as *const u8, text_len);
        let s = match std::str::from_utf8(bytes) {
            Ok(v) => v,
            Err(e) => {
                set_error(
                    TkStatus::TkErrInvalidUtf8,
                    format!("Invalid UTF-8 in tk_encode_ex: {}", e),
                );
                return ptr::null_mut();
            }
        };
        run_encode(&(*tokenizer).inner, s, add_special_tokens, offset_type)
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_pair(
    tokenizer: *const TkTokenizer,
    text_a: *const c_char,
    text_b: *const c_char,
    add_special_tokens: bool,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_encode_pair");
            return ptr::null_mut();
        }
        let sa = match cstr_to_str(text_a, "text_a") {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };
        let sb = match cstr_to_str(text_b, "text_b") {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };
        run_encode(
            &(*tokenizer).inner,
            (sa, sb),
            add_special_tokens,
            TkOffsetType::TkOffsetByte,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_pair_ex(
    tokenizer: *const TkTokenizer,
    text_a: *const c_char,
    text_a_len: usize,
    text_b: *const c_char,
    text_b_len: usize,
    add_special_tokens: bool,
    offset_type: TkOffsetType,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() || text_a.is_null() || text_b.is_null() {
            set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_encode_pair_ex",
            );
            return ptr::null_mut();
        }
        let sa = match std::str::from_utf8(slice::from_raw_parts(text_a as *const u8, text_a_len))
        {
            Ok(v) => v,
            Err(e) => {
                set_error(
                    TkStatus::TkErrInvalidUtf8,
                    format!("Invalid UTF-8 in `text_a`: {}", e),
                );
                return ptr::null_mut();
            }
        };
        let sb = match std::str::from_utf8(slice::from_raw_parts(text_b as *const u8, text_b_len))
        {
            Ok(v) => v,
            Err(e) => {
                set_error(
                    TkStatus::TkErrInvalidUtf8,
                    format!("Invalid UTF-8 in `text_b`: {}", e),
                );
                return ptr::null_mut();
            }
        };
        run_encode(&(*tokenizer).inner, (sa, sb), add_special_tokens, offset_type)
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_pretokenized(
    tokenizer: *const TkTokenizer,
    words: *const *const c_char,
    word_count: usize,
    add_special_tokens: bool,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() || (word_count > 0 && words.is_null()) {
            set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_encode_pretokenized",
            );
            return ptr::null_mut();
        }
        let mut slice_words: Vec<&str> = Vec::with_capacity(word_count);
        if word_count > 0 {
            let ptrs = slice::from_raw_parts(words, word_count);
            for (i, &p) in ptrs.iter().enumerate() {
                match cstr_to_str(p, &format!("words[{}]", i)) {
                    Ok(w) => slice_words.push(w),
                    Err(_) => return ptr::null_mut(),
                }
            }
        }
        run_encode(
            &(*tokenizer).inner,
            &slice_words[..],
            add_special_tokens,
            TkOffsetType::TkOffsetByte,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_batch(
    tokenizer: *const TkTokenizer,
    texts: *const *const c_char,
    count: usize,
    add_special_tokens: bool,
    out_encodings: *mut *mut TkEncoding,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if count == 0 {
            return TkStatus::TkOk;
        }
        if tokenizer.is_null() || texts.is_null() || out_encodings.is_null() {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_encode_batch",
            );
        }
        let ptr_slice = slice::from_raw_parts(texts, count);
        let mut inputs = Vec::with_capacity(count);
        for (i, &p) in ptr_slice.iter().enumerate() {
            match cstr_to_str(p, &format!("texts[{}]", i)) {
                Ok(s) => inputs.push(s),
                Err(code) => return code,
            }
        }
        match (*tokenizer).inner.encode_batch(inputs, add_special_tokens) {
            Ok(encodings) => {
                let out_slice = slice::from_raw_parts_mut(out_encodings, count);
                for (i, enc) in encodings.into_iter().enumerate() {
                    out_slice[i] = Box::into_raw(Box::new(TkEncoding::from_encoding(enc)));
                }
                TkStatus::TkOk
            }
            Err(e) => set_error(
                TkStatus::TkErrModel,
                format!("tk_encode_batch failed: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_batch_fast(
    tokenizer: *const TkTokenizer,
    texts: *const *const c_char,
    count: usize,
    add_special_tokens: bool,
    out_encodings: *mut *mut TkEncoding,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if count == 0 {
            return TkStatus::TkOk;
        }
        if tokenizer.is_null() || texts.is_null() || out_encodings.is_null() {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_encode_batch_fast",
            );
        }
        let ptr_slice = slice::from_raw_parts(texts, count);
        let mut inputs = Vec::with_capacity(count);
        for (i, &p) in ptr_slice.iter().enumerate() {
            match cstr_to_str(p, &format!("texts[{}]", i)) {
                Ok(s) => inputs.push(s),
                Err(code) => return code,
            }
        }
        match (*tokenizer)
            .inner
            .encode_batch_fast(inputs, add_special_tokens)
        {
            Ok(encodings) => {
                let out_slice = slice::from_raw_parts_mut(out_encodings, count);
                for (i, enc) in encodings.into_iter().enumerate() {
                    out_slice[i] = Box::into_raw(Box::new(TkEncoding::from_encoding(enc)));
                }
                TkStatus::TkOk
            }
            Err(e) => set_error(
                TkStatus::TkErrModel,
                format!("tk_encode_batch_fast failed: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encode_batch_pair(
    tokenizer: *const TkTokenizer,
    texts_a: *const *const c_char,
    texts_b: *const *const c_char,
    count: usize,
    add_special_tokens: bool,
    out_encodings: *mut *mut TkEncoding,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if count == 0 {
            return TkStatus::TkOk;
        }
        if tokenizer.is_null()
            || texts_a.is_null()
            || texts_b.is_null()
            || out_encodings.is_null()
        {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_encode_batch_pair",
            );
        }
        let slice_a = slice::from_raw_parts(texts_a, count);
        let slice_b = slice::from_raw_parts(texts_b, count);
        let mut inputs = Vec::with_capacity(count);
        for i in 0..count {
            let sa = match cstr_to_str(slice_a[i], &format!("texts_a[{}]", i)) {
                Ok(s) => s,
                Err(code) => return code,
            };
            let sb = match cstr_to_str(slice_b[i], &format!("texts_b[{}]", i)) {
                Ok(s) => s,
                Err(code) => return code,
            };
            inputs.push((sa, sb));
        }
        match (*tokenizer).inner.encode_batch(inputs, add_special_tokens) {
            Ok(encodings) => {
                let out_slice = slice::from_raw_parts_mut(out_encodings, count);
                for (i, enc) in encodings.into_iter().enumerate() {
                    out_slice[i] = Box::into_raw(Box::new(TkEncoding::from_encoding(enc)));
                }
                TkStatus::TkOk
            }
            Err(e) => set_error(
                TkStatus::TkErrModel,
                format!("tk_encode_batch_pair failed: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_batch_free(encodings: *mut *mut TkEncoding, count: usize) {
    if encodings.is_null() || count == 0 {
        return;
    }
    let slice = slice::from_raw_parts_mut(encodings, count);
    for slot in slice.iter_mut() {
        if !slot.is_null() {
            tk_encoding_free(*slot);
            *slot = ptr::null_mut();
        }
    }
}

/* ========================================================================== *
 * Encoding Inspection, Manipulation & Alignment Queries
 * ========================================================================== */

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_free(encoding: *mut TkEncoding) {
    if !encoding.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(encoding));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_len(encoding: *const TkEncoding) -> usize {
    if encoding.is_null() {
        0
    } else {
        (*encoding).inner.len()
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_is_empty(encoding: *const TkEncoding) -> bool {
    if encoding.is_null() {
        true
    } else {
        (*encoding).inner.is_empty()
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_n_sequences(encoding: *const TkEncoding) -> usize {
    if encoding.is_null() {
        0
    } else {
        (*encoding).inner.n_sequences()
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_ids(
    encoding: *const TkEncoding,
    out_len: *mut usize,
) -> *const u32 {
    if encoding.is_null() {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return ptr::null();
    }
    let slice = (*encoding).inner.get_ids();
    if !out_len.is_null() {
        *out_len = slice.len();
    }
    slice.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_type_ids(
    encoding: *const TkEncoding,
    out_len: *mut usize,
) -> *const u32 {
    if encoding.is_null() {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return ptr::null();
    }
    let slice = (*encoding).inner.get_type_ids();
    if !out_len.is_null() {
        *out_len = slice.len();
    }
    slice.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_attention_mask(
    encoding: *const TkEncoding,
    out_len: *mut usize,
) -> *const u32 {
    if encoding.is_null() {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return ptr::null();
    }
    let slice = (*encoding).inner.get_attention_mask();
    if !out_len.is_null() {
        *out_len = slice.len();
    }
    slice.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_special_tokens_mask(
    encoding: *const TkEncoding,
    out_len: *mut usize,
) -> *const u32 {
    if encoding.is_null() {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return ptr::null();
    }
    let slice = (*encoding).inner.get_special_tokens_mask();
    if !out_len.is_null() {
        *out_len = slice.len();
    }
    slice.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_token(
    encoding: *const TkEncoding,
    index: usize,
) -> *const c_char {
    if encoding.is_null() {
        return ptr::null();
    }
    let enc = &*encoding;
    match enc.c_tokens.get(index) {
        Some(cstr) => cstr.as_ptr(),
        None => ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_offset(
    encoding: *const TkEncoding,
    index: usize,
    out_offset: *mut TkOffset,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.get_offsets().get(index) {
        Some(&(start, end)) => {
            if !out_offset.is_null() {
                *out_offset = TkOffset { start, end };
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_offsets(
    encoding: *const TkEncoding,
    out_offsets: *mut TkOffset,
    max_len: usize,
) -> usize {
    if encoding.is_null() || out_offsets.is_null() || max_len == 0 {
        return 0;
    }
    let offsets = (*encoding).inner.get_offsets();
    let count = offsets.len().min(max_len);
    let out_slice = slice::from_raw_parts_mut(out_offsets, count);
    for i in 0..count {
        out_slice[i] = TkOffset {
            start: offsets[i].0,
            end: offsets[i].1,
        };
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_word_id(
    encoding: *const TkEncoding,
    index: usize,
    out_word_id: *mut u32,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.get_word_ids().get(index) {
        Some(Some(wid)) => {
            if !out_word_id.is_null() {
                *out_word_id = *wid;
            }
            true
        }
        _ => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_word_ids(
    encoding: *const TkEncoding,
    out_word_ids: *mut i64,
    max_len: usize,
) -> usize {
    if encoding.is_null() || out_word_ids.is_null() || max_len == 0 {
        return 0;
    }
    let words = (*encoding).inner.get_word_ids();
    let count = words.len().min(max_len);
    let out_slice = slice::from_raw_parts_mut(out_word_ids, count);
    for i in 0..count {
        out_slice[i] = match words[i] {
            Some(w) => w as i64,
            None => -1,
        };
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_sequence_id(
    encoding: *const TkEncoding,
    index: usize,
    out_seq_id: *mut usize,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.token_to_sequence(index) {
        Some(seq) => {
            if !out_seq_id.is_null() {
                *out_seq_id = seq;
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_sequence_ids(
    encoding: *const TkEncoding,
    out_seq_ids: *mut i64,
    max_len: usize,
) -> usize {
    if encoding.is_null() || out_seq_ids.is_null() || max_len == 0 {
        return 0;
    }
    let seqs = (*encoding).inner.get_sequence_ids();
    let count = seqs.len().min(max_len);
    let out_slice = slice::from_raw_parts_mut(out_seq_ids, count);
    for i in 0..count {
        out_slice[i] = match seqs[i] {
            Some(s) => s as i64,
            None => -1,
        };
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_overflowing_count(
    encoding: *const TkEncoding,
) -> usize {
    if encoding.is_null() {
        0
    } else {
        (*encoding).inner.get_overflowing().len()
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_get_overflowing(
    encoding: *const TkEncoding,
    index: usize,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        if encoding.is_null() {
            return ptr::null_mut();
        }
        match (*encoding).inner.get_overflowing().get(index) {
            Some(ov) => Box::into_raw(Box::new(TkEncoding::from_encoding(ov.clone()))),
            None => ptr::null_mut(),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_word_to_tokens(
    encoding: *const TkEncoding,
    word_id: u32,
    sequence_id: usize,
    out_start_token: *mut usize,
    out_end_token: *mut usize,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.word_to_tokens(word_id, sequence_id) {
        Some((start, end)) => {
            if !out_start_token.is_null() {
                *out_start_token = start;
            }
            if !out_end_token.is_null() {
                *out_end_token = end;
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_word_to_chars(
    encoding: *const TkEncoding,
    word_id: u32,
    sequence_id: usize,
    out_offset: *mut TkOffset,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.word_to_chars(word_id, sequence_id) {
        Some((start, end)) => {
            if !out_offset.is_null() {
                *out_offset = TkOffset { start, end };
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_token_to_chars(
    encoding: *const TkEncoding,
    token_index: usize,
    out_sequence_id: *mut usize,
    out_offset: *mut TkOffset,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.token_to_chars(token_index) {
        Some((seq_id, (start, end))) => {
            if !out_sequence_id.is_null() {
                *out_sequence_id = seq_id;
            }
            if !out_offset.is_null() {
                *out_offset = TkOffset { start, end };
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_token_to_word(
    encoding: *const TkEncoding,
    token_index: usize,
    out_sequence_id: *mut usize,
    out_word_id: *mut u32,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.token_to_word(token_index) {
        Some((seq_id, word_id)) => {
            if !out_sequence_id.is_null() {
                *out_sequence_id = seq_id;
            }
            if !out_word_id.is_null() {
                *out_word_id = word_id;
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_char_to_token(
    encoding: *const TkEncoding,
    pos: usize,
    sequence_id: usize,
    out_token_index: *mut usize,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.char_to_token(pos, sequence_id) {
        Some(idx) => {
            if !out_token_index.is_null() {
                *out_token_index = idx;
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_char_to_word(
    encoding: *const TkEncoding,
    pos: usize,
    sequence_id: usize,
    out_word_id: *mut u32,
) -> bool {
    if encoding.is_null() {
        return false;
    }
    match (*encoding).inner.char_to_word(pos, sequence_id) {
        Some(wid) => {
            if !out_word_id.is_null() {
                *out_word_id = wid;
            }
            true
        }
        None => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_truncate(
    encoding: *mut TkEncoding,
    max_len: usize,
    stride: usize,
    direction: TkDirection,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if encoding.is_null() {
            return set_error(TkStatus::TkErrNullPtr, "Null `encoding` in tk_encoding_truncate");
        }
        if max_len == 0 || stride >= max_len {
            return set_error(
                TkStatus::TkErrTruncation,
                format!("Invalid truncate arguments: stride ({}) must be < max_len ({})", stride, max_len),
            );
        }
        let dir = match direction {
            TkDirection::TkDirectionRight => TruncationDirection::Right,
            TkDirection::TkDirectionLeft => TruncationDirection::Left,
        };
        let enc = &mut *encoding;
        enc.inner.truncate(max_len, stride, dir);
        enc.refresh_c_tokens();
        TkStatus::TkOk
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_pad(
    encoding: *mut TkEncoding,
    target_length: usize,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: *const c_char,
    direction: TkDirection,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if encoding.is_null() {
            return set_error(TkStatus::TkErrNullPtr, "Null `encoding` in tk_encoding_pad");
        }
        let ptok = if pad_token.is_null() {
            "[PAD]"
        } else {
            match cstr_to_str(pad_token, "pad_token") {
                Ok(s) => s,
                Err(code) => return code,
            }
        };
        let dir = match direction {
            TkDirection::TkDirectionRight => PaddingDirection::Right,
            TkDirection::TkDirectionLeft => PaddingDirection::Left,
        };
        let enc = &mut *encoding;
        enc.inner.pad(target_length, pad_id, pad_type_id, ptok, dir);
        enc.refresh_c_tokens();
        TkStatus::TkOk
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_encoding_merge(
    encodings: *const *const TkEncoding,
    count: usize,
    growing_offsets: bool,
) -> *mut TkEncoding {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if count > 0 && encodings.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `encodings` array in tk_encoding_merge");
            return ptr::null_mut();
        }
        let mut vec = Vec::with_capacity(count);
        if count > 0 {
            let ptrs = slice::from_raw_parts(encodings, count);
            for (i, &p) in ptrs.iter().enumerate() {
                if p.is_null() {
                    set_error(
                        TkStatus::TkErrNullPtr,
                        format!("Null `encodings[{}]` in tk_encoding_merge", i),
                    );
                    return ptr::null_mut();
                }
                vec.push((*p).inner.clone());
            }
        }
        let merged = Encoding::merge(vec, growing_offsets);
        Box::into_raw(Box::new(TkEncoding::from_encoding(merged)))
    })
}

/* ========================================================================== *
 * Decoding (Single, Batch, and Streaming)
 * ========================================================================== */

#[no_mangle]
pub unsafe extern "C" fn tk_decode(
    tokenizer: *const TkTokenizer,
    ids: *const u32,
    len: usize,
    skip_special_tokens: bool,
) -> *mut c_char {
    with_ffi_guard(ptr::null_mut(), || {
        clear_error_internal();
        if tokenizer.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `tokenizer` in tk_decode");
            return ptr::null_mut();
        }
        if len > 0 && ids.is_null() {
            set_error(TkStatus::TkErrNullPtr, "Null `ids` in tk_decode");
            return ptr::null_mut();
        }
        let slice = if len == 0 {
            &[]
        } else {
            slice::from_raw_parts(ids, len)
        };
        match (*tokenizer).inner.decode(slice, skip_special_tokens) {
            Ok(s) => string_to_c_heap(s),
            Err(e) => {
                set_error(TkStatus::TkErrModel, format!("Decode failed: {}", e));
                ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_decode_batch(
    tokenizer: *const TkTokenizer,
    batch_ids: *const *const u32,
    batch_lengths: *const usize,
    batch_size: usize,
    skip_special_tokens: bool,
    out_strings: *mut *mut c_char,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if batch_size == 0 {
            return TkStatus::TkOk;
        }
        if tokenizer.is_null()
            || batch_ids.is_null()
            || batch_lengths.is_null()
            || out_strings.is_null()
        {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_decode_batch",
            );
        }
        let ids_ptrs = slice::from_raw_parts(batch_ids, batch_size);
        let lengths = slice::from_raw_parts(batch_lengths, batch_size);
        let mut sentences: Vec<&[u32]> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let len = lengths[i];
            if len == 0 {
                sentences.push(&[]);
            } else if ids_ptrs[i].is_null() {
                return set_error(
                    TkStatus::TkErrNullPtr,
                    format!("Null `batch_ids[{}]` with non-zero length", i),
                );
            } else {
                sentences.push(slice::from_raw_parts(ids_ptrs[i], len));
            }
        }
        match (*tokenizer)
            .inner
            .decode_batch(&sentences, skip_special_tokens)
        {
            Ok(decoded) => {
                let out_slice = slice::from_raw_parts_mut(out_strings, batch_size);
                for (i, s) in decoded.into_iter().enumerate() {
                    out_slice[i] = string_to_c_heap(s);
                }
                TkStatus::TkOk
            }
            Err(e) => set_error(
                TkStatus::TkErrModel,
                format!("tk_decode_batch failed: {}", e),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_string_free(str_ptr: *mut c_char) {
    if !str_ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(CString::from_raw(str_ptr));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_string_batch_free(strings: *mut *mut c_char, count: usize) {
    if strings.is_null() || count == 0 {
        return;
    }
    let slice = slice::from_raw_parts_mut(strings, count);
    for slot in slice.iter_mut() {
        if !slot.is_null() {
            tk_string_free(*slot);
            *slot = ptr::null_mut();
        }
    }
}

#[no_mangle]
pub extern "C" fn tk_decode_stream_new(skip_special_tokens: bool) -> *mut TkDecodeStream {
    with_ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(TkDecodeStream {
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_decode_stream_step(
    stream: *mut TkDecodeStream,
    tokenizer: *const TkTokenizer,
    token_id: u32,
    out_chunk: *mut *mut c_char,
) -> TkStatus {
    with_ffi_guard(TkStatus::TkErrPanic, || {
        clear_error_internal();
        if stream.is_null() || tokenizer.is_null() || out_chunk.is_null() {
            return set_error(
                TkStatus::TkErrNullPtr,
                "Null pointer passed to tk_decode_stream_step",
            );
        }
        *out_chunk = ptr::null_mut();
        let st = &mut *stream;
        let tk = &(*tokenizer).inner;

        if tk.id_to_token(token_id).is_none() {
            return set_error(
                TkStatus::TkErrOutOfRange,
                format!("Invalid token ID {} in tk_decode_stream_step", token_id),
            );
        }

        st.ids.push(token_id);
        let decoded = match tk.decode(&st.ids, st.skip_special_tokens) {
            Ok(s) => s,
            Err(e) => {
                st.ids.pop();
                return set_error(
                    TkStatus::TkErrModel,
                    format!("DecodeStream decode failed: {}", e),
                );
            }
        };

        if decoded.ends_with('\u{FFFD}') {
            // Incomplete multi-byte UTF-8 sequence; wait for subsequent tokens.
            return TkStatus::TkOk;
        }

        if decoded.len() > st.prefix.len() && decoded.starts_with(&st.prefix) {
            let chunk = decoded[st.prefix.len()..].to_string();
            st.prefix = decoded;
            st.prefix_index = st.ids.len();
            *out_chunk = string_to_c_heap(chunk);
        }
        TkStatus::TkOk
    })
}

#[no_mangle]
pub unsafe extern "C" fn tk_decode_stream_free(stream: *mut TkDecodeStream) {
    if !stream.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(stream));
        }));
    }
}
