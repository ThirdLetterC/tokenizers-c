use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/tokenizers.h");

    // Copy the embeddable header into dist/include
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let src_header = PathBuf::from(&manifest_dir).join("include").join("tokenizers.h");
        let dist_include = PathBuf::from(&manifest_dir).join("dist").join("include");
        let dist_lib = PathBuf::from(&manifest_dir).join("dist").join("lib");
        let _ = fs::create_dir_all(&dist_include);
        let _ = fs::create_dir_all(&dist_lib);
        if src_header.exists() {
            let _ = fs::copy(&src_header, dist_include.join("tokenizers.h"));
        }
    }
}
