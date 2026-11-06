//! Shared location helper for the xtask commands, so every command
//! resolves the repository root through one contract instead of a copy
//! per module.

use std::path::{Path, PathBuf};

/// The repository root, resolved from the compiled-in xtask location so
/// every command is faithful no matter which directory invoked it. The
/// xtask package always lives exactly one level below the root, which is
/// what makes the compiled-in location a reliable anchor.
pub fn repository_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .expect("the xtask package always lives one level below the root")
        .to_path_buf()
}

/// Every Rust source under `directory`, recursively.
pub fn collect_rust_sources(
    directory: &Path,
    source_paths: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for directory_entry in std::fs::read_dir(directory)? {
        let entry_path = directory_entry?.path();
        if entry_path.is_dir() {
            collect_rust_sources(&entry_path, source_paths)?;
        } else if entry_path.extension().is_some_and(is_rust_extension) {
            source_paths.push(entry_path);
        }
    }
    Ok(())
}

fn is_rust_extension(extension: &std::ffi::OsStr) -> bool {
    extension == "rs"
}
