//! The per-worker MLX bindings context: its submission stream and the linked
//! version.
//!
//! The context holds exactly the bindings-owned state (which stream work is
//! submitted to, which upstream version is linked), while embedding policy —
//! memory limits and the metallib path selection — stays with the embedding
//! application, which composes this context.

use std::ffi::CStr;

use crate::{MlxCError, MlxStream, error::check_status, raw};

/// Process-global official MLX C bindings context configured for one worker.
#[derive(Debug)]
pub struct MlxBindingsContext {
    submission_stream: MlxStream,
    version: String,
}

impl MlxBindingsContext {
    /// Creates the context after the non-terminating error handler is
    /// installed: reads the linked version and acquires the default GPU stream.
    pub fn new() -> Result<Self, MlxCError> {
        let version = read_mlx_version()?;
        let submission_stream = MlxStream::default_gpu()?;
        Ok(Self {
            submission_stream,
            version,
        })
    }

    /// Creates a context whose submissions target the default CPU stream.
    ///
    /// MLX implements several families only on the CPU, including matrix
    /// factorization, inversion, and solving, and its GPU backend rejects
    /// float64. Every wrapper submits to its context's stream, so those
    /// families are reachable only through a context built here.
    pub fn on_cpu() -> Result<Self, MlxCError> {
        let version = read_mlx_version()?;
        let submission_stream = MlxStream::default_cpu()?;
        Ok(Self {
            submission_stream,
            version,
        })
    }

    /// Creates a context whose submissions target `stream`, such as one from
    /// `MlxStream::on_device`. Streams are thread affine unless built through
    /// `MlxStream::new_thread_unsafe`, so use this context on the thread that
    /// created the stream.
    pub fn on_stream(stream: MlxStream) -> Result<Self, MlxCError> {
        let version = read_mlx_version()?;
        Ok(Self {
            submission_stream: stream,
            version,
        })
    }

    /// The stream every submission from this context targets.
    pub const fn stream(&self) -> &MlxStream {
        &self.submission_stream
    }

    /// The submission stream under its original name, which is the GPU stream
    /// unless the context was built through `on_cpu` or `on_stream`; prefer
    /// `stream`.
    pub const fn gpu_stream(&self) -> &MlxStream {
        self.stream()
    }

    /// The linked upstream MLX version.
    pub fn version(&self) -> &str {
        &self.version
    }
}

fn read_mlx_version() -> Result<String, MlxCError> {
    // SAFETY: The error handler is installed before this constructor and the
    // returned owned handle is released on every subsequent path.
    let raw_version = unsafe { raw::mlx_string_new() };
    let mut owned_version = OwnedMlxString(raw_version);
    // SAFETY: `owned_version` contains a live MLX string handle and this call
    // only replaces its owned string value.
    let status = unsafe { raw::mlx_version(&mut owned_version.0) };
    check_status(status, "read the linked MLX version")?;
    // SAFETY: The pointer remains valid while `owned_version` is live.
    let version_pointer = unsafe { raw::mlx_string_data(owned_version.0) };
    if version_pointer.is_null() {
        return Err(MlxCError {
            operation: "read the linked MLX version",
            description: "MLX returned a null version string".to_owned(),
        });
    }
    // SAFETY: MLX C documents a null-terminated string owned by the handle.
    Ok(unsafe { CStr::from_ptr(version_pointer) }
        .to_string_lossy()
        .into_owned())
}

struct OwnedMlxString(raw::mlx_string);

impl Drop for OwnedMlxString {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live handle exactly once and does not
        // use it afterward. MLX C accepts the value form for destruction.
        unsafe {
            raw::mlx_string_free(self.0);
        }
    }
}
