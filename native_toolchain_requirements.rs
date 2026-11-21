//! The oldest host toolchain releases the crate builds with, and the notes
//! that name them when a build fails because the host is older. A
//! self-contained, std-only source module shared by the build script and the
//! test tree.
//!
//! Both minimums were found by building on hosts one release too old: the
//! failures there are compiler diagnostics about the vendored sources, which
//! say nothing about the toolchain being the cause, so the build script adds
//! that sentence itself.

/// The oldest libclang that parses the vendored headers. bindgen parses with
/// the host's libclang for the aarch64-apple-darwin target on every host,
/// and libclang 14 and 15 reject the `__bf16` type that mlx/c/half.h
/// declares for that target ("__bf16 is not supported on this target").
pub const MINIMUM_LIBCLANG_MAJOR_VERSION: u32 = 16;

/// The oldest GCC whose standard library compiles the pinned MLX-C: its
/// private headers call C++20 `std::format`, which libstdc++ first ships in
/// GCC 13. macOS builds use Apple's toolchain and are not affected.
pub const MINIMUM_GCC_MAJOR_VERSION: u32 = 13;

/// The diagnostic a libclang older than the minimum reports for the vendored
/// half-precision declarations.
const UNSUPPORTED_BF16_DIAGNOSTIC: &str = "__bf16 is not supported on this target";

/// The sentence to add to a failed header parse when the diagnostic is the
/// one an outdated libclang produces, or `None` for any other failure, whose
/// own diagnostic already names its cause.
pub fn header_parse_failure_note(clang_diagnostic: &str) -> Option<String> {
    if !clang_diagnostic.contains(UNSUPPORTED_BF16_DIAGNOSTIC) {
        return None;
    }
    Some(format!(
        "bindgen parses the vendored headers with the host's libclang, which must be \
         version {MINIMUM_LIBCLANG_MAJOR_VERSION} or newer; older releases reject the \
         __bf16 type the headers declare. Install a newer libclang (for example \
         libclang-{MINIMUM_LIBCLANG_MAJOR_VERSION}-dev) and point LIBCLANG_PATH at it \
         if it is not the default."
    ))
}

/// What a Linux host needs beyond CMake for the automatic native build, for
/// the warning printed when that build fails.
pub fn linux_native_build_prerequisites() -> String {
    format!(
        "On Linux the build also needs a C++ compiler whose standard library provides \
         C++20 std::format (GCC {MINIMUM_GCC_MAJOR_VERSION} or newer) and the BLAS and \
         LAPACK development headers."
    )
}
