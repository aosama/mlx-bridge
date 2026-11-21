//! Contracts for the toolchain minimums and the notes that name them when a
//! host is too old. The module is a source file shared with the build script,
//! so these tests include it directly. Nothing here needs a native build.

#[path = "../native_toolchain_requirements.rs"]
mod native_toolchain_requirements;

use native_toolchain_requirements::{
    MINIMUM_GCC_MAJOR_VERSION, MINIMUM_LIBCLANG_MAJOR_VERSION, header_parse_failure_note,
    linux_native_build_prerequisites,
};

/// The build script's error text when libclang 14 (Debian 12's default)
/// parsed the vendored headers, captured verbatim from that build.
const OUTDATED_LIBCLANG_DIAGNOSTIC: &str = "ClangDiagnostic(\"/usr/lib/llvm-14/lib/clang/14.0.6/include/arm_bf16.h:14:9: \
     error: __bf16 is not supported on this target\\n/src/vendor/mlx-c/mlx/c/half.h:19:9: \
     error: __bf16 is not supported on this target\\n\")";

#[test]
fn should_name_the_minimum_libclang_when_the_parse_rejects_bf16() {
    let toolchain_note = header_parse_failure_note(OUTDATED_LIBCLANG_DIAGNOSTIC)
        .expect("the outdated-libclang diagnostic should get a toolchain note");

    assert!(toolchain_note.contains("libclang"), "{toolchain_note}");
    assert!(
        toolchain_note.contains(&format!(
            "version {MINIMUM_LIBCLANG_MAJOR_VERSION} or newer"
        )),
        "{toolchain_note}"
    );
}

#[test]
fn should_leave_an_unrelated_parse_failure_to_its_own_diagnostic() {
    let unrelated_diagnostics = [
        "ClangDiagnostic(\"mlx/c/array.h:3:10: fatal error: 'stdint.h' file not found\")",
        "",
        // The type name alone is not the outdated-libclang symptom.
        "unknown type name '__bf16_alias'",
    ];

    for unrelated_diagnostic in unrelated_diagnostics {
        assert_eq!(
            header_parse_failure_note(unrelated_diagnostic),
            None,
            "{unrelated_diagnostic:?} must not be blamed on the libclang release"
        );
    }
}

#[test]
fn should_name_the_minimum_gcc_and_the_blas_headers_for_linux() {
    let prerequisites = linux_native_build_prerequisites();

    assert!(
        prerequisites.contains(&format!("GCC {MINIMUM_GCC_MAJOR_VERSION} or newer")),
        "{prerequisites}"
    );
    assert!(prerequisites.contains("std::format"), "{prerequisites}");
    assert!(prerequisites.contains("BLAS and LAPACK"), "{prerequisites}");
}

#[test]
fn should_hold_the_minimums_measured_on_hosts_one_release_too_old() {
    // libclang 15 and GCC 12 were the newest releases that failed; a change
    // to either constant must come with a new measurement, not a guess.
    assert_eq!(MINIMUM_LIBCLANG_MAJOR_VERSION, 16);
    assert_eq!(MINIMUM_GCC_MAJOR_VERSION, 13);
}
