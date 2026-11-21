//! Contracts for the places where the native build recipe differs between
//! the hosts the engine supports. The module is a source file shared with
//! the build script, so these tests include it directly, the way the digest
//! contracts include the engine. Nothing here needs a native build.

#[path = "../native_host_platform.rs"]
mod native_host_platform;

use std::path::Path;

use native_host_platform::NativeHostPlatform;

#[test]
fn should_build_metal_with_its_kernels_inside_the_prefix_on_macos() {
    let configure_arguments =
        NativeHostPlatform::MacOs.mlx_backend_configure_arguments(Path::new("/opt/mlx"));

    assert_eq!(
        configure_arguments,
        vec!["-DMLX_METAL_PATH=/opt/mlx/lib", "-DMLX_METAL_JIT=ON"]
    );
}

#[test]
fn should_build_the_cpu_backend_alone_on_linux() {
    let configure_arguments =
        NativeHostPlatform::Linux.mlx_backend_configure_arguments(Path::new("/opt/mlx"));

    assert_eq!(configure_arguments, vec!["-DMLX_BUILD_METAL=OFF"]);
    assert!(
        !configure_arguments
            .iter()
            .any(
                |configure_argument| configure_argument.contains("METAL_PATH")
                    || configure_argument.contains("METAL_JIT")
            ),
        "a host without Metal must not be given Metal kernel settings"
    );
}

#[test]
fn should_expect_jaccl_only_where_mlx_builds_it() {
    assert!(NativeHostPlatform::MacOs.builds_jaccl());
    assert!(!NativeHostPlatform::Linux.builds_jaccl());
}

#[test]
fn should_name_the_missing_tool_in_the_install_hint_for_every_host() {
    for host_platform in [NativeHostPlatform::MacOs, NativeHostPlatform::Linux] {
        assert!(
            host_platform.tool_install_hint("cmake").contains("cmake"),
            "{host_platform:?} must say which tool to install"
        );
    }
    // Homebrew is the macOS route only.
    assert!(
        !NativeHostPlatform::Linux
            .tool_install_hint("cmake")
            .contains("brew")
    );
}

#[test]
fn should_recognize_the_host_these_tests_run_on() {
    // Boundary between the supported hosts and the rest: the lanes that run
    // this suite are macOS and Linux, and each must see itself.
    let expected_host_platform = if cfg!(target_os = "macos") {
        Some(NativeHostPlatform::MacOs)
    } else if cfg!(target_os = "linux") {
        Some(NativeHostPlatform::Linux)
    } else {
        None
    };

    assert_eq!(NativeHostPlatform::of_this_host(), expected_host_platform);
}

const LINUX_X86_64: &str = "x86_64-unknown-linux-gnu";
const LINUX_AARCH64: &str = "aarch64-unknown-linux-gnu";
const LINUX_X86_64_MUSL: &str = "x86_64-unknown-linux-musl";
const MACOS_AARCH64: &str = "aarch64-apple-darwin";
const MACOS_X86_64: &str = "x86_64-apple-darwin";

#[test]
fn should_link_when_the_crate_is_built_for_its_own_host() {
    assert_eq!(
        NativeHostPlatform::for_linking(
            Some(NativeHostPlatform::Linux),
            LINUX_X86_64,
            LINUX_X86_64,
            false
        ),
        Some(NativeHostPlatform::Linux)
    );
    assert_eq!(
        NativeHostPlatform::for_linking(
            Some(NativeHostPlatform::MacOs),
            MACOS_AARCH64,
            MACOS_AARCH64,
            false
        ),
        Some(NativeHostPlatform::MacOs)
    );
}

#[test]
fn should_not_link_when_the_crate_is_built_for_another_operating_system() {
    // The lint lane: a Linux runner compiling for Apple silicon. The pair
    // would be built for Linux and could never link into that target.
    assert_eq!(
        NativeHostPlatform::for_linking(
            Some(NativeHostPlatform::Linux),
            LINUX_X86_64,
            MACOS_AARCH64,
            false
        ),
        None
    );
}

#[test]
fn should_not_link_when_the_crate_is_built_for_another_processor_or_c_library() {
    // Same operating system, different target: the host's archives are the
    // wrong architecture or ABI for the binary being linked.
    for (host_platform, host_triple, build_triple) in [
        (NativeHostPlatform::Linux, LINUX_X86_64, LINUX_AARCH64),
        (NativeHostPlatform::Linux, LINUX_X86_64, LINUX_X86_64_MUSL),
        (NativeHostPlatform::MacOs, MACOS_AARCH64, MACOS_X86_64),
    ] {
        assert_eq!(
            NativeHostPlatform::for_linking(Some(host_platform), host_triple, build_triple, false),
            None,
            "{host_triple} building {build_triple} must not link"
        );
    }
}

#[test]
fn should_not_link_a_documentation_only_build() {
    // docs.rs builds for its Linux host as well as for the documented
    // target; none of those builds may start a native compilation.
    assert_eq!(
        NativeHostPlatform::for_linking(
            Some(NativeHostPlatform::Linux),
            LINUX_X86_64,
            LINUX_X86_64,
            true
        ),
        None
    );
}

#[test]
fn should_not_link_on_an_unsupported_host() {
    assert_eq!(
        NativeHostPlatform::for_linking(
            None,
            "x86_64-pc-windows-msvc",
            "x86_64-pc-windows-msvc",
            false
        ),
        None
    );
    // Boundary: empty triples, as when Cargo's variables are absent, never
    // count as a match for a supported host.
    assert_eq!(
        NativeHostPlatform::for_linking(Some(NativeHostPlatform::Linux), "", LINUX_X86_64, false),
        None
    );
}
