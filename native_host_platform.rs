//! The host operating systems the native build engine can compile the pinned
//! pair on, and the few places where the recipe differs between them. A
//! self-contained, std-only source module shared by the engine, the build
//! script and the test tree, like the forward edit set.
//!
//! MLX selects its backends from the host: Metal on macOS, and its CPU
//! backend alone on Linux, where no Metal stack exists. Everything else in
//! the recipe (the pins, the static archives, the gap shim) is the same, so
//! the difference is kept here instead of spreading host checks through the
//! engine.

use std::path::Path;

/// A host the engine can build and link the pinned pair on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeHostPlatform {
    /// Apple silicon macOS: MLX's Metal backend beside its CPU backend.
    MacOs,
    /// Linux: MLX's CPU backend alone, with BLAS and LAPACK from the system.
    Linux,
}

impl NativeHostPlatform {
    /// The platform of the machine running this code, or `None` on a host
    /// MLX's pinned release cannot be built on by this engine. Build scripts
    /// and xtask are compiled for the host, so `cfg!` names the host here.
    pub fn of_this_host() -> Option<Self> {
        if cfg!(target_os = "macos") {
            Some(Self::MacOs)
        } else if cfg!(target_os = "linux") {
            Some(Self::Linux)
        } else {
            None
        }
    }

    /// The platform the link ladder may run for, or `None` when this build
    /// must not link: the host is unsupported, the crate is being compiled
    /// for another target triple than the host's (the pair is built for the
    /// host, so a binary for any other operating system, processor or C
    /// library could never link it), or docs.rs is building the
    /// documentation, which never links and whose time-boxed, offline
    /// builder must not start a native compilation.
    // Only the build script's link ladder calls this; xtask includes the
    // engine, and with it this module, without linking anything.
    #[allow(dead_code)]
    pub fn for_linking(
        host_platform: Option<Self>,
        host_target_triple: &str,
        build_target_triple: &str,
        is_documentation_only_build: bool,
    ) -> Option<Self> {
        let host_platform = host_platform?;
        if is_documentation_only_build || host_target_triple != build_target_triple {
            return None;
        }
        Some(host_platform)
    }

    /// The CMake arguments that select MLX's backends for this host.
    pub fn mlx_backend_configure_arguments(self, prefix: &Path) -> Vec<String> {
        match self {
            Self::MacOs => vec![
                // The kernel library's baked path must live INSIDE the
                // prefix: MLX compiles METAL_PATH from this variable into
                // the binary, and with the build-tree default a consumer
                // machine must keep the entire (hundreds of megabytes)
                // build tree alive or every binary dies at startup — the
                // build-succeeds/load-dies class of defect this engine
                // exists to prevent.
                format!("-DMLX_METAL_PATH={}", prefix.join("lib").display()),
                "-DMLX_METAL_JIT=ON".to_owned(),
            ],
            // MLX already turns Metal off when the host is not Darwin; the
            // explicit switch keeps the recipe readable and independent of
            // that default.
            Self::Linux => vec!["-DMLX_BUILD_METAL=OFF".to_owned()],
        }
    }

    /// Whether MLX's build produces the jaccl archive on this host. jaccl is
    /// MLX's Thunderbolt transport for distributed work and MLX compiles it
    /// on Darwin only.
    pub fn builds_jaccl(self) -> bool {
        matches!(self, Self::MacOs)
    }

    /// How a person installs a missing build tool on this host, for the
    /// warning that names the tool.
    pub fn tool_install_hint(self, tool_name: &str) -> String {
        match self {
            Self::MacOs => format!("brew install {tool_name}"),
            Self::Linux => format!("install {tool_name} with the system package manager"),
        }
    }
}
