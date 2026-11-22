//! Standalone bindgen generation for the public MLX-C API, from the headers
//! vendored under `vendor/mlx-c/`, plus library discovery for the pinned
//! native pair.
//!
//! This build script translates the vendored headers of the pinned MLX-C
//! release into Rust declarations and emits the bridged-symbol inventory that
//! the coverage contract compares against those same headers.
//!
//! Linking a binary that uses this crate needs an MLX-C library. Discovery
//! order, first match wins: the `MLX_BRIDGE_MLX_C_PREFIX` environment
//! variable, the standard Homebrew prefixes, `pkg-config mlx-c`, and — the
//! zero-configuration default — the typed native-build engine in
//! `native_build.rs` (this crate's own code, no shell), which reuses the
//! machine-shared identity-keyed prefix or else downloads the checksum-verified
//! pinned sources and compiles them once into it. The
//! pair therefore compiles once per machine per pin, and every later
//! consumer reuses it. The ladder runs on macOS and on Linux, where the pair
//! is MLX's CPU backend alone; the link line for each host is composed in
//! `native_link_directives.rs`. Any other host gets a warning instead and
//! compiles or documents without linking.

#[path = "native_build.rs"]
mod native_build;
#[path = "native_link_directives.rs"]
mod native_link_directives;
#[path = "native_prefix_variable.rs"]
mod native_prefix_variable;
#[path = "native_toolchain_requirements.rs"]
mod native_toolchain_requirements;

use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
};

use native_build::NativeHostPlatform;
use native_link_directives::{
    MLX_EXPORTED_TARGETS_FILE, StaticPrefixArchives, exported_mlx_link_libraries,
    linux_static_pair_link_directives, macos_static_pair_link_directives,
    shared_pair_link_directives,
};
use native_prefix_variable::{
    FORMER_MLX_C_PREFIX_VARIABLE, MLX_C_PREFIX_VARIABLE, former_prefix_variable_notice,
};
use native_toolchain_requirements::{header_parse_failure_note, linux_native_build_prerequisites};
use pkg_config::Config as PkgConfig;

const VENDORED_HEADERS_DIRECTORY: &str = "vendor/mlx-c";

const BINDGEN_FUNCTION_ALLOWLIST: &str = "mlx_.*";
const BINDGEN_TYPE_ALLOWLIST: &str = "mlx_.*";
// The gap shim's own symbols: this repository's mlxrs_* C ABI for MLX
// functions the vendored MLX-C pin has not wrapped. They join the SAME
// bindgen parse (one combined umbrella) so shim declarations share the
// single type universe with the MLX-C declarations they build on.
const GAP_SHIM_FUNCTION_ALLOWLIST: &str = "mlxrs_.*";
const GAP_SHIM_TYPE_ALLOWLIST: &str = "mlxrs_.*";

/// The generated inventory module consumed by `raw.rs`; it exposes the exact
/// bridged symbol sets the coverage contract compares against the vendored
/// headers.
const BRIDGED_INVENTORY_FILE_NAME: &str = "bridged_inventory.rs";

/// docs.rs sets this variable in the environment of every build it runs.
const DOCS_RS_BUILD_VARIABLE: &str = "DOCS_RS";

/// The freestanding stubs that replace host libc in the self-contained
/// parse; see `self_contained_parse_arguments`.
const FREESTANDING_HEADERS_DIRECTORY: &str = "vendor/freestanding";

/// Clang arguments giving bindgen ONE parse identity on every host: the
/// aarch64 Apple-darwin view of the ABI this crate binds, in a
/// self-contained freestanding environment. Three distinct fatalities
/// surfaced while proving the Linux lane — each a different symptom of
/// letting host state describe the parse. First, the host libclang missed
/// the Debian/Ubuntu multiarch include directory and the host `<stdint.h>`
/// died on `bits/libc-header-start.h`. Second, the default triple did not
/// match the 64-bit host, so glibc's `gnu/stubs.h` selected the 32-bit
/// stubs. Third — the deepest, and invisible until the first two were
/// fixed — the vendored `half.h` gates the float16 and bfloat16 slice of
/// the ABI on `__aarch64__`, so any x86_64 parse silently DROPS symbols
/// like `mlx_array_data_bfloat16` that the bridge promises to carry; and
/// conversely an aarch64 parse of an x86_64 host's glibc is incoherent,
/// because that glibc's `stubs.h` speaks only the x86 family. The
/// conclusion is structural: no host header set can serve this parse.
/// `--target` names the Apple silicon ABI view; `-nostdsysteminc` drops
/// every host system include path while KEEPING clang's builtin headers
/// (`-nostdinc` would drop those too — float.h, stddef.h, and the ARM
/// intrinsic headers the vendored gates include); the freestanding stubs
/// vendored beside the headers supply the few libc-owned pieces they
/// touch (fixed-width integers, an opaque FILE, size_t via stddef, an
/// empty complex). docs.rs, the Ubuntu lint lane, and every Mac therefore
/// generate byte-identical declarations, and the bindings stop embedding
/// host FILE layouts nothing ever used.
fn self_contained_parse_arguments(freestanding_directory: &Path) -> Vec<String> {
    vec![
        "--target=aarch64-apple-darwin".to_owned(),
        // A frontend-only flag, so it rides behind -Xclang; -nostdinc would
        // also drop clang's builtin headers (float.h, stddef.h, the ARM
        // intrinsic headers), which the parse very much keeps.
        "-Xclang".to_owned(),
        "-nostdsysteminc".to_owned(),
        format!("-isystem{}", freestanding_directory.display()),
    ]
}

/// The header parse's failure, with the toolchain sentence appended when the
/// diagnostic is one an outdated libclang produces: that diagnostic names a
/// vendored header line, not the libclang release that is really at fault.
fn explain_header_parse_failure(parse_failure: bindgen::BindgenError) -> Box<dyn Error> {
    let parse_failure_text = parse_failure.to_string();
    match header_parse_failure_note(&parse_failure_text) {
        Some(toolchain_note) => format!("{parse_failure_text}\n{toolchain_note}").into(),
        None => parse_failure.into(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_directory =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is unset")?);
    let output_directory = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is unset")?);
    let include_directory = manifest_directory.join(VENDORED_HEADERS_DIRECTORY);
    let freestanding_directory = manifest_directory.join(FREESTANDING_HEADERS_DIRECTORY);
    let mlx_c_header = include_directory.join("mlx/c/mlx.h");
    if !mlx_c_header.is_file() {
        return Err(format!(
            "missing vendored MLX C umbrella header at {}",
            mlx_c_header.display()
        )
        .into());
    }

    println!("cargo:rerun-if-changed={}", include_directory.display());
    // The gap shim: header joined into the combined parse below, sources
    // compiled by the native engine, so both belong to the rebuild face.
    let gap_shim_include_directory = manifest_directory.join("native_gap_shim/include");
    let gap_shim_header = gap_shim_include_directory.join("mlxrs/gap_shim.h");
    if !gap_shim_header.is_file() {
        return Err(format!("missing gap shim header at {}", gap_shim_header.display()).into());
    }
    println!(
        "cargo:rerun-if-changed={}",
        manifest_directory.join("native_gap_shim").display()
    );
    let combined_umbrella_header = output_directory.join("combined_umbrella.h");
    std::fs::write(
        &combined_umbrella_header,
        "#include \"mlx/c/mlx.h\"\n#include \"mlxrs/gap_shim.h\"\n",
    )?;
    // The prefix variable only steers the link ladder; bindings-only
    // consumers (no `link` feature) never read it, so their builds do not
    // rebuild when it moves.
    if link_feature_is_enabled() {
        println!("cargo:rerun-if-env-changed={DOCS_RS_BUILD_VARIABLE}");
        println!("cargo:rerun-if-env-changed={MLX_C_PREFIX_VARIABLE}");
        // The former name is watched too, so removing it clears the warning
        // below on the next build instead of leaving it cached.
        println!("cargo:rerun-if-env-changed={FORMER_MLX_C_PREFIX_VARIABLE}");
    }

    let bindings = bindgen::Builder::default()
        // One combined umbrella: the vendored MLX-C tree plus the gap
        // shim's header, parsed together so the mlxrs_* declarations bind
        // against the same mlx_array/mlx_stream types the shim's C header
        // includes from mlx/c/mlx.h.
        .header(combined_umbrella_header.to_string_lossy())
        .clang_arg(format!("-I{}", include_directory.display()))
        .clang_arg(format!("-I{}", gap_shim_include_directory.display()))
        .clang_args(self_contained_parse_arguments(&freestanding_directory))
        .allowlist_function(BINDGEN_FUNCTION_ALLOWLIST)
        .allowlist_type(BINDGEN_TYPE_ALLOWLIST)
        .allowlist_function(GAP_SHIM_FUNCTION_ALLOWLIST)
        .allowlist_type(GAP_SHIM_TYPE_ALLOWLIST)
        .generate_comments(false)
        .generate()
        .map_err(explain_header_parse_failure)?;
    let bindings_content = bindings.to_string();
    write_bridged_inventory(&bindings_content, &output_directory)?;
    bindings.write_to_file(output_directory.join("mlx_c_bindings.rs"))?;

    // The link feature carries the entire link ladder. Bindings-only
    // consumers — those providing their own native MLX build, deliberately
    // not this crate's pinned pair — depend with default-features = false
    // and receive the declarations, the captured-error translation, the
    // owned handle types, and the operation families with ZERO link
    // directives: they own the linker, and this crate must not select a
    // library for them or trigger side effects like a pair build.
    if link_feature_is_enabled() {
        emit_link_instructions();
    }
    Ok(())
}

/// Whether the default-on `link` feature is active for this compilation.
/// Cargo exports every enabled feature as CARGO_FEATURE_<NAME>.
fn link_feature_is_enabled() -> bool {
    env::var_os("CARGO_FEATURE_LINK").is_some_and(|feature_value| feature_value != "0")
}

/// Emits the linker instructions for an MLX-C library. A build that cannot
/// link stops at a warning before any rung: a host that is neither macOS nor
/// Linux, a cross-compilation to any other target triple, or a docs.rs
/// documentation build. The pinned pair cannot serve any of them, so even an
/// explicit prefix variable would only yield unusable flags. Otherwise the
/// ladder, first match wins: the prefix variable, Homebrew (macOS only),
/// pkg-config, and finally the machine-shared identity-keyed prefix
/// (building it if absent — the zero-configuration default on a machine
/// that provides no MLX-C of its own). When nothing resolves, a warning is
/// printed and no link flags are emitted: library compilation and
/// documentation builds still work, and the failure surfaces only when a
/// consumer links a binary.
fn emit_link_instructions() {
    // Cargo hands every build script both triples; they differ exactly when
    // the crate is cross-compiled.
    let Some(host_platform) = NativeHostPlatform::for_linking(
        NativeHostPlatform::of_this_host(),
        &env::var("HOST").unwrap_or_default(),
        &env::var("TARGET").unwrap_or_default(),
        env::var_os(DOCS_RS_BUILD_VARIABLE).is_some(),
    ) else {
        // Documentation and check lanes never link, so the ladder stops
        // here with a warning instead of burning build minutes (docs.rs
        // builds are time-boxed) on a CMake attempt that cannot serve them.
        println!(
            "cargo:warning=no MLX-C library is linked for this build; the crate compiles \
             and documents, but linking a binary that uses it needs a build on macOS \
             (Apple silicon) or Linux for that same host, not a cross-compilation"
        );
        return;
    };
    warn_about_former_prefix_variable();
    if let Some(prefix) = prefix_from_environment() {
        emit_prefix_link_flags(&prefix, host_platform);
        return;
    }
    if host_platform == NativeHostPlatform::MacOs
        && let Some(prefix) = homebrew_prefix_with_mlxc()
    {
        emit_prefix_link_flags(&prefix, host_platform);
        return;
    }
    if let Ok(mlx_c) = PkgConfig::new().atleast_version("0.8.0").probe("mlx-c") {
        for library_name in &mlx_c.libs {
            println!("cargo:rustc-link-lib={library_name}");
        }
        for search_path in &mlx_c.link_paths {
            println!("cargo:rustc-link-search=native={}", search_path.display());
        }
        for framework in &mlx_c.frameworks {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        for framework_path in &mlx_c.framework_paths {
            println!(
                "cargo:rustc-link-search=framework={}",
                framework_path.display()
            );
        }
        return;
    }

    // Nothing located: the zero-configuration cold path. The typed engine in
    // `native_build.rs` compiles the pinned pair into the machine-shared
    // prefix — the same directory `cargo xtask build-native` maintains — so
    // this compilation is shared with every other consumer on the machine.
    match native_build::ensure_pinned_pair() {
        native_build::NativeBuildOutcome::Built | native_build::NativeBuildOutcome::Reused => {
            if let Some(prefix) = native_build::shared_prefix()
                .filter(|candidate| native_build::prefix_carries_mlxc_library(candidate))
            {
                emit_prefix_link_flags(&prefix, host_platform);
                return;
            }
            println!(
                "cargo:warning=the native build reported success but the shared prefix \
                 holds no MLX-C library; continuing without link flags"
            );
        }
        native_build::NativeBuildOutcome::Failed => println!(
            "cargo:warning=no MLX-C library could be located or built; the crate compiles, \
             but linking a binary that uses it will report undefined MLX symbols. Set \
             {MLX_C_PREFIX_VARIABLE}, run `cargo xtask build-native`, or install CMake \
             ({}) so the pinned sources can be built automatically.{}",
            host_platform.tool_install_hint("cmake"),
            if host_platform == NativeHostPlatform::Linux {
                format!(" {}", linux_native_build_prerequisites())
            } else {
                String::new()
            }
        ),
    }
}

/// Tells a build that still sets only the former prefix variable why its
/// prefix is not being used, before the ladder moves on without it.
fn warn_about_former_prefix_variable() {
    let prefix_variable_notice = former_prefix_variable_notice(
        env::var_os(MLX_C_PREFIX_VARIABLE).is_some(),
        env::var_os(FORMER_MLX_C_PREFIX_VARIABLE).is_some(),
    );
    if let Some(prefix_variable_notice) = prefix_variable_notice {
        println!("cargo:warning={prefix_variable_notice}");
    }
}

/// The explicit prefix override, validated before use.
fn prefix_from_environment() -> Option<PathBuf> {
    let prefix = PathBuf::from(env::var_os(MLX_C_PREFIX_VARIABLE)?);
    if native_build::prefix_carries_mlxc_library(&prefix) {
        return Some(prefix);
    }
    println!(
        "cargo:warning={MLX_C_PREFIX_VARIABLE} is set but {} holds no MLX-C library; \
         continuing down the discovery ladder",
        prefix.join("lib").display()
    );
    None
}

/// Finds a Homebrew installation whose `lib` already carries MLX-C (the
/// bottled `mlx-c` formula links here after `brew install mlx-c`).
fn homebrew_prefix_with_mlxc() -> Option<PathBuf> {
    let mut candidates = vec![PathBuf::from("/opt/homebrew")];
    if let Some(home) = env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join("homebrew"));
    }
    candidates
        .into_iter()
        .find(|prefix| native_build::prefix_carries_mlxc_library(prefix))
}

/// Emits the link flags for a prefix whose `lib` holds the libraries: the
/// two shared libraries plus an embedded rpath, or the static pair with
/// everything MLX needs spelled out for this host. The lines themselves are
/// composed in `native_link_directives.rs`.
fn emit_prefix_link_flags(prefix: &Path, host_platform: NativeHostPlatform) {
    let library_directory = prefix.join("lib");
    println!(
        "cargo:rustc-link-search=native={}",
        library_directory.display()
    );
    let link_directives = if native_build::prefix_carries_shared_mlxc_library(prefix) {
        shared_pair_link_directives(&library_directory)
    } else {
        static_prefix_link_directives(prefix, host_platform)
    };
    for link_directive in link_directives {
        println!("{link_directive}");
    }
}

/// The link directives for a static prefix on this host. On Linux the
/// libraries MLX was configured against are read back from the package MLX
/// installed into the prefix.
fn static_prefix_link_directives(prefix: &Path, host_platform: NativeHostPlatform) -> Vec<String> {
    let prefix_archives = StaticPrefixArchives {
        carries_gap_shim: native_build::prefix_carries_gap_shim_library(prefix),
        carries_gguf_archive: native_build::prefix_carries_gguf_archive(prefix),
    };
    match host_platform {
        NativeHostPlatform::MacOs => macos_static_pair_link_directives(prefix_archives),
        NativeHostPlatform::Linux => {
            let exported_targets_file = prefix.join(MLX_EXPORTED_TARGETS_FILE);
            let exported_link_libraries = std::fs::read_to_string(&exported_targets_file)
                .ok()
                .as_deref()
                .and_then(exported_mlx_link_libraries);
            if exported_link_libraries.is_none() {
                println!(
                    "cargo:warning={} does not name the libraries MLX was built against; \
                     linking continues without a BLAS and will report undefined symbols \
                     unless the consumer links one",
                    exported_targets_file.display()
                );
            }
            linux_static_pair_link_directives(
                prefix_archives,
                &exported_link_libraries.unwrap_or_default(),
            )
        }
    }
}

/// Extracts the bridged `mlx_*` functions and types plus the bridged
/// `mlxrs_*` gap-shim functions from the generated bindings and emits
/// them as public constant slices so the coverage contract can compare
/// the bridge against the vendored headers and the shim header without
/// re-running bindgen.
fn write_bridged_inventory(
    bindings_content: &str,
    output_directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let identifier_end = |segment: &str| {
        segment
            .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .unwrap_or(segment.len())
    };
    let mut bridged_functions: Vec<String> = bindings_content
        .split("pub fn ")
        .skip(1)
        .map(|segment| &segment[..identifier_end(segment)])
        .filter(|name| name.starts_with("mlx_"))
        .map(str::to_owned)
        .collect();
    bridged_functions.sort();
    bridged_functions.dedup();

    // bindgen 0.73 skips C variadic functions; the hand-written declaration
    // in `raw.rs` restores the one variadic MLX-C entry point.
    bridged_functions.push("mlx_error".to_owned());

    let mut bridged_types: Vec<String> = ["pub struct ", "pub enum ", "pub union ", "pub type "]
        .iter()
        .flat_map(|marker| {
            bindings_content
                .split(marker)
                .skip(1)
                .map(|segment| &segment[..identifier_end(segment)])
                .filter(|name| name.starts_with("mlx_"))
                .map(str::to_owned)
                .collect::<Vec<String>>()
        })
        .collect();
    // bindgen aliases constified C enums through `pub use self::mlx_x_ as
    // mlx_x;`; the alias is the name the headers expose, so it belongs in the
    // inventory alongside the tag.
    for alias_segment in bindings_content.split("pub use self::").skip(1) {
        let Some((_tag, alias_with_rest)) = alias_segment.split_once(" as ") else {
            continue;
        };
        let alias = &alias_with_rest[..identifier_end(alias_with_rest)];
        if alias.starts_with("mlx_") {
            bridged_types.push(alias.to_owned());
        }
    }
    bridged_types.sort();
    bridged_types.dedup();

    // The gap shim's bridged functions, kept as their own slice: shim
    // symbols are this repository's surface, not the vendored MLX-C
    // surface, and the audit and inventory contract track the two
    // separately so shim entries can retire the day MLX-C catches up.
    let mut gap_shim_bridged_functions: Vec<String> = bindings_content
        .split("pub fn ")
        .skip(1)
        .map(|segment| &segment[..identifier_end(segment)])
        .filter(|name| name.starts_with("mlxrs_"))
        .map(str::to_owned)
        .collect();
    gap_shim_bridged_functions.sort();
    gap_shim_bridged_functions.dedup();

    let inventory_source = format!(
        "// Generated by this crate's build script from the bindgen output.\n\
         // Every listed symbol exists in the compiled bridge; the coverage\n\
         // contract compares this inventory against the vendored headers.\n\n\
         /// Every MLX-C function bridged by the generated raw declarations.\n\
         pub const BRIDGED_FUNCTIONS: [&str; {}] = {:?};\n\n\
         /// Every MLX-C type bridged by the generated raw declarations.\n\
         pub const BRIDGED_TYPES: [&str; {}] = {:?};\n\n\
         /// Every gap-shim function bridged by the generated raw\n\
         /// declarations: this repository's mlxrs_* surface over the MLX\n\
         /// functions the vendored MLX-C pin has not wrapped.\n\
         pub const GAP_SHIM_BRIDGED_FUNCTIONS: [&str; {}] = {:?};\n",
        bridged_functions.len(),
        bridged_functions,
        bridged_types.len(),
        bridged_types,
        gap_shim_bridged_functions.len(),
        gap_shim_bridged_functions,
    );
    std::fs::write(
        output_directory.join(BRIDGED_INVENTORY_FILE_NAME),
        inventory_source,
    )?;
    Ok(())
}
