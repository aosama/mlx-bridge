//! The upstream surface audit: prove which MLX functions the vendored
//! MLX-C tree actually carries, so a capability MLX ships cannot silently
//! lack a Rust path into this crate.
//!
//! The audit has two legs:
//!
//! 1. The upstream leg (needs `--mlx-source <checkout>`): parse the public
//!    C++ free functions out of every MLX header the pinned checkout's
//!    umbrella `mlx/mlx.h` reaches, directly or through other MLX headers
//!    (what a consumer of the umbrella can call), map each C++ name onto the
//!    C naming convention MLX-C follows, and report every MLX function with
//!    neither a C wrapper nor a gap shim entry. Those functions are the only
//!    place coverage can be lost upstream of this repository: the build
//!    script's bindgen translation is mechanical and full-tree, so
//!    anything MLX-C declares is necessarily bridged here.
//!
//! 2. The inventory leg (always runs): re-extract the vendored MLX-C C
//!    function surface from the vendored umbrella header (following local
//!    includes transitively, the same closure bindgen parses) and compare
//!    it with the function rows recorded in `COVERAGE_INVENTORY.md`, so a
//!    pin bump or vendored-header edit that forgets to refresh the record
//!    fails loudly. Types stay the build script's inventory domain: they
//!    are emitted per compile from the bindgen output, where the type
//!    extraction is exact rather than regex-approximate.
//!
//! Matching is deliberately layered: an exact family-prefixed name first
//! (`mlx::core::linalg::norm` → `mlx_linalg_norm`), then the bare prefix
//! (`fft::fftn` also tries `mlx_fftn`), then a suffix match for names the
//! C side composes differently (`cuda_kernel` family under `mlx_fast_`).
//! A gap shim entry counts when it carries the same family-prefixed name with
//! `mlxrs_` in place of `mlx_` (`config::get` → `mlxrs_config_get`).
//! Everything still unmatched lands in the report against the reviewed
//! allowlist below, which records why each surviving gap is accepted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::inventory_records::{audit_coverage_inventory_record, audit_gap_shim_record};
use crate::repository;
use crate::safe_wrapper_coverage::audit_safe_wrapper_coverage;
use crate::surface_parsing::{
    cpp_public_free_function_names, function_names_in_c_source, function_names_with_prefix,
    quoted_includes,
};

/// MLX public functions that MLX-C verifiably does not wrap, kept reviewed
/// so the audit fails only on NEW gaps. Every entry pairs the C++ function
/// name with the reason the gap is accepted. Verified against the pinned
/// pair MLX v0.32.5 + MLX-C v0.8.0 (the vendored header tree).
const REVIEWED_UNWRAPPED_MLX_FUNCTIONS: &[(&str, &str)] = &[
    (
        "allocator",
        "returns MLX's C++ allocator object, which C cannot hold; the memory \
         controls it carries reach Rust through MLX-C's memory family",
    ),
    (
        "can_reuse_alien_buffer",
        "on Metal it reads the storage mode of an MTL::Buffer object, which this \
         crate never holds: MlxArray::from_managed_buffer adopts host memory, and \
         a host pointer passed here would be read as a Metal object",
    ),
    (
        "exporter",
        "served by the mlx_function_exporter_* family under the exporter-suffix naming",
    ),
    (
        "get_arch_list",
        "reports the CUDA architectures MLX was compiled for; this crate builds MLX \
         without the CUDA backend on every host it supports",
    ),
    (
        "import_function",
        "served by the mlx_imported_function_* family under the imported naming",
    ),
    (
        "precompiled_cuda_kernel",
        "CUDA runs only on CUDA hardware; this crate builds MLX without the CUDA backend \
         on every host it supports",
    ),
    (
        "register_config_handler",
        "installs one C function that answers every configuration lookup in the \
         process, MLX's own included, from MLX's worker threads; config_update, \
         config_get, and MlxConfigScope cover setting and reading values",
    ),
    (
        "to_stream",
        "converts the C++ StreamOrDevice variant, which the C API does not model",
    ),
];

/// The C-name prefixes tried for each MLX header, in order. The header file
/// stem names the MLX family; most families mirror their namespace in the
/// C prefix, and the bare `mlx_` fallback catches the members the C side
/// hoisted into the top namespace (for example `mlx_async_eval`).
fn candidate_prefixes_for_header(header_relative_path: &str) -> &'static [&'static str] {
    // MLX-side paths carry the mlx/ prefix ("mlx/distributed/ops.h");
    // routing cares about the path inside the mlx namespace.
    let header_relative_path = header_relative_path.trim_start_matches("mlx/");
    let header_file_stem = header_relative_path
        .rsplit('/')
        .next()
        .unwrap_or(header_relative_path)
        .trim_end_matches(".h");
    match header_file_stem {
        "linalg" => &["mlx_linalg_", "mlx_"],
        "fft" => &["mlx_fft_", "mlx_"],
        "fast" => &["mlx_fast_", "mlx_"],
        "random" => &["mlx_random_", "mlx_"],
        // The backend headers carry their C families under their own
        // prefix; the fast fallback exists because MLX-C's fast.h also
        // exposes cuda-kernel entry points as mlx_fast_cuda_kernel_*.
        "metal" => &["mlx_metal_", "mlx_"],
        // No bare fallback: the config functions are generic words (get,
        // update), and a future top-level MLX-C mlx_get or mlx_update would
        // otherwise mark them wrapped without having anything to do with
        // configuration.
        "config" => &["mlx_config_"],
        "cuda" => &["mlx_cuda_", "mlx_fast_", "mlx_"],
        "distributed" | "ops" if header_relative_path.starts_with("distributed/") => {
            &["mlx_distributed_", "mlx_"]
        }
        _ => &["mlx_"],
    }
}

/// One parsed MLX public function: the C++ name and the vendored-relative
/// header it was declared in (the header drives the C-name candidates).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MlxPublicFunction {
    header_relative_path: String,
    function_name: String,
}

pub fn run(arguments: &[String]) -> ExitCode {
    let mut mlx_source_path: Option<PathBuf> = None;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        let argument = &arguments[argument_index];
        if argument == "--mlx-source" && argument_index + 1 < arguments.len() {
            mlx_source_path = Some(PathBuf::from(&arguments[argument_index + 1]));
            argument_index += 2;
        } else if argument == "--mlx-source" {
            eprintln!("audit-upstream-surface: --mlx-source needs a path to an MLX checkout");
            print_usage();
            return ExitCode::FAILURE;
        } else if let Some(path_tail) = argument.strip_prefix("--mlx-source=") {
            mlx_source_path = Some(PathBuf::from(path_tail));
            argument_index += 1;
        } else {
            eprintln!("audit-upstream-surface: unknown argument {argument:?}");
            print_usage();
            return ExitCode::FAILURE;
        }
    }

    let repository_root = repository::repository_root();
    let vendored_c_functions = extract_vendored_c_functions(&repository_root);
    if vendored_c_functions.is_empty() {
        eprintln!("audit-upstream-surface: no vendored MLX-C functions found under vendor/");
        return ExitCode::FAILURE;
    }
    println!(
        "[audit] vendored MLX-C surface: {} functions",
        vendored_c_functions.len()
    );

    let safe_wrappers_complete =
        audit_safe_wrapper_coverage(&repository_root, &vendored_c_functions);
    let inventory_in_sync =
        audit_coverage_inventory_record(&repository_root, &vendored_c_functions)
            && safe_wrappers_complete;
    let gap_shim_functions = extract_gap_shim_functions(&repository_root);
    if gap_shim_functions.is_empty() {
        eprintln!("audit-upstream-surface: no gap shim functions found under native_gap_shim/");
        return ExitCode::FAILURE;
    }
    println!(
        "[audit] gap shim surface: {} functions",
        gap_shim_functions.len()
    );
    let shim_record_in_sync = audit_gap_shim_record(&repository_root, &gap_shim_functions);

    let upstream_audit_passed = match mlx_source_path {
        Some(checkout_path) => {
            audit_upstream_surface(&checkout_path, &vendored_c_functions, &gap_shim_functions)
        }
        None => {
            println!("[audit] upstream leg skipped: pass --mlx-source <mlx checkout> to run it");
            true
        }
    };

    if inventory_in_sync && shim_record_in_sync && upstream_audit_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn print_usage() {
    eprintln!(
        "usage: cargo xtask audit-upstream-surface [--mlx-source <path-to-mlx-checkout>]\n\n\
         Audits that no MLX public function lacks a C wrapper in the vendored\n\
         MLX-C tree (the only layer where coverage can be lost upstream), and\n\
         that COVERAGE_INVENTORY.md's function record matches the vendored\n\
         headers. The upstream leg needs a checkout of the pinned MLX tag\n\
         (PINNED_MLX_TAG in native_build.rs); the inventory leg always runs."
    );
}

/// Extract every `mlx_*` function the vendored umbrella header tree
/// declares. Follows quoted includes transitively from
/// `vendor/mlx-c/mlx/c/mlx.h` — the same closure the build script hands to
/// bindgen — so the extraction is comparable with the recorded inventory.
fn extract_vendored_c_functions(repository_root: &Path) -> BTreeSet<String> {
    let include_root = repository_root.join("vendor/mlx-c");
    let umbrella_path = include_root.join("mlx/c/mlx.h");
    let mut visited_headers: BTreeSet<PathBuf> = BTreeSet::new();
    let mut declared_functions: BTreeSet<String> = BTreeSet::new();
    visit_vendored_header(
        &umbrella_path,
        &include_root,
        &mut visited_headers,
        &mut declared_functions,
    );
    declared_functions
}

fn visit_vendored_header(
    header_path: &Path,
    include_root: &Path,
    visited_headers: &mut BTreeSet<PathBuf>,
    declared_functions: &mut BTreeSet<String>,
) {
    if !visited_headers.insert(header_path.to_path_buf()) {
        return;
    }
    let header_source = match std::fs::read_to_string(header_path) {
        Ok(source) => source,
        Err(read_error) => {
            eprintln!(
                "[audit] cannot read vendored header {}: {read_error}",
                header_path.display()
            );
            return;
        }
    };
    for function_name in function_names_in_c_source(&header_source) {
        declared_functions.insert(function_name);
    }
    for included_relative_path in quoted_includes(&header_source) {
        // The vendored tree's quoted includes are include-root relative
        // (they all read "mlx/c/…", matching the build script's single -I
        // on vendor/mlx-c); the including file's own directory is kept as
        // a secondary resolution so the walker survives a tree that ever
        // mixes both styles.
        let root_resolved = include_root.join(&included_relative_path);
        let included_path = if root_resolved.is_file() {
            root_resolved
        } else {
            header_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&included_relative_path)
        };
        visit_vendored_header(
            &included_path,
            include_root,
            visited_headers,
            declared_functions,
        );
    }
}

/// Parse the MLX checkout's public C++ free functions from every header the
/// umbrella `mlx/mlx.h` reaches (see `public_mlx_headers`).
fn extract_mlx_public_functions(mlx_checkout_root: &Path) -> BTreeSet<MlxPublicFunction> {
    let mut public_functions = BTreeSet::new();
    for header_relative_path in public_mlx_headers(mlx_checkout_root) {
        let header_path = mlx_checkout_root.join(&header_relative_path);
        let Some(header_source) = std::fs::read_to_string(&header_path).ok() else {
            continue;
        };
        for function_name in cpp_public_free_function_names(&header_source) {
            public_functions.insert(MlxPublicFunction {
                header_relative_path: header_relative_path.clone(),
                function_name,
            });
        }
    }
    public_functions
}

/// Every MLX header the umbrella `mlx/mlx.h` reaches, directly or through
/// other MLX headers. MLX's public surface is what a consumer of the
/// umbrella can call, and MLX also declares public functions in headers the
/// umbrella includes only indirectly: MLX v0.32.5's `config` module arrives
/// through `mlx/utils.h`, and `allocator.h` and `dtype.h` arrive the same
/// way. Walking only the umbrella's own includes missed all three. Only
/// quoted `mlx/` includes that exist in the checkout are followed, so system
/// and third-party headers never enter the audit.
fn public_mlx_headers(mlx_checkout_root: &Path) -> BTreeSet<String> {
    let mut reached_headers = BTreeSet::new();
    let mut headers_to_visit = vec!["mlx/mlx.h".to_owned()];
    while let Some(header_relative_path) = headers_to_visit.pop() {
        let header_path = mlx_checkout_root.join(&header_relative_path);
        let Some(header_source) = std::fs::read_to_string(&header_path).ok() else {
            continue;
        };
        for included_relative_path in quoted_includes(&header_source) {
            let is_mlx_header = included_relative_path.starts_with("mlx/");
            if is_mlx_header && !reached_headers.contains(&included_relative_path) {
                reached_headers.insert(included_relative_path.clone());
                headers_to_visit.push(included_relative_path);
            }
        }
    }
    reached_headers
}

/// The composition layer: MLX-C sometimes re-composes a C++ name's words
/// in a different order (`new_stream` → `mlx_stream_new`,
/// `device_info` → `mlx_device_info_new`). A multi-word C++ name counts
/// as wrapped when some C symbol's underscore segments contain every
/// segment of the C++ name. Single-word names are deliberately excluded:
/// a lone word (`eval`) must never fuzzy-match a composed C name
/// (`mlx_async_eval`), which would silently mask a real gap.
fn is_wrapped_by_composition(function_name: &str, vendored_c_functions: &BTreeSet<String>) -> bool {
    let required_segments: Vec<&str> = function_name.split('_').collect();
    if required_segments.len() < 2 {
        return false;
    }
    vendored_c_functions.iter().any(|c_function_name| {
        let c_segments: std::collections::BTreeSet<&str> = c_function_name.split('_').collect();
        required_segments
            .iter()
            .all(|required_segment| c_segments.contains(required_segment))
    })
}

/// The repository's gap-shim declarations: the mlxrs_* functions the
/// native engine compiles as the third component, read straight from the
/// shim's own header (the single source of truth for its surface).
fn extract_gap_shim_functions(repository_root: &Path) -> BTreeSet<String> {
    let shim_header_path = repository_root.join("native_gap_shim/include/mlxrs/gap_shim.h");
    let Ok(shim_header_source) = std::fs::read_to_string(&shim_header_path) else {
        return BTreeSet::new();
    };
    function_names_with_prefix(&shim_header_source, "mlxrs_")
        .into_iter()
        .collect()
}

/// The suffixes a shim entry adds when one C++ overload set needs separate C
/// entry points. Only these count as overloads: `mlxrs_einsum_path` serves the
/// separate public function `einsum_path`, not an overload of `einsum`.
const GAP_SHIM_OVERLOAD_SUFFIXES: &[&str] = &["axes", "axis"];

/// The shim entry serving one MLX function, if any. Shim symbols follow
/// MLX-C's naming with `mlxrs_` in place of `mlx_`, so each family prefix
/// for the declaring header is tried (`config::get` → `mlxrs_config_get`),
/// as an exact name or one of its overload entry points. A name that only
/// extends the function's (`mlxrs_fast_scaled_dot_product_attention_windowed`)
/// is a separate entry, not this function's wrapper.
fn gap_shim_entry_for(
    function_name: &str,
    header_relative_path: &str,
    gap_shim_functions: &BTreeSet<String>,
) -> Option<String> {
    for c_prefix in candidate_prefixes_for_header(header_relative_path) {
        let shim_prefix = c_prefix.replacen("mlx_", "mlxrs_", 1);
        let exact_shim_symbol = format!("{shim_prefix}{function_name}");
        if gap_shim_functions.contains(&exact_shim_symbol) {
            return Some(exact_shim_symbol);
        }
        for overload_suffix in GAP_SHIM_OVERLOAD_SUFFIXES {
            let overload_symbol = format!("{exact_shim_symbol}_{overload_suffix}");
            if gap_shim_functions.contains(&overload_symbol) {
                return Some(overload_symbol);
            }
        }
    }
    None
}

/// The upstream leg: map every MLX public function onto the vendored C
/// surface plus the repository's gap shim, and report the ones with no
/// wrapper anywhere.
fn audit_upstream_surface(
    mlx_checkout_root: &Path,
    vendored_c_functions: &BTreeSet<String>,
    gap_shim_functions: &BTreeSet<String>,
) -> bool {
    let public_functions = extract_mlx_public_functions(mlx_checkout_root);
    if public_functions.is_empty() {
        eprintln!(
            "[audit] no MLX public functions parsed from {} — is this a \
             checkout of the MLX source tree?",
            mlx_checkout_root.display()
        );
        return false;
    }

    let mut unwrapped: BTreeMap<String, Vec<&MlxPublicFunction>> = BTreeMap::new();
    let mut exact_match_count = 0usize;
    let mut composition_match_count = 0usize;
    let mut shim_wrap_count = 0usize;
    let mut redundant_shim_count = 0usize;
    for public_function in &public_functions {
        let candidate_prefixes =
            candidate_prefixes_for_header(&public_function.header_relative_path);
        let exact_candidate = candidate_prefixes
            .iter()
            .map(|prefix| format!("{prefix}{}", public_function.function_name))
            .find(|candidate| vendored_c_functions.contains(candidate.as_str()));
        let wrapped_by_mlxc_composition =
            is_wrapped_by_composition(&public_function.function_name, vendored_c_functions);
        let shim_entry = gap_shim_entry_for(
            &public_function.function_name,
            &public_function.header_relative_path,
            gap_shim_functions,
        );
        if exact_candidate.is_some() || wrapped_by_mlxc_composition {
            if exact_candidate.is_some() {
                exact_match_count += 1;
            } else {
                composition_match_count += 1;
            }
            // The retirement governor: MLX-C now carries this function, so
            // a shim entry for it is redundant surface to retire.
            if let Some(redundant_shim_symbol) = shim_entry {
                redundant_shim_count += 1;
                eprintln!(
                    "[audit]   REDUNDANT SHIM: {redundant_shim_symbol} — MLX-C now \
                     wraps {}; retire the shim entry and move the wrapper to \
                     the MLX-C family",
                    public_function.function_name
                );
            }
            continue;
        }
        if shim_entry.is_some() {
            shim_wrap_count += 1;
            continue;
        }
        unwrapped
            .entry(public_function.function_name.clone())
            .or_default()
            .push(public_function);
    }

    println!(
        "[audit] MLX public free functions: {} across {} headers",
        public_functions.len(),
        public_functions
            .iter()
            .map(|function| function.header_relative_path.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    );
    println!(
        "[audit] wrapped: {} by MLX-C ({} exact, {} composition), {} by the gap shim",
        exact_match_count + composition_match_count,
        exact_match_count,
        composition_match_count,
        shim_wrap_count
    );

    let mut new_gap_count = 0usize;
    let mut reviewed_gap_count = 0usize;
    let mut sorted_unwrapped: Vec<(&String, &Vec<&MlxPublicFunction>)> = unwrapped.iter().collect();
    sorted_unwrapped.sort_by(|left, right| left.0.cmp(right.0));
    for (function_name, declarations) in &sorted_unwrapped {
        match REVIEWED_UNWRAPPED_MLX_FUNCTIONS
            .iter()
            .find(|(reviewed_name, _)| reviewed_name == function_name)
        {
            Some(_) => reviewed_gap_count += 1,
            None => new_gap_count += declarations.len(),
        }
    }
    println!(
        "[audit] unwrapped upstream functions: {} ({} reviewed, {} NEW)",
        sorted_unwrapped.len(),
        reviewed_gap_count,
        new_gap_count
    );
    for (function_name, declarations) in &sorted_unwrapped {
        match REVIEWED_UNWRAPPED_MLX_FUNCTIONS
            .iter()
            .find(|(reviewed_name, _)| reviewed_name == function_name)
        {
            Some((_, rationale)) => {
                for declaration in declarations.iter() {
                    println!(
                        "[audit]   reviewed: {function_name} ({}) — {rationale}",
                        declaration.header_relative_path
                    );
                }
            }
            None => {
                for declaration in declarations.iter() {
                    println!(
                        "[audit]   NEW GAP: {function_name} ({})",
                        declaration.header_relative_path
                    );
                }
            }
        }
    }
    if new_gap_count == 0 && redundant_shim_count == 0 {
        println!(
            "[audit] upstream leg: PASS (every MLX public function is wrapped \
             by MLX-C, wrapped by the gap shim, or reviewed)"
        );
        true
    } else {
        if new_gap_count > 0 {
            eprintln!(
                "[audit] upstream leg: FAIL — {new_gap_count} new unwrapped MLX \
                 functions; wrap them upstream in MLX-C, add a gap shim entry, \
                 or add a reviewed allowlist entry"
            );
        }
        if redundant_shim_count > 0 {
            eprintln!(
                "[audit] upstream leg: FAIL — {redundant_shim_count} redundant \
                 shim entries; retire them now that MLX-C wraps their functions"
            );
        }
        false
    }
}

#[cfg(test)]
#[path = "audit_upstream_surface_tests.rs"]
mod tests;
