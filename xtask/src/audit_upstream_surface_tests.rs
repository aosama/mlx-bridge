//! Contracts for the upstream surface audit's matching and header walking.
//! A sibling file keeps the audit module under the repository's per-file
//! line ceiling; it compiles only under `cfg(test)`.

use super::*;

#[test]
fn should_match_shim_entries_exactly_and_by_extension() {
    let shim_surface: BTreeSet<String> = [
        "mlxrs_matrix_transpose".to_owned(),
        "mlxrs_unique".to_owned(),
        "mlxrs_minmax_axes".to_owned(),
        "mlxrs_minmax_axis".to_owned(),
    ]
    .into_iter()
    .collect();
    // Exact shim names match their C++ function...
    assert_eq!(
        gap_shim_entry_for("matrix_transpose", "mlx/ops.h", &shim_surface),
        Some("mlxrs_matrix_transpose".to_owned())
    );
    // ...and a C++ overload whose C entry points extend the name (the
    // minmax axes/axis variants) matches through the extension rule.
    assert!(
        gap_shim_entry_for("minmax", "mlx/ops.h", &shim_surface)
            .is_some_and(|shim_symbol| shim_symbol.starts_with("mlxrs_minmax_"))
    );
    assert_eq!(
        gap_shim_entry_for("vmap", "mlx/transforms.h", &shim_surface),
        None
    );
}

#[test]
fn should_not_treat_a_separate_public_function_as_an_overload() {
    let shim_surface: BTreeSet<String> = ["mlxrs_einsum_path".to_owned()].into_iter().collect();

    assert_eq!(
        gap_shim_entry_for("einsum", "mlx/einsum.h", &shim_surface),
        None
    );
    assert_eq!(
        gap_shim_entry_for("einsum_path", "mlx/einsum.h", &shim_surface),
        Some("mlxrs_einsum_path".to_owned())
    );
}

#[test]
fn should_apply_family_prefixes_and_composition_fallback() {
    let vendored_surface: BTreeSet<String> = [
        "mlx_linalg_norm",
        "mlx_fft_fftn",
        "mlx_abs",
        "mlx_stream_new",
        "mlx_device_info_new",
        "mlx_fast_metal_kernel_new",
        "mlx_async_eval",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(
        candidate_prefixes_for_header("mlx/linalg.h"),
        &["mlx_linalg_", "mlx_"] as &[&str]
    );
    // Exact family names win first.
    assert!(vendored_surface.contains("mlx_linalg_norm"));
    // Composition matches reordered multi-word names...
    assert!(is_wrapped_by_composition("new_stream", &vendored_surface));
    assert!(is_wrapped_by_composition("device_info", &vendored_surface));
    assert!(is_wrapped_by_composition("metal_kernel", &vendored_surface));
    // ...but a single-word name must never fuzzy-match: eval would
    // otherwise silently hide behind mlx_async_eval.
    assert!(!is_wrapped_by_composition("eval", &vendored_surface));
    assert!(!is_wrapped_by_composition("unique", &vendored_surface));
}

#[test]
fn should_match_a_shim_entry_through_its_header_family() {
    let shim_surface: BTreeSet<String> = ["mlxrs_config_get".to_owned()].into_iter().collect();

    assert_eq!(
        gap_shim_entry_for("get", "mlx/config.h", &shim_surface),
        Some("mlxrs_config_get".to_owned())
    );
    // The same word declared in another family is a different function.
    assert_eq!(gap_shim_entry_for("get", "mlx/ops.h", &shim_surface), None);
}

#[test]
fn should_not_count_an_extended_shim_name_as_the_function_it_extends() {
    // The windowed attention entry serves a parameter, not the function MLX-C
    // already wraps; counting it would report a redundant shim entry and
    // retire a wrapper MLX-C does not have.
    let shim_surface: BTreeSet<String> =
        ["mlxrs_fast_scaled_dot_product_attention_windowed".to_owned()]
            .into_iter()
            .collect();

    assert_eq!(
        gap_shim_entry_for("scaled_dot_product_attention", "mlx/fast.h", &shim_surface),
        None
    );
}

#[test]
fn should_not_let_a_bare_c_name_cover_a_config_function() {
    // Negative: config's single-word names must reach Rust through the
    // config family, never through an unrelated top-level C function.
    assert_eq!(
        candidate_prefixes_for_header("mlx/config.h"),
        &["mlx_config_"] as &[&str]
    );
    let vendored_surface: BTreeSet<String> = ["mlx_get".to_owned(), "mlx_update".to_owned()]
        .into_iter()
        .collect();
    let covering_candidate = candidate_prefixes_for_header("mlx/config.h")
        .iter()
        .map(|prefix| format!("{prefix}get"))
        .find(|candidate| vendored_surface.contains(candidate));

    assert_eq!(covering_candidate, None);
    assert!(!is_wrapped_by_composition("get", &vendored_surface));
}

/// A synthetic MLX checkout: each entry is a header path and its source.
fn synthetic_mlx_checkout(headers: &[(&str, &str)]) -> std::path::PathBuf {
    let checkout_root = std::env::temp_dir().join(format!(
        "mlx-bridge-audit-walk-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the system clock should be past the Unix epoch")
            .as_nanos()
    ));
    for (header_relative_path, header_source) in headers {
        let header_path = checkout_root.join(header_relative_path);
        std::fs::create_dir_all(header_path.parent().expect("headers sit in directories"))
            .expect("the synthetic checkout should create");
        std::fs::write(&header_path, header_source).expect("the synthetic header should write");
    }
    checkout_root
}

#[test]
fn should_walk_headers_the_umbrella_reaches_only_indirectly() {
    let checkout_root = synthetic_mlx_checkout(&[
        ("mlx/mlx.h", "#include \"mlx/utils.h\"\n"),
        (
            "mlx/utils.h",
            "#include <vector>\n#include \"mlx/config.h\"\n#include \"mlx/utils.h\"\n",
        ),
        (
            "mlx/config.h",
            "MLX_API int get(const char* name, int default_val);\n",
        ),
    ]);

    let reached_headers = public_mlx_headers(&checkout_root);
    let public_functions = extract_mlx_public_functions(&checkout_root);

    assert_eq!(
        reached_headers,
        ["mlx/config.h".to_owned(), "mlx/utils.h".to_owned()]
            .into_iter()
            .collect()
    );
    assert!(public_functions.contains(&MlxPublicFunction {
        header_relative_path: "mlx/config.h".to_owned(),
        function_name: "get".to_owned(),
    }));
    let _ = std::fs::remove_dir_all(&checkout_root);
}

#[test]
fn should_skip_includes_outside_mlx_and_headers_missing_from_the_checkout() {
    // Boundary: third-party and missing headers end the walk quietly
    // instead of failing it or widening the audited surface.
    let checkout_root = synthetic_mlx_checkout(&[(
        "mlx/mlx.h",
        "#include \"metal_cpp/Metal.hpp\"\n#include \"mlx/absent.h\"\n",
    )]);

    let reached_headers = public_mlx_headers(&checkout_root);

    assert_eq!(
        reached_headers,
        ["mlx/absent.h".to_owned()].into_iter().collect()
    );
    assert!(extract_mlx_public_functions(&checkout_root).is_empty());
    let _ = std::fs::remove_dir_all(&checkout_root);
}

#[test]
fn should_reach_nothing_without_an_umbrella_header() {
    // Negative: a directory that is not an MLX checkout yields no surface,
    // which the upstream leg reports as a failure.
    let checkout_root = synthetic_mlx_checkout(&[("README.md", "not MLX\n")]);

    assert!(public_mlx_headers(&checkout_root).is_empty());
    let _ = std::fs::remove_dir_all(&checkout_root);
}
