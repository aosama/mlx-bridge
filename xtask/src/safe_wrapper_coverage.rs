//! The safe-wrapper leg: every vendored MLX-C function must be reached by the
//! crate's safe code, or carry a written reason why a programmer never needs it.
//! This is what keeps "no reaching into `raw`" a checked property instead of a
//! claim that erodes with each pin bump.

use std::collections::BTreeSet;
use std::path::Path;

use crate::repository::collect_rust_sources;
use crate::rust_item_parsing::mentions_whole_word;

/// MLX-C functions no safe code calls, each with why nobody needs them directly.
const FUNCTIONS_WITHOUT_SAFE_WRAPPER: &[(&str, &str)] = &[
    (
        "mlx_array_new_data_managed",
        "MlxArray::from_managed_buffer owns its buffer through the payload variant",
    ),
    (
        "mlx_array_new_double",
        "alias of mlx_array_new_float64, served by from_scalar_f64",
    ),
    (
        "mlx_array_new_float",
        "alias of mlx_array_new_float32, served by from_scalar_f32",
    ),
    (
        "mlx_array_set_data",
        "replacing a handle's contents is assigning a new MlxArray built from the data",
    ),
    (
        "mlx_array_set_double",
        "alias of mlx_array_set_float64, served by set_f64",
    ),
    (
        "mlx_array_set_float",
        "alias of mlx_array_set_float32, served by set_f32",
    ),
    (
        "mlx_closure_custom_jvp_new_func",
        "payload-free C callback form; MlxClosureCustomJvp::from_fn accepts any named function",
    ),
    (
        "mlx_closure_custom_new_func",
        "payload-free C callback form; MlxClosureCustom::from_fn accepts any named function",
    ),
    (
        "mlx_closure_custom_vmap_new_func",
        "payload-free C callback form; MlxClosureCustomVmap::from_fn accepts any named function",
    ),
    (
        "mlx_closure_kwargs_new_func",
        "payload-free C callback form; MlxClosureKwargs::from_fn accepts any named function",
    ),
    (
        "mlx_closure_new_unary",
        "single-array C callback form; MlxClosure::from_fn takes slices of any length",
    ),
    (
        "mlx_closure_value_and_grad_new_func",
        "payload-free C callback form; MlxClosureValueAndGrad::from_fn accepts any named function",
    ),
    (
        "mlx_compile_cache_new",
        "empty placeholder; MlxCompileCache::empty obtains the live process cache instead",
    ),
    (
        "mlx_stream_new",
        "empty placeholder for an output parameter; every safe path returns a populated stream",
    ),
    (
        "mlx_vector_stream_append_data",
        "served by MlxVectorStream::append_streams",
    ),
    (
        "mlx_vector_stream_new_value",
        "served by MlxVectorStream::from_stream",
    ),
    (
        "mlx_vector_stream_set_data",
        "served by MlxVectorStream::set_streams",
    ),
    (
        "mlx_vector_stream_set_value",
        "served by MlxVectorStream::set_stream",
    ),
    (
        "mlx_vector_string_append_data",
        "served by MlxVectorString::append_strings",
    ),
    (
        "mlx_vector_string_new_value",
        "served by MlxVectorString::from_string",
    ),
    (
        "mlx_vector_string_set_data",
        "served by MlxVectorString::set_strings",
    ),
    (
        "mlx_vector_vector_array_append_data",
        "served by MlxVectorVectorArray::append_vectors",
    ),
    (
        "mlx_vector_vector_array_new_value",
        "served by MlxVectorVectorArray::from_vector",
    ),
    (
        "mlx_vector_vector_array_set_data",
        "served by MlxVectorVectorArray::set_vectors",
    ),
    (
        "mlx_vector_vector_array_set_value",
        "served by MlxVectorVectorArray::set_vector",
    ),
];

/// Reads every Rust source under `directory` into one searchable text.
fn combined_rust_source(directory: &Path) -> std::io::Result<String> {
    let mut source_paths = Vec::new();
    collect_rust_sources(directory, &mut source_paths)?;
    let mut combined_source = String::new();
    for source_path in &source_paths {
        combined_source.push_str(&std::fs::read_to_string(source_path)?);
        combined_source.push('\n');
    }
    Ok(combined_source)
}

/// The vendored functions neither called nor reasoned about, and the registry
/// entries that no longer need their reason.
fn coverage_problems(
    vendored_functions: &BTreeSet<String>,
    safe_source: &str,
    registry: &[(&str, &str)],
) -> (Vec<String>, Vec<String>) {
    let mut registered_names = BTreeSet::new();
    for (registered_name, _reason) in registry {
        registered_names.insert((*registered_name).to_owned());
    }
    let mut unreachable_functions = Vec::new();
    for function_name in vendored_functions {
        if !registered_names.contains(function_name)
            && !mentions_whole_word(safe_source, function_name)
        {
            unreachable_functions.push(function_name.clone());
        }
    }
    let mut stale_entries = Vec::new();
    for registered_name in &registered_names {
        if !vendored_functions.contains(registered_name)
            || mentions_whole_word(safe_source, registered_name)
        {
            stale_entries.push(registered_name.clone());
        }
    }
    (unreachable_functions, stale_entries)
}

/// Fails when a vendored function has no safe path and no recorded reason, or
/// when a recorded reason has gone stale.
pub(crate) fn audit_safe_wrapper_coverage(
    repository_root: &Path,
    vendored_functions: &BTreeSet<String>,
) -> bool {
    let source_root = repository_root.join("src");
    let safe_source = match combined_rust_source(&source_root) {
        Ok(safe_source) => safe_source,
        Err(read_error) => {
            eprintln!(
                "[audit] cannot read {}: {read_error}",
                source_root.display()
            );
            return false;
        }
    };
    let (unreachable_functions, stale_entries) = coverage_problems(
        vendored_functions,
        &safe_source,
        FUNCTIONS_WITHOUT_SAFE_WRAPPER,
    );
    for function_name in &unreachable_functions {
        eprintln!("[audit]   NO SAFE PATH: {function_name}");
    }
    for registered_name in &stale_entries {
        eprintln!("[audit]   STALE REASON: {registered_name} no longer needs its registry entry");
    }
    println!(
        "[audit] vendored functions without a safe wrapper: {} (all reasoned)",
        FUNCTIONS_WITHOUT_SAFE_WRAPPER.len()
    );
    if unreachable_functions.is_empty() && stale_entries.is_empty() {
        println!("[audit] safe wrapper leg: PASS");
        return true;
    }
    eprintln!("[audit] safe wrapper leg: FAIL — wrap the function, or record why nobody needs it");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendored(function_names: &[&str]) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for function_name in function_names {
            names.insert((*function_name).to_owned());
        }
        names
    }

    #[test]
    fn should_pass_when_every_function_is_called_or_reasoned() {
        let functions = vendored(&["mlx_add", "mlx_alias"]);

        let (unreachable, stale) =
            coverage_problems(&functions, "raw::mlx_add(x)", &[("mlx_alias", "an alias")]);

        assert!(unreachable.is_empty());
        assert!(stale.is_empty());
    }

    #[test]
    fn should_flag_a_function_with_no_safe_path() {
        let functions = vendored(&["mlx_add", "mlx_new_thing"]);

        let (unreachable, _stale) = coverage_problems(&functions, "raw::mlx_add(x)", &[]);

        assert_eq!(unreachable, vec!["mlx_new_thing"]);
    }

    #[test]
    fn should_not_count_a_longer_identifier_as_a_mention() {
        let functions = vendored(&["mlx_add"]);

        let (unreachable, _stale) = coverage_problems(&functions, "raw::mlx_addmm(x)", &[]);

        assert_eq!(unreachable, vec!["mlx_add"]);
    }

    #[test]
    fn should_flag_a_reason_whose_function_is_now_called() {
        let functions = vendored(&["mlx_add"]);

        let (_unreachable, stale) = coverage_problems(
            &functions,
            "raw::mlx_add(x)",
            &[("mlx_add", "no longer true")],
        );

        assert_eq!(stale, vec!["mlx_add"]);
    }

    #[test]
    fn should_flag_a_reason_whose_function_left_the_headers() {
        let functions = vendored(&[]);

        let (_unreachable, stale) = coverage_problems(&functions, "", &[("mlx_gone", "removed")]);

        assert_eq!(stale, vec!["mlx_gone"]);
    }
}
