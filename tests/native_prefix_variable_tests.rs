//! Contracts for the notice a build receives when it still selects its
//! MLX-C prefix through the variable name the crate used as `mlx-c-rust`.
//! The module is a source file shared with the build script, so these tests
//! include it directly, the way the digest contracts include the engine.

#[path = "../native_prefix_variable.rs"]
mod native_prefix_variable;

use native_prefix_variable::{
    FORMER_MLX_C_PREFIX_VARIABLE, MLX_C_PREFIX_VARIABLE, former_prefix_variable_notice,
};

#[test]
fn should_warn_and_name_both_variables_when_only_the_former_one_is_set() {
    let prefix_variable_notice = former_prefix_variable_notice(false, true)
        .expect("a build relying on the former variable must be told");

    // The reader needs both names: the one to remove and the one to set.
    assert!(prefix_variable_notice.contains(FORMER_MLX_C_PREFIX_VARIABLE));
    assert!(prefix_variable_notice.contains(MLX_C_PREFIX_VARIABLE));
}

#[test]
fn should_stay_quiet_when_neither_variable_is_set() {
    // The zero-configuration build is the common case and must not gain a
    // warning from the rename.
    assert_eq!(former_prefix_variable_notice(false, false), None);
}

#[test]
fn should_stay_quiet_when_only_the_current_variable_is_set() {
    assert_eq!(former_prefix_variable_notice(true, false), None);
}

#[test]
fn should_stay_quiet_when_both_variables_are_set() {
    // Boundary: a build mid-migration exports both names. The current one
    // already steers the ladder, so the leftover former one is harmless.
    assert_eq!(former_prefix_variable_notice(true, true), None);
}

#[test]
fn should_keep_the_former_name_distinct_from_the_current_name() {
    // If the two constants ever converged, the warning would fire for every
    // build that sets the prefix correctly.
    assert_ne!(FORMER_MLX_C_PREFIX_VARIABLE, MLX_C_PREFIX_VARIABLE);
    assert_eq!(FORMER_MLX_C_PREFIX_VARIABLE, "MLX_C_RUST_MLX_C_PREFIX");
}
