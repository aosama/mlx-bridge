//! The explicit MLX-C prefix override variable: its current name, the name
//! it carried while the crate was published as `mlx-c-rust`, and the notice
//! for builds that still set only the former name. A source module shared
//! by the build script and the test tree, like the native build engine.

/// The variable the link ladder reads first: a prefix whose `lib/` holds the
/// MLX-C library the consumer wants linked.
pub const MLX_C_PREFIX_VARIABLE: &str = "MLX_BRIDGE_MLX_C_PREFIX";

/// The same variable under the crate's former name. It is read only to
/// decide whether to warn and is never honored as a prefix: honoring it
/// would keep two authoritative names alive indefinitely, while ignoring it
/// silently would link the stock pair into a build that asked for its own
/// MLX-C, for example a patched fork.
pub const FORMER_MLX_C_PREFIX_VARIABLE: &str = "MLX_C_RUST_MLX_C_PREFIX";

/// The warning for a build that still selects its prefix through the former
/// variable name, or `None` when there is nothing to say. A build that sets
/// the current name has already migrated, so a leftover former name beside
/// it stays quiet.
pub fn former_prefix_variable_notice(
    is_current_variable_set: bool,
    is_former_variable_set: bool,
) -> Option<String> {
    if is_current_variable_set || !is_former_variable_set {
        return None;
    }
    Some(format!(
        "{FORMER_MLX_C_PREFIX_VARIABLE} is set but no longer read since the crate was \
         renamed to mlx-bridge; set {MLX_C_PREFIX_VARIABLE} to keep linking your own \
         MLX-C. Continuing down the discovery ladder without that prefix"
    ))
}
