//! Contracts for MLX's configuration values: a value set in Rust reaches
//! MLX's table under the name given, lookups resolve in MLX's documented
//! order, and scopes stay on their thread and leave when dropped. Names here
//! carry a test prefix so no MLX tunable changes; every test runs in its own
//! process, so values never leak between contracts.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::gpu_context;
use mlx_bridge::{MlxConfigScope, config_get, config_update};

#[test]
fn should_read_back_a_value_set_for_the_process() {
    gpu_context();

    config_update("MLX_BRIDGE_TEST_PROCESS_VALUE", 41).unwrap();

    assert_eq!(config_get("MLX_BRIDGE_TEST_PROCESS_VALUE", 7).unwrap(), 41);
}

#[test]
fn should_return_the_default_for_a_name_nothing_sets() {
    // Boundary: no update, no scope, and no environment variable.
    gpu_context();

    assert_eq!(config_get("MLX_BRIDGE_TEST_UNSET_NAME", -3).unwrap(), -3);
}

#[test]
fn should_keep_the_first_answer_for_a_name_across_defaults() {
    // MLX records the first read's answer as the process value, so a second
    // read with another default returns the first answer. Callers relying
    // on a default per read would be surprised; this pins the behavior.
    gpu_context();

    let first_answer = config_get("MLX_BRIDGE_TEST_RECORDED_NAME", 5).unwrap();
    let second_answer = config_get("MLX_BRIDGE_TEST_RECORDED_NAME", 9).unwrap();

    assert_eq!(first_answer, 5);
    assert_eq!(second_answer, 5);
}

#[test]
fn should_reject_a_name_with_an_interior_null_byte() {
    gpu_context();

    assert!(config_update("MLX_BRIDGE\0TEST", 1).is_err());
    assert!(config_get("MLX_BRIDGE\0TEST", 1).is_err());
    assert!(MlxConfigScope::new(&[("MLX_BRIDGE\0TEST", 1)]).is_err());
}

#[test]
fn should_apply_scoped_values_over_process_values_until_the_scope_drops() {
    gpu_context();
    config_update("MLX_BRIDGE_TEST_SCOPED_NAME", 1).unwrap();

    let scope = MlxConfigScope::new(&[("MLX_BRIDGE_TEST_SCOPED_NAME", 2)]).unwrap();
    let inside_scope = config_get("MLX_BRIDGE_TEST_SCOPED_NAME", 0).unwrap();
    drop(scope);
    let after_scope = config_get("MLX_BRIDGE_TEST_SCOPED_NAME", 0).unwrap();

    assert_eq!(inside_scope, 2);
    assert_eq!(after_scope, 1);
}

#[test]
fn should_let_the_newest_scope_win_and_survive_its_elder_dropping_first() {
    // Boundary: scopes need not drop in reverse order; dropping the outer one
    // first must leave the inner one in force.
    gpu_context();
    let outer_scope = MlxConfigScope::new(&[("MLX_BRIDGE_TEST_NESTED_NAME", 10)]).unwrap();
    let inner_scope = MlxConfigScope::new(&[("MLX_BRIDGE_TEST_NESTED_NAME", 20)]).unwrap();

    let with_both = config_get("MLX_BRIDGE_TEST_NESTED_NAME", 0).unwrap();
    drop(outer_scope);
    let with_inner_only = config_get("MLX_BRIDGE_TEST_NESTED_NAME", 0).unwrap();
    drop(inner_scope);

    assert_eq!(with_both, 20);
    assert_eq!(with_inner_only, 20);
}

#[test]
fn should_take_the_last_value_for_a_name_repeated_in_one_scope() {
    gpu_context();

    let _scope = MlxConfigScope::new(&[
        ("MLX_BRIDGE_TEST_REPEATED_NAME", 3),
        ("MLX_BRIDGE_TEST_REPEATED_NAME", 4),
    ])
    .unwrap();

    assert_eq!(config_get("MLX_BRIDGE_TEST_REPEATED_NAME", 0).unwrap(), 4);
}

#[test]
fn should_not_apply_a_scope_on_another_thread() {
    // Scopes are MLX thread-local state, which is why the guard is not Send.
    gpu_context();
    let _scope = MlxConfigScope::new(&[("MLX_BRIDGE_TEST_THREAD_NAME", 8)]).unwrap();

    let other_thread_answer =
        std::thread::spawn(|| config_get("MLX_BRIDGE_TEST_THREAD_NAME", -1).unwrap())
            .join()
            .unwrap();

    assert_eq!(other_thread_answer, -1);
    assert_eq!(config_get("MLX_BRIDGE_TEST_THREAD_NAME", -1).unwrap(), 8);
}

#[test]
fn should_accept_an_empty_scope() {
    // Boundary: a scope with no values is valid and changes nothing.
    gpu_context();

    let _scope = MlxConfigScope::new(&[]).unwrap();

    assert_eq!(config_get("MLX_BRIDGE_TEST_EMPTY_SCOPE", 6).unwrap(), 6);
}
