//! Contracts for initializing with an all-gather callback that hold without
//! JACCL hardware; the trampoline's own contracts live beside it in the source.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::gpu_context;
use mlx_c_rust::{MlxCError, initialize_distributed_with_all_gather};

fn unused_all_gather(
    _rank: i32,
    _world_size: i32,
    _source: &[u8],
    _destination: &mut [u8],
) -> Result<(), MlxCError> {
    Ok(())
}

#[test]
fn should_refuse_a_backend_other_than_jaccl_with_a_translated_error() {
    gpu_context();

    let failure = initialize_distributed_with_all_gather(true, "not-a-backend", unused_all_gather)
        .expect_err("MLX only accepts the callback for JACCL");

    assert!(failure.operation.contains("all-gather"), "{failure}");
    assert!(!failure.description.is_empty());
}

#[test]
fn should_reject_a_backend_name_with_an_interior_null_byte() {
    gpu_context();

    assert!(initialize_distributed_with_all_gather(true, "a\0b", unused_all_gather).is_err());
}
