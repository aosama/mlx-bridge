# Distributed

MLX can run across cooperating processes. This crate wraps its group model and communication operations. The contract tests only exercise a single-process group, because multiple machines or a launcher are needed for more, so treat multi-process behavior as MLX's rather than something this crate has verified.

## Groups

`initialize_distributed` takes `strict` and a backend name: `"any"`, `"mpi"`, `"nccl"`, `"jaccl"`, or `"ring"`. With `strict` set to `false`, a process with no configured backend gets a singleton group, so the same code runs alone or distributed.

```rust,standalone_crate
use mlx_c_rust::{MlxCError, clear_distributed_backends, initialize_distributed, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let group = initialize_distributed(false, "any")?;
    assert_eq!(group.rank()?, 0);
    assert_eq!(group.size()?, 1);

    // An unknown name is refused rather than falling back.
    assert!(initialize_distributed(false, "not-a-backend").is_err());

    clear_distributed_backends()?;
    Ok(())
}
```

`MlxDistributedGroup::split` divides a group by color and key.

## Communication

These are methods on `MlxArray`, taking the group and a stream: `all_sum`, `all_gather`, `all_max`, `all_min`, `sum_scatter`, `send_to_rank`, `receive_from_rank`, and `receive_like`.

## JACCL with your own side channel

`initialize_distributed_with_all_gather` initializes JACCL with a Rust callback for exchanging metadata. It receives this process's rank, the world size, this rank's bytes, and a destination holding `world_size * source.len()` bytes ordered by rank. MLX then owns the callback.

```rust,no_run
use mlx_c_rust::{MlxCError, initialize_distributed_with_all_gather, install_non_terminating_error_handler};

fn exchange_with_peers(
    rank: i32,
    _world_size: i32,
    source: &[u8],
    destination: &mut [u8],
) -> Result<(), MlxCError> {
    // A real implementation sends `source` to every peer and collects theirs.
    let start = source.len() * usize::try_from(rank).unwrap_or(0);
    destination[start..start + source.len()].copy_from_slice(source);
    Ok(())
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let group = initialize_distributed_with_all_gather(true, "jaccl", exchange_with_peers)?;
    println!("rank {} of {}", group.rank()?, group.size()?);
    Ok(())
}
```

A cached JACCL group is returned without calling a new callback, so call `clear_distributed_backends` first to replace one. A failure or panic in the callback becomes an error instead of unwinding through C. This path has not been run end to end, because it needs JACCL hardware; its callback handling is tested directly.
