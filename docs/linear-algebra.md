# Linear algebra

The `linalg_*` methods wrap MLX's linear algebra family.

## Use a CPU context

MLX implements matrix factorization, inversion, and solving only on the CPU, so these calls fail on a default GPU context. Create a context with `MlxBindingsContext::on_cpu`. Arrays don't need moving, because Apple silicon memory is unified.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let matrix = MlxArray::from_f32(&[2.0, 0.0, 0.0, 4.0], &[2, 2])?;

    let gpu = MlxBindingsContext::new()?;
    let failure = gpu.linalg_inv(&matrix).expect_err("inversion is CPU-only");
    assert!(failure.description.contains("GPU"));

    let cpu = MlxBindingsContext::on_cpu()?;
    assert_eq!(cpu.linalg_inv(&matrix)?.to_vec_f32()?, vec![0.5, 0.0, 0.0, 0.25]);
    Ok(())
}
```

Norms are the exception: `linalg_norm`, `linalg_norm_l2`, and `linalg_norm_matrix` run on the GPU too.

## Solving and factorizing

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::on_cpu()?;

    let coefficients = MlxArray::from_f32(&[2.0, 0.0, 0.0, 4.0], &[2, 2])?;
    let targets = MlxArray::from_f32(&[2.0, 8.0], &[2])?;
    assert_eq!(context.linalg_solve(&coefficients, &targets)?.to_vec_f32()?, vec![1.0, 2.0]);

    let symmetric = MlxArray::from_f32(&[4.0, 2.0, 2.0, 10.0], &[2, 2])?;
    let lower = context.linalg_cholesky(&symmetric, false)?;
    assert_eq!(lower.to_vec_f32()?, vec![2.0, 0.0, 1.0, 3.0]);
    let upper = context.linalg_cholesky(&symmetric, true)?;
    assert_eq!(upper.to_vec_f32()?, vec![2.0, 1.0, 0.0, 3.0]);

    let (orthogonal, triangular) = context.linalg_qr(&symmetric)?;
    assert_eq!(context.matmul(&orthogonal, &triangular)?.shape(), vec![2, 2]);
    Ok(())
}
```

## Singular matrices

A singular matrix builds a graph without complaint, and the failure arrives as an error when the graph runs:

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::on_cpu()?;
    let singular = MlxArray::from_f32(&[1.0, 2.0, 2.0, 4.0], &[2, 2])?;

    let inverse = context.linalg_inv(&singular)?;

    assert!(inverse.evaluate().is_err());
    Ok(())
}
```

## Singular values and eigenvalues

`linalg_svd` returns one array, the singular values, when `compute_uv` is `false`, and three when it is `true`. `linalg_eigvalsh` takes `"L"` or `"U"` to say which triangle to read.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::on_cpu()?;
    let matrix = MlxArray::from_f32(&[3.0, 0.0, 0.0, 2.0], &[2, 2])?;

    let singular_values = context.linalg_svd(&matrix, false)?;
    assert_eq!(singular_values.len(), 1);
    assert_eq!(singular_values.array_at(0, "read")?.to_vec_f32()?, vec![3.0, 2.0]);
    assert_eq!(context.linalg_svd(&matrix, true)?.len(), 3);

    let eigenvalues = context.linalg_eigvalsh(&matrix, "L")?;
    assert_eq!(eigenvalues.to_vec_f32()?, vec![2.0, 3.0]);
    Ok(())
}
```

## Norms

An empty axis list means the whole array. `linalg_norm` takes a numeric order, and `linalg_norm_matrix` takes a named one such as `"fro"`.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let vector = MlxArray::from_f32(&[3.0, 4.0], &[2])?;
    let matrix = MlxArray::from_f32(&[1.0, 2.0, 2.0, 4.0], &[2, 2])?;

    assert_eq!(context.linalg_norm_l2(&vector, &[], false)?.item_f32()?, 5.0);
    assert_eq!(context.linalg_norm(&vector, 1.0, &[], false)?.item_f32()?, 7.0);
    assert_eq!(context.linalg_norm_matrix(&matrix, "fro", &[], false)?.item_f32()?, 5.0);
    Ok(())
}
```

## Double precision

The GPU rejects `float64`, so double-precision work also belongs on a CPU context:

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::on_cpu()?;
    let values = MlxArray::from_f64(&[1.5, 2.5], &[2])?;

    assert_eq!(context.flip(&values)?.to_vec_f64()?, vec![2.5, 1.5]);
    Ok(())
}
```

Every method is listed in the [operations reference](reference/operations.md).
