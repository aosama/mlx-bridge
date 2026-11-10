# Random numbers

MLX's random numbers are functional: output depends only on a key, so the same key always gives the same numbers.

## Keys

`random_key` builds a key from a seed. Reusing a key repeats the sample, so split a key whenever you need independent randomness.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let low = MlxArray::from_scalar_f32(0.0);
    let high = MlxArray::from_scalar_f32(1.0);

    let key = context.random_key(42)?;
    let first = context.random_uniform(&low, &high, &[4], MlxDtype::Float32, Some(&key))?;
    let repeat = context.random_uniform(&low, &high, &[4], MlxDtype::Float32, Some(&key))?;
    assert_eq!(first.to_vec_f32()?, repeat.to_vec_f32()?);

    let (left_key, right_key) = context.split_random_key(&key)?;
    let left = context.random_uniform(&low, &high, &[4], MlxDtype::Float32, Some(&left_key))?;
    let right = context.random_uniform(&low, &high, &[4], MlxDtype::Float32, Some(&right_key))?;
    assert_ne!(left.to_vec_f32()?, right.to_vec_f32()?);

    // Many keys at once come back as one array, one key per row.
    assert_eq!(context.random_split_num(&key, 4)?.shape(), vec![4, 2]);
    Ok(())
}
```

## The global generator

Passing `None` as the key uses a process-wide generator, whose state advances with each call. `seed_random_generation` resets it. Prefer explicit keys in libraries, because global state couples unrelated code.

```rust,standalone_crate
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn sample(context: &MlxBindingsContext) -> Result<Vec<f32>, MlxCError> {
    let low = MlxArray::from_scalar_f32(0.0);
    let high = MlxArray::from_scalar_f32(1.0);
    context.random_uniform(&low, &high, &[4], MlxDtype::Float32, None)?.to_vec_f32()
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;

    context.seed_random_generation(7)?;
    let first = sample(&context)?;
    let following = sample(&context)?;
    assert_ne!(first, following);

    context.seed_random_generation(7)?;
    assert_eq!(first, sample(&context)?);
    Ok(())
}
```

## Distributions

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let key = context.random_key(1)?;

    let normal = context.random_normal(&[2], MlxDtype::Float32, 5.0, 0.0, &key)?;
    assert_eq!(normal.to_vec_f32()?, vec![5.0, 5.0], "zero deviation leaves only the mean");

    let integers = context.random_randint(
        &MlxArray::from_scalar_i32(3),
        &MlxArray::from_scalar_i32(6),
        &[32],
        MlxDtype::Int32,
        Some(&key),
    )?;
    for sampled_integer in integers.to_vec_i32()? {
        assert!((3..6).contains(&sampled_integer));
    }

    let always = context.random_bernoulli(&MlxArray::from_scalar_f32(1.0), &[2], Some(&key))?;
    assert_eq!(always.to_vec_bool()?, vec![true, true]);

    assert_eq!(context.random_permutation_arange(8, Some(&key))?.shape(), vec![8]);

    let logits = MlxArray::from_f32(&[-100.0, 100.0, -100.0], &[3])?;
    let category = context.categorical_sample_with_key(&logits, 0, &key)?;
    assert_eq!(category.to_vec_u32()?, vec![1]);
    Ok(())
}
```

Also available: `random_gumbel`, `random_laplace`, `random_truncated_normal`, `random_multivariate_normal`, `random_bits`, `random_normal_broadcast`, `random_permutation`, and `random_categorical_shape`. See the [operations reference](reference/operations.md).
