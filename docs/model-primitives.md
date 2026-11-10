# Model primitives

This crate has no layers or optimizers, but it exposes MLX's fast primitives that models are built from. For a high-level framework, see [mlx-rs](https://github.com/oxiglade/mlx-rs); the two can be used side by side.

## Scaled dot-product attention

Inputs are shaped `(batch, heads, sequence, dimensions)`. The three entry points differ in masking.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let query = MlxArray::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let keys = MlxArray::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2])?;
    let values = MlxArray::from_f32(&[2.0, 2.0, 4.0, 6.0], &[1, 1, 2, 2])?;

    let attended = context.scaled_dot_product_attention(&query, &keys, &values, 1.0)?;
    assert_eq!(attended.shape(), vec![1, 1, 1, 2]);

    // A boolean mask hides keys: here, everything but the second.
    let keep_second = context.astype(&MlxArray::from_i32(&[0, 1], &[1, 1, 1, 2])?, MlxDtype::Bool)?;
    let masked = context.masked_scaled_dot_product_attention(&query, &keys, &values, 1.0, &keep_second)?;
    assert_eq!(masked.to_vec_f32()?, vec![4.0, 6.0]);

    // The causal entry hides later keys from earlier queries.
    let causal = context.causal_scaled_dot_product_attention(&keys, &keys, &values, 1.0)?;
    assert_eq!(causal.to_vec_f32()?[..2], [2.0, 2.0]);
    Ok(())
}
```

## Rotary position embeddings

`rope` takes the input, rotated dimensions, base, and the token offset to start from. An offset lets you continue a sequence during incremental decoding. The other `rope_with_*` entry points take per-token positions or custom frequencies.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let token = MlxArray::from_f32(&[1.0, 0.0], &[1, 1, 2])?;
    let two_tokens = MlxArray::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 2])?;

    let from_offset = context.rope(&token, 2, 10_000.0, 1)?.to_vec_f32()?;
    let second_token = context.rope(&two_tokens, 2, 10_000.0, 0)?.to_vec_f32()?;

    assert!((from_offset[0] - second_token[2]).abs() < 1e-5);
    assert!((from_offset[1] - second_token[3]).abs() < 1e-5);
    Ok(())
}
```

## Convolution

Inputs are channels-last: `(batch, width, channels)`. MLX orders weights `(output_channels, width, input_channels)`, which is an easy thing to get backwards. `conv2d`, `conv3d`, `conv_general`, and the `conv_transpose*` family follow the same pattern.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let signal = MlxArray::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1])?;
    let adjacent_sum = MlxArray::from_f32(&[1.0, 1.0], &[1, 2, 1])?;

    // Arguments after the weight: stride, padding, dilation, groups.
    let convolved = context.conv1d(&signal, &adjacent_sum, 1, 0, 1, 1)?;

    assert_eq!(convolved.shape(), vec![1, 3, 1]);
    assert_eq!(convolved.to_vec_f32()?, vec![3.0, 5.0, 7.0]);
    Ok(())
}
```

## Quantization

`quantize_affine` returns the packed weights, scales, and biases, and `dequantize_affine` restores them approximately. MLX accepts group sizes of 32, 64, and 128. Quantized arrays pack `32 / bits` elements into each 32-bit word.

```rust
use mlx_bridge::{MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let ramp = context.multiply_scalar(&context.arange_f32(0.0, 256.0, 1.0)?, 0.01)?;
    let weights = context.reshape(&ramp, &[4, 64])?;

    let (quantized, scales, biases) = context.quantize_affine(&weights, 32, 8)?;
    assert_eq!(quantized.shape(), vec![4, 16]);
    assert_eq!(scales.shape(), vec![4, 2]);

    let restored = context.dequantize_affine(&quantized, &scales, &biases, 32, 8)?;
    assert_eq!(restored.shape(), weights.shape());
    Ok(())
}
```

Related entry points are `quantized_matmul_affine`, `gather_quantized_matmul_affine`, `qqmm`, `gather_qqmm`, and the NVFP4 family: `quantize_nvfp4`, `dequantize_nvfp4`, and `quantized_matmul_nvfp4`.

## Fast primitives

`rms_norm` and `layer_norm` are fused normalizations. `fast_cross_entropy` computes cross entropy from logits and class targets. `gated_delta_update` is the Qwen-style gated-delta recurrence, returning outputs first and final state second.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;

    let logits = MlxArray::from_f32(&[0.0, 0.0], &[1, 2])?;
    let target = MlxArray::from_i32(&[1], &[1])?;
    let loss = context.fast_cross_entropy(&logits, &target)?.to_vec_f32()?;
    assert!((loss[0] - std::f32::consts::LN_2).abs() < 1e-5);

    let queries = context.full(&[1, 3, 1, 4], 0.5, MlxDtype::Float32)?;
    let keys = context.full(&[1, 3, 1, 4], 0.5, MlxDtype::Float32)?;
    let values = context.full(&[1, 3, 2, 5], 1.0, MlxDtype::Float32)?;
    let gates = context.full(&[1, 3, 2], 0.5, MlxDtype::Float32)?;
    let beta = context.full(&[1, 3, 2], 0.5, MlxDtype::Float32)?;

    let outputs = context.gated_delta_update(&queries, &keys, &values, &gates, &beta, None, None)?;
    assert_eq!(outputs.array_at(0, "read the outputs")?.shape(), vec![1, 3, 2, 5]);
    assert_eq!(outputs.array_at(1, "read the state")?.shape(), vec![1, 2, 5, 4]);
    Ok(())
}
```

## Prebuilt compiled graphs

Compiling a repeated composite fuses its dispatches into fewer kernel launches. The crate ships several for common patterns, each compiled once and replayed across calls. Create the handle once and reuse it.

| Handle | Replays |
|---|---|
| `MlxCompiledSwiGlu` | `silu(gate) * input` |
| `MlxCompiledDualHeadNorm` | Two RMS norms in one launch |
| `MlxCompiledElementwiseGraphs` | Attention gates, SiLU, signed casts, vision RoPE, gated-delta decay, and more |
| `MlxCompiledGatedDeltaRateActivation` | Update rates and decay factors together |
| `MlxCompiledMultiOutputGraph` | Your own graph-builder function |

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxCompiledSwiGlu, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let compiled = MlxCompiledSwiGlu::new()?;
    let gate = MlxArray::from_f32(&[-1.0, 0.0, 2.0], &[3])?;
    let input = MlxArray::from_f32(&[3.0, 4.0, 5.0], &[3])?;

    let replayed = context.apply_compiled_swiglu(&compiled, &gate, &input)?.to_vec_f32()?;
    let eager = context.multiply(&context.silu(&gate)?, &input)?.to_vec_f32()?;

    for value_index in 0..replayed.len() {
        assert!((replayed[value_index] - eager[value_index]).abs() < 1e-5);
    }
    Ok(())
}
```

One naming trap: `apply_compiled_softplus` computes `log1p(exp(x * ln2)) / ln2`, a beta-scaled softplus, not the standard `softplus`, and `apply_compiled_attention_output_gate` uses a sigmoid gate, unlike `apply_softplus_attention_gate`. The [compile reference](reference/compile-and-transforms.md) documents each formula.
