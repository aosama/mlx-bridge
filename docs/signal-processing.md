# Signal processing

This guide covers the FFT family and the short-time transforms. Results are complex unless a real inverse is requested, so read them with `to_vec_complex`, or take parts with `real` and `imag`.

## FFT conventions

Each transform takes a length `n`, an axis, and a normalization from `raw`:

| Constant | Meaning |
|---|---|
| `raw::mlx_fft_norm__MLX_FFT_NORM_BACKWARD` | No scaling forward, `1/n` on the inverse (the usual default) |
| `raw::mlx_fft_norm__MLX_FFT_NORM_ORTHO` | `1/sqrt(n)` both ways |
| `raw::mlx_fft_norm__MLX_FFT_NORM_FORWARD` | `1/n` forward, none on the inverse |

## Real transforms

`fft_rfft` keeps only the non-negative frequencies, so `n` real samples give `n / 2 + 1` bins. `fft_irfft` inverts it, and needs the original length.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler, raw};

const BACKWARD: raw::mlx_fft_norm = raw::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let signal = MlxArray::from_f32(&[1.0, 1.0, 1.0, 1.0], &[4])?;

    let spectrum = context.fft_rfft(&signal, 4, 0, BACKWARD)?;
    assert_eq!(spectrum.dtype(), MlxDtype::Complex64);
    assert_eq!(spectrum.shape(), vec![3]);

    // A constant signal puts all its energy in the first bin.
    let first_bin = spectrum.to_vec_complex()?[0];
    assert!((first_bin.0 - 4.0).abs() < 1e-5);

    let recovered = context.fft_irfft(&spectrum, 4, 0, BACKWARD)?;
    assert_eq!(recovered.shape(), vec![4]);
    Ok(())
}
```

## Other transforms

- `fft_fft` and `fft_ifft` are the complex forms.
- `fft_fft2`, `fft_ifft2`, `fft_fftn`, `fft_ifftn`, and the `rfft` counterparts take a list of lengths and axes.
- `fft_fftfreq(n, d)` and `fft_rfftfreq(n, d)` give bin frequencies for sample spacing `d`.
- `fft_fftshift` and `fft_ifftshift` move the zero-frequency bin to the center and back.

## Short-time transforms

`stft` analyzes the last axis in overlapping frames, and returns axes ordered `(bins, frames)`. `istft` inverts it.

| `stft` argument | Meaning | When `None` |
|---|---|---|
| `n_fft` | Transform length | Required |
| `hop_length` | Samples between frames | MLX's default |
| `win_length` | Window length | MLX's default |
| `window` | Window samples | MLX's default |
| `center` | Pad so frames center on samples | Required |
| `pad_mode` | Padding for `center`, such as `"reflect"` | Required |
| `norm` | Normalization | Required |
| `onesided` | Keep only non-negative frequencies | Chosen from the input type |

```rust
use mlx_bridge::{MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler, raw};

const ORTHO: raw::mlx_fft_norm = raw::mlx_fft_norm__MLX_FFT_NORM_ORTHO;

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let signal = context.arange_f32(0.0, 8.0, 1.0)?;

    let frames = context.stft(&signal, 4, Some(2), None, None, true, "reflect", ORTHO, None)?;
    assert_eq!(frames.dtype(), MlxDtype::Complex64);

    // length restores exactly the original sample count.
    let recovered = context.istft(&frames, Some(4), Some(2), None, None, true, ORTHO, None, Some(8))?;
    assert_eq!(recovered.shape(), vec![8]);

    let original = signal.to_vec_f32()?;
    let round_trip = recovered.to_vec_f32()?;
    for sample_index in 0..original.len() {
        assert!((round_trip[sample_index] - original[sample_index]).abs() < 1e-3);
    }
    Ok(())
}
```

Every method is listed in the [operations reference](reference/operations.md).
