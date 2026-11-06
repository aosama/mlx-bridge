# Custom Metal kernels

`MlxMetalKernel` runs your own Metal shader on arrays. You write the function body, and MLX generates the signature from your argument names and compiles the shader when the graph runs.

## A complete kernel

Inputs and outputs appear in the shader under the names you give, and `thread_position_in_grid` is available as usual. Declare each output's shape and element type, then choose a grid and thread group.

```rust
use mlx_c_rust::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, install_non_terminating_error_handler};

const DOUBLING_SHADER: &str = "
    uint element_position = thread_position_in_grid.x;
    output_values[element_position] = input_values[element_position] * 2.0;
";

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let kernel = MlxMetalKernel::new("docs_doubling", &["input_values"], &["output_values"], DOUBLING_SHADER)?;
    let input = MlxArray::from_f32(&[1.0, 2.0, 3.0], &[3])?;
    let output_declarations = vec![MlxMetalKernelOutput::new(vec![3], MlxDtype::Float32)];

    let outputs = context.apply_metal_kernel(&kernel, &[&input], &output_declarations, [3, 1, 1], [1, 1, 1], &[])?;

    assert_eq!(outputs[0].to_vec_f32()?, vec![2.0, 4.0, 6.0]);
    Ok(())
}
```

## Template arguments

Template arguments reach the shader as compile-time values. Each is a dtype, an integer, or a boolean.

```rust
use mlx_c_rust::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument, install_non_terminating_error_handler};

const SCALING_SHADER: &str = "
    uint element_position = thread_position_in_grid.x;
    output_values[element_position] = element_type(scale) * input_values[element_position];
";

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let kernel = MlxMetalKernel::new("docs_scaling", &["input_values"], &["output_values"], SCALING_SHADER)?;
    let input = MlxArray::from_f32(&[1.0, 2.0], &[2])?;
    let output_declarations = vec![MlxMetalKernelOutput::new(vec![2], MlxDtype::Float32)];
    let template_arguments = [
        MlxMetalKernelTemplateArgument::Dtype { name: "element_type", dtype: MlxDtype::Float32 },
        MlxMetalKernelTemplateArgument::Integer { name: "scale", integer_template_argument: 3 },
    ];

    let outputs = context.apply_metal_kernel(&kernel, &[&input], &output_declarations, [2, 1, 1], [1, 1, 1], &template_arguments)?;

    assert_eq!(outputs[0].to_vec_f32()?, vec![3.0, 6.0]);
    Ok(())
}
```

## Other options

- `new_with_header` places code before the generated function, for helpers and includes.
- `new_with_options` also declares outputs `atomic`, which a kernel needs to write them safely from many threads.
- `apply_metal_kernel_with_output_initialization` fills outputs first, so a shader can leave elements unwritten.
- `apply_metal_kernel_in_graph` is for use inside graph builders.

## Errors arrive in two stages

The crate rejects a kernel with no inputs or outputs, a null byte in any string, or a grid or thread group with a non-positive dimension straight away. A shader that doesn't compile only fails when the output is evaluated, because compilation happens then.

```rust
use mlx_c_rust::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let kernel = MlxMetalKernel::new("docs_broken", &["input_values"], &["output_values"], "not metal shading language;")?;
    let input = MlxArray::from_f32(&[1.0], &[1])?;
    let output_declarations = vec![MlxMetalKernelOutput::new(vec![1], MlxDtype::Float32)];

    let outputs = context.apply_metal_kernel(&kernel, &[&input], &output_declarations, [1, 1, 1], [1, 1, 1], &[])?;

    let failure = outputs[0].evaluate().expect_err("the shader fails when the graph runs");
    assert!(failure.operation.contains("evaluate"));
    Ok(())
}
```

## Capturing GPU work

`start_metal_capture` and `stop_metal_capture` record a trace you can open in Xcode. Metal only captures when the process started with `MTL_CAPTURE_ENABLED=1`; otherwise starting returns an error.

## Metal library location

`metallib_path` and `set_metallib_path` read and override where MLX loads its kernels from. The automatic native build bakes in the right path, so most programs never need this. Override it before the first GPU use, which includes creating a default context, because kernels load once.
