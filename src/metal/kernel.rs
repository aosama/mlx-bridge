use std::ffi::{CString, NulError};

use crate::MlxBindingsContext;
use crate::{MlxArray, MlxArrayVector, MlxCError, MlxDtype, MlxStream, error::check_status, raw};

/// Output shape and dtype requested from one custom Metal kernel launch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxMetalKernelOutput {
    shape: Vec<i32>,
    dtype: MlxDtype,
}

impl MlxMetalKernelOutput {
    /// Declares one output the kernel writes, with its shape and element type.
    #[must_use]
    pub const fn new(shape: Vec<i32>, dtype: MlxDtype) -> Self {
        Self { shape, dtype }
    }
}

/// Compile-time template argument passed to an MLX custom Metal kernel launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MlxMetalKernelTemplateArgument {
    Dtype {
        name: &'static str,
        dtype: MlxDtype,
    },
    Integer {
        name: &'static str,
        integer_template_argument: i32,
    },
    Boolean {
        name: &'static str,
        boolean_template_argument: bool,
    },
}

/// Owned MLX custom Metal kernel function compiled from a source string.
#[derive(Debug)]
pub struct MlxMetalKernel {
    raw_kernel: raw::mlx_fast_metal_kernel,
}

impl MlxMetalKernel {
    /// Declares a kernel from its name, argument names, and Metal source body,
    /// which MLX wraps in a generated function signature.
    ///
    /// # Errors
    /// Fails when no inputs or outputs are named, or a string holds a null byte.
    pub fn new(
        kernel_name: &str,
        input_names: &[&str],
        output_names: &[&str],
        kernel_source: &str,
    ) -> Result<Self, MlxCError> {
        Self::new_with_header(kernel_name, input_names, output_names, "", kernel_source)
    }

    /// Like `new`, with source placed before the generated function, for helper
    /// functions and includes.
    ///
    /// # Errors
    /// Fails when no inputs or outputs are named, or a string holds a null byte.
    pub fn new_with_header(
        kernel_name: &str,
        input_names: &[&str],
        output_names: &[&str],
        kernel_header: &str,
        kernel_source: &str,
    ) -> Result<Self, MlxCError> {
        Self::new_with_options(
            kernel_name,
            input_names,
            output_names,
            kernel_header,
            kernel_source,
            false,
        )
    }

    /// Like `new_with_header`, also choosing whether outputs are declared as
    /// `atomic`, which a kernel needs to write them from many threads safely.
    ///
    /// # Errors
    /// Fails when no inputs or outputs are named, or a string holds a null byte.
    pub fn new_with_options(
        kernel_name: &str,
        input_names: &[&str],
        output_names: &[&str],
        kernel_header: &str,
        kernel_source: &str,
        has_atomic_outputs: bool,
    ) -> Result<Self, MlxCError> {
        const OPERATION: &str = "create an MLX custom Metal kernel";
        if input_names.is_empty() || output_names.is_empty() {
            return Err(MlxCError {
                operation: OPERATION,
                description: "custom Metal kernels must declare at least one input and output"
                    .to_owned(),
            });
        }
        let kernel_name = c_string(kernel_name, OPERATION, "kernel name")?;
        let kernel_source = c_string(kernel_source, OPERATION, "kernel source")?;
        let kernel_header = c_string(kernel_header, OPERATION, "kernel header")?;
        let input_name_vector = MlxStringVector::new(input_names, OPERATION)?;
        let output_name_vector = MlxStringVector::new(output_names, OPERATION)?;
        // SAFETY: C strings and vectors stay live through this copying constructor;
        // Row-contiguous inputs and the explicit atomic-output contract match this kernel.
        let raw_kernel = unsafe {
            raw::mlx_fast_metal_kernel_new(
                kernel_name.as_ptr(),
                input_name_vector.raw(),
                output_name_vector.raw(),
                kernel_source.as_ptr(),
                kernel_header.as_ptr(),
                true,
                has_atomic_outputs,
            )
        };
        if raw_kernel.ctx.is_null() {
            return Err(MlxCError {
                operation: OPERATION,
                description: "MLX returned an empty custom Metal kernel handle".to_owned(),
            });
        }
        Ok(Self { raw_kernel })
    }

    const fn raw(&self) -> raw::mlx_fast_metal_kernel {
        self.raw_kernel
    }
}

impl Drop for MlxMetalKernel {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live kernel handle exactly once.
        unsafe {
            raw::mlx_fast_metal_kernel_free(self.raw_kernel);
        }
    }
}

impl MlxBindingsContext {
    /// Applies one owned custom Metal kernel on this context's stream.
    pub fn apply_metal_kernel(
        &self,
        kernel: &MlxMetalKernel,
        input_arrays: &[&MlxArray],
        output_specs: &[MlxMetalKernelOutput],
        grid: [i32; 3],
        thread_group: [i32; 3],
        template_arguments: &[MlxMetalKernelTemplateArgument],
    ) -> Result<Vec<MlxArray>, MlxCError> {
        self.apply_metal_kernel_with_output_initialization(
            kernel,
            input_arrays,
            output_specs,
            grid,
            thread_group,
            template_arguments,
            None,
        )
    }

    /// Like `apply_metal_kernel`, also filling outputs with
    /// `output_initial_value` so a kernel can leave elements unwritten.
    ///
    /// # Errors
    /// Fails when the launch dimensions or argument counts are invalid. A shader
    /// that does not compile fails later, when the output is evaluated.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_metal_kernel_with_output_initialization(
        &self,
        kernel: &MlxMetalKernel,
        input_arrays: &[&MlxArray],
        output_specs: &[MlxMetalKernelOutput],
        grid: [i32; 3],
        thread_group: [i32; 3],
        template_arguments: &[MlxMetalKernelTemplateArgument],
        output_initial_value: Option<f32>,
    ) -> Result<Vec<MlxArray>, MlxCError> {
        const OPERATION: &str = "apply an MLX custom Metal kernel";
        validate_kernel_launch(input_arrays, output_specs, grid, thread_group)?;
        let mut kernel_config =
            MlxMetalKernelConfig::new(output_specs, grid, thread_group, output_initial_value)?;
        kernel_config.add_template_arguments(template_arguments)?;
        let outputs = launch_configured_metal_kernel(
            kernel,
            input_arrays,
            &kernel_config,
            self.gpu_stream(),
            OPERATION,
        )?;
        if outputs.len() != output_specs.len() {
            return Err(MlxCError {
                operation: OPERATION,
                description: format!(
                    "custom Metal kernel returned {} outputs but {} were requested",
                    outputs.len(),
                    output_specs.len()
                ),
            });
        }
        Ok(outputs)
    }

    /// Applies a kernel with a caller-built [`MlxMetalKernelConfig`].
    ///
    /// MLX's kernel call takes `verbose` on the call, and MLX-C stores that flag
    /// on the config passed to `mlx_fast_metal_kernel_apply`. The convenience
    /// methods build a config with verbose left off, which is MLX's default, so
    /// [`MlxMetalKernelConfig::set_verbose`] only affects a launch that receives
    /// this config.
    ///
    /// # Errors
    /// Fails when the launch has no inputs, or MLX rejects the launch. A shader
    /// that does not compile fails later, when the output is evaluated.
    pub fn apply_metal_kernel_with_config(
        &self,
        kernel: &MlxMetalKernel,
        input_arrays: &[&MlxArray],
        kernel_config: &MlxMetalKernelConfig,
    ) -> Result<Vec<MlxArray>, MlxCError> {
        launch_configured_metal_kernel(
            kernel,
            input_arrays,
            kernel_config,
            self.gpu_stream(),
            "apply an MLX custom Metal kernel",
        )
    }
}

/// Applies one custom Metal kernel inside a compiled-graph trace.
///
/// Identical to [`MlxBindingsContext::apply_metal_kernel`] minus the runtime handle:
/// compiled-graph builders run without a runtime, on the stream the trace is
/// being built for. Errors surface as plain status codes. Launch dimensions are
/// rejected by [`MlxMetalKernelConfig::new`] unless every component is positive.
#[allow(clippy::too_many_arguments)]
pub fn apply_metal_kernel_in_graph(
    kernel: &MlxMetalKernel,
    input_arrays: &[&MlxArray],
    output_specs: &[MlxMetalKernelOutput],
    grid: [i32; 3],
    thread_group: [i32; 3],
    template_arguments: &[MlxMetalKernelTemplateArgument],
    gpu_stream: &MlxStream,
) -> Result<Vec<MlxArray>, i32> {
    let mut kernel_config = MlxMetalKernelConfig::new(output_specs, grid, thread_group, None)
        .map_err(graph_launch_failed)?;
    kernel_config
        .add_template_arguments(template_arguments)
        .map_err(graph_launch_failed)?;
    let outputs =
        apply_metal_kernel_in_graph_with_config(kernel, input_arrays, &kernel_config, gpu_stream)?;
    if outputs.len() != output_specs.len() {
        return Err(1);
    }
    Ok(outputs)
}

/// Like [`apply_metal_kernel_in_graph`], with a caller-built config so
/// [`MlxMetalKernelConfig::set_verbose`] reaches the trace.
///
/// # Errors
/// Returns `1` when the launch has no inputs or a handle cannot be built, and
/// MLX's own status when the apply call itself fails.
pub fn apply_metal_kernel_in_graph_with_config(
    kernel: &MlxMetalKernel,
    input_arrays: &[&MlxArray],
    kernel_config: &MlxMetalKernelConfig,
    gpu_stream: &MlxStream,
) -> Result<Vec<MlxArray>, i32> {
    const OPERATION: &str = "apply a Metal kernel in a graph trace";
    if input_arrays.is_empty() {
        return Err(1);
    }
    let input_vector = MlxArrayVector::new(input_arrays).map_err(graph_launch_failed)?;
    let mut output_vector = MlxArrayVector::empty(OPERATION).map_err(graph_launch_failed)?;
    let status = apply_metal_kernel_raw(
        kernel,
        &input_vector,
        kernel_config,
        gpu_stream,
        &mut output_vector,
    );
    if status != 0 {
        return Err(status);
    }
    collect_kernel_outputs(&output_vector, OPERATION).map_err(graph_launch_failed)
}

/// One launch's configuration for a custom Metal kernel: its outputs, grid, thread
/// group, template arguments, and whether MLX prints the generated shader.
#[derive(Debug)]
pub struct MlxMetalKernelConfig {
    raw_config: raw::mlx_fast_metal_kernel_config,
}

impl MlxMetalKernelConfig {
    /// Builds a launch configuration from output declarations, grid, thread
    /// group, and an optional output fill value.
    ///
    /// # Errors
    /// Fails when a grid or thread-group dimension is not positive, or when MLX
    /// rejects a configuration value.
    pub fn new(
        output_specs: &[MlxMetalKernelOutput],
        grid: [i32; 3],
        thread_group: [i32; 3],
        output_initial_value: Option<f32>,
    ) -> Result<Self, MlxCError> {
        const OPERATION: &str = "configure an MLX custom Metal kernel";
        reject_non_positive_launch_dimensions(grid, thread_group, OPERATION)?;
        // SAFETY: The runtime error handler is installed before model loading,
        // and MLX returns one owned config handle.
        let raw_config = unsafe { raw::mlx_fast_metal_kernel_config_new() };
        if raw_config.ctx.is_null() {
            return Err(MlxCError {
                operation: OPERATION,
                description: "MLX returned an empty custom Metal kernel config".to_owned(),
            });
        }
        let kernel_config = Self { raw_config };
        kernel_config.set_output_initial_value(output_initial_value, OPERATION)?;
        kernel_config.add_output_specs(output_specs, OPERATION)?;
        kernel_config.set_launch_dimensions(grid, thread_group, OPERATION)?;
        Ok(kernel_config)
    }

    fn set_output_initial_value(
        &self,
        output_initial_value: Option<f32>,
        operation: &'static str,
    ) -> Result<(), MlxCError> {
        if let Some(output_initial_value) = output_initial_value {
            // SAFETY: The scalar initialization value is copied by MLX.
            let status = unsafe {
                raw::mlx_fast_metal_kernel_config_set_init_value(self.raw(), output_initial_value)
            };
            check_status(status, operation)?;
        }
        Ok(())
    }

    fn add_output_specs(
        &self,
        output_specs: &[MlxMetalKernelOutput],
        operation: &'static str,
    ) -> Result<(), MlxCError> {
        for output_spec in output_specs {
            // SAFETY: The shape slice remains live for this copying call.
            let status = unsafe {
                raw::mlx_fast_metal_kernel_config_add_output_arg(
                    self.raw(),
                    output_spec.shape.as_ptr(),
                    output_spec.shape.len(),
                    output_spec.dtype.to_raw(),
                )
            };
            check_status(status, operation)?;
        }
        Ok(())
    }

    fn set_launch_dimensions(
        &self,
        grid: [i32; 3],
        thread_group: [i32; 3],
        operation: &'static str,
    ) -> Result<(), MlxCError> {
        // SAFETY: Non-positive grid components were rejected by `new`, so these
        // three scalars are positive. The config handle is live and MLX copies them.
        let status = unsafe {
            raw::mlx_fast_metal_kernel_config_set_grid(self.raw(), grid[0], grid[1], grid[2])
        };
        check_status(status, operation)?;
        // SAFETY: Non-positive thread-group components were rejected by `new`,
        // so these three scalars are positive. The handle is live and MLX copies them.
        let status = unsafe {
            raw::mlx_fast_metal_kernel_config_set_thread_group(
                self.raw(),
                thread_group[0],
                thread_group[1],
                thread_group[2],
            )
        };
        check_status(status, operation)
    }

    const fn raw(&self) -> raw::mlx_fast_metal_kernel_config {
        self.raw_config
    }

    /// Adds compile-time template arguments to this launch.
    ///
    /// The signature takes `&mut self` because the call writes into the config
    /// MLX already allocated. A shared reference would hide that write.
    ///
    /// # Errors
    /// Fails when a name contains an interior null byte, or MLX rejects a value.
    pub fn add_template_arguments(
        &mut self,
        template_arguments: &[MlxMetalKernelTemplateArgument],
    ) -> Result<(), MlxCError> {
        const OPERATION: &str = "configure an MLX custom Metal kernel";
        for template_argument in template_arguments {
            match *template_argument {
                MlxMetalKernelTemplateArgument::Dtype { name, dtype } => {
                    let argument_name = c_string(name, OPERATION, "template dtype name")?;
                    // SAFETY: Name is live for this copying call; dtype is validated by MLX.
                    let status = unsafe {
                        raw::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            self.raw(),
                            argument_name.as_ptr(),
                            dtype.to_raw(),
                        )
                    };
                    check_status(status, OPERATION)?;
                }
                MlxMetalKernelTemplateArgument::Integer {
                    name,
                    integer_template_argument,
                } => {
                    let argument_name = c_string(name, OPERATION, "template integer name")?;
                    // SAFETY: Name is live for this copying call; the scalar argument is copied.
                    let status = unsafe {
                        raw::mlx_fast_metal_kernel_config_add_template_arg_int(
                            self.raw(),
                            argument_name.as_ptr(),
                            integer_template_argument,
                        )
                    };
                    check_status(status, OPERATION)?;
                }
                MlxMetalKernelTemplateArgument::Boolean {
                    name,
                    boolean_template_argument,
                } => {
                    let argument_name = c_string(name, OPERATION, "template boolean name")?;
                    // SAFETY: Name is live for this copying call; the scalar argument is copied.
                    let status = unsafe {
                        raw::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            self.raw(),
                            argument_name.as_ptr(),
                            boolean_template_argument,
                        )
                    };
                    check_status(status, OPERATION)?;
                }
            }
        }
        Ok(())
    }

    /// Asks MLX to print the generated shader when this config is applied.
    ///
    /// This is the `verbose` argument of MLX's kernel call. It has no effect
    /// until the config is passed to
    /// [`MlxBindingsContext::apply_metal_kernel_with_config`] or
    /// [`apply_metal_kernel_in_graph_with_config`]. The convenience launch
    /// methods leave it off, which is MLX's default.
    pub fn set_verbose(&mut self, verbose: bool) -> Result<(), MlxCError> {
        const OPERATION: &str = "configure MLX custom Metal kernel logging";
        // SAFETY: The config handle is live and the flag is a plain value.
        let status = unsafe { raw::mlx_fast_metal_kernel_config_set_verbose(self.raw(), verbose) };
        check_status(status, OPERATION)
    }
}

impl Drop for MlxMetalKernelConfig {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live config handle exactly once.
        unsafe {
            raw::mlx_fast_metal_kernel_config_free(self.raw_config);
        }
    }
}

#[derive(Debug)]
struct MlxStringVector {
    raw_vector: raw::mlx_vector_string,
}

impl MlxStringVector {
    fn new(strings: &[&str], operation: &'static str) -> Result<Self, MlxCError> {
        let c_strings = strings
            .iter()
            .map(|source_string| c_string(source_string, operation, "string vector entry"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut string_pointers = c_strings
            .iter()
            .map(|c_string| c_string.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: Pointers remain live for this copying constructor.
        let raw_vector = unsafe {
            raw::mlx_vector_string_new_data(string_pointers.as_mut_ptr(), string_pointers.len())
        };
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty string vector handle".to_owned(),
            });
        }
        Ok(Self { raw_vector })
    }

    const fn raw(&self) -> raw::mlx_vector_string {
        self.raw_vector
    }
}

impl Drop for MlxStringVector {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live string vector handle exactly once.
        unsafe {
            raw::mlx_vector_string_free(self.raw_vector);
        }
    }
}

fn launch_configured_metal_kernel(
    kernel: &MlxMetalKernel,
    input_arrays: &[&MlxArray],
    kernel_config: &MlxMetalKernelConfig,
    gpu_stream: &MlxStream,
    operation: &'static str,
) -> Result<Vec<MlxArray>, MlxCError> {
    if input_arrays.is_empty() {
        return Err(MlxCError {
            operation,
            description: "custom Metal kernel launches require at least one input".to_owned(),
        });
    }
    let input_vector = MlxArrayVector::new(input_arrays)?;
    let mut output_vector = MlxArrayVector::empty(operation)?;
    let status = apply_metal_kernel_raw(
        kernel,
        &input_vector,
        kernel_config,
        gpu_stream,
        &mut output_vector,
    );
    check_status(status, operation)?;
    collect_kernel_outputs(&output_vector, operation)
}

fn apply_metal_kernel_raw(
    kernel: &MlxMetalKernel,
    input_vector: &MlxArrayVector,
    kernel_config: &MlxMetalKernelConfig,
    gpu_stream: &MlxStream,
    output_vector: &mut MlxArrayVector,
) -> i32 {
    // SAFETY: Kernel, input vector, config, stream, and output vector are live.
    // MLX writes the lazy outputs into the vector before returning.
    unsafe {
        raw::mlx_fast_metal_kernel_apply(
            output_vector.raw_mut(),
            kernel.raw(),
            input_vector.raw(),
            kernel_config.raw(),
            gpu_stream.raw(),
        )
    }
}

fn collect_kernel_outputs(
    output_vector: &MlxArrayVector,
    operation: &'static str,
) -> Result<Vec<MlxArray>, MlxCError> {
    let output_count = output_vector.len();
    let mut outputs = Vec::with_capacity(output_count);
    for output_index in 0..output_count {
        outputs.push(output_vector.array_at(output_index, operation)?);
    }
    Ok(outputs)
}

fn graph_launch_failed<T>(_launch_error: T) -> i32 {
    1
}

fn validate_kernel_launch(
    input_arrays: &[&MlxArray],
    output_specs: &[MlxMetalKernelOutput],
    grid: [i32; 3],
    thread_group: [i32; 3],
) -> Result<(), MlxCError> {
    const OPERATION: &str = "apply an MLX custom Metal kernel";
    if input_arrays.is_empty() || output_specs.is_empty() {
        return Err(MlxCError {
            operation: OPERATION,
            description: "custom Metal kernel launches require at least one input and output"
                .to_owned(),
        });
    }
    reject_non_positive_launch_dimensions(grid, thread_group, OPERATION)
}

fn reject_non_positive_launch_dimensions(
    grid: [i32; 3],
    thread_group: [i32; 3],
    operation: &'static str,
) -> Result<(), MlxCError> {
    if dimensions_are_positive(grid) && dimensions_are_positive(thread_group) {
        return Ok(());
    }
    Err(MlxCError {
        operation,
        description: "custom Metal kernel grid and threadgroup dimensions must be positive"
            .to_owned(),
    })
}

fn dimensions_are_positive(dimensions: [i32; 3]) -> bool {
    for dimension in dimensions {
        if dimension <= 0 {
            return false;
        }
    }
    true
}

fn c_string(
    source_text: &str,
    operation: &'static str,
    value_description: &'static str,
) -> Result<CString, MlxCError> {
    CString::new(source_text).map_err(|source| nul_error(operation, value_description, source))
}

fn nul_error(
    operation: &'static str,
    value_description: &'static str,
    source: NulError,
) -> MlxCError {
    MlxCError {
        operation,
        description: format!("{value_description} contains an interior NUL byte: {source}"),
    }
}
