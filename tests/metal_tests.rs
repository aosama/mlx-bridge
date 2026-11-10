//! Tier 7 contracts for the Metal family: custom kernel launches with verified
//! results, template arguments reaching the shader, launch validation, a shader
//! failure that defers to evaluation but still arrives as a typed error, the
//! metallib path override, and capture control.

use std::fs::File;
use std::io::Read;
use std::os::unix::io::FromRawFd;

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{assert_translated_failure, float_array, floats_of, gpu_context};
use mlx_bridge::{
    MlxDtype, MlxMetalKernel, MlxMetalKernelConfig, MlxMetalKernelOutput,
    MlxMetalKernelTemplateArgument, apply_metal_kernel_in_graph_with_config, metallib_path,
    set_metallib_path,
};

const DOUBLING_SHADER: &str = "
    uint element_position = thread_position_in_grid.x;
    output_values[element_position] = input_values[element_position] * 2.0;
";

const TEMPLATE_SCALING_SHADER: &str = "
    uint element_position = thread_position_in_grid.x;
    output_values[element_position] = element_type(scale) * input_values[element_position];
";

fn doubling_kernel(kernel_name: &str) -> MlxMetalKernel {
    MlxMetalKernel::new(
        kernel_name,
        &["input_values"],
        &["output_values"],
        DOUBLING_SHADER,
    )
    .expect("the kernel should build")
}

fn single_float_output(element_count: i32) -> Vec<MlxMetalKernelOutput> {
    vec![MlxMetalKernelOutput::new(
        vec![element_count],
        MlxDtype::Float32,
    )]
}

#[test]
fn should_launch_a_custom_kernel_and_return_its_results() {
    let context = gpu_context();
    let kernel = MlxMetalKernel::new(
        "doubling_kernel",
        &["input_values"],
        &["output_values"],
        DOUBLING_SHADER,
    )
    .expect("the kernel should build");
    let input = float_array(&[1.0, 2.0, 3.0], &[3]);

    let outputs = context
        .apply_metal_kernel(
            &kernel,
            &[&input],
            &single_float_output(3),
            [3, 1, 1],
            [1, 1, 1],
            &[],
        )
        .expect("the launch should build");

    assert_eq!(outputs.len(), 1);
    assert_eq!(floats_of(&outputs[0]), vec![2.0, 4.0, 6.0]);
}

#[test]
fn should_pass_template_arguments_to_the_shader() {
    let context = gpu_context();
    let kernel = MlxMetalKernel::new(
        "template_scaling_kernel",
        &["input_values"],
        &["output_values"],
        TEMPLATE_SCALING_SHADER,
    )
    .expect("the kernel should build");
    let input = float_array(&[1.0, 2.0], &[2]);
    let template_arguments = [
        MlxMetalKernelTemplateArgument::Dtype {
            name: "element_type",
            dtype: MlxDtype::Float32,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "scale",
            integer_template_argument: 3,
        },
    ];

    let outputs = context
        .apply_metal_kernel(
            &kernel,
            &[&input],
            &single_float_output(2),
            [2, 1, 1],
            [1, 1, 1],
            &template_arguments,
        )
        .expect("the launch should build");

    assert_eq!(floats_of(&outputs[0]), vec![3.0, 6.0]);
}

#[test]
fn should_initialize_outputs_the_shader_leaves_unwritten() {
    let context = gpu_context();
    let kernel = MlxMetalKernel::new(
        "untouched_output_kernel",
        &["input_values"],
        &["output_values"],
        "
    uint element_position = thread_position_in_grid.x;
    if (element_position == 0) {
        output_values[element_position] = input_values[element_position];
    }
",
    )
    .expect("the kernel should build");
    let input = float_array(&[1.0, 2.0], &[2]);

    let outputs = context
        .apply_metal_kernel_with_output_initialization(
            &kernel,
            &[&input],
            &single_float_output(2),
            [2, 1, 1],
            [1, 1, 1],
            &[],
            Some(5.0),
        )
        .expect("the launch should build");

    assert_eq!(floats_of(&outputs[0]), vec![1.0, 5.0]);
}

#[test]
fn should_defer_a_shader_compilation_failure_to_evaluation() {
    let context = gpu_context();
    let kernel = MlxMetalKernel::new(
        "broken_shader_kernel",
        &["input_values"],
        &["output_values"],
        "this is not metal shading language;",
    )
    .expect("building the kernel does not compile the shader");
    let input = float_array(&[1.0], &[1]);

    let outputs = context
        .apply_metal_kernel(
            &kernel,
            &[&input],
            &single_float_output(1),
            [1, 1, 1],
            [1, 1, 1],
            &[],
        )
        .expect("the lazy graph builds without touching the shader");

    let failure = outputs[0]
        .evaluate()
        .expect_err("the shader fails when the graph runs, without aborting");
    assert_translated_failure(&failure, "evaluate");
}

#[test]
fn should_reject_a_kernel_without_inputs_or_outputs() {
    assert!(MlxMetalKernel::new("no_inputs", &[], &["output_values"], DOUBLING_SHADER).is_err());
    assert!(MlxMetalKernel::new("no_outputs", &["input_values"], &[], DOUBLING_SHADER).is_err());
}

#[test]
fn should_reject_kernel_text_with_an_interior_null_byte() {
    assert!(MlxMetalKernel::new("a\0b", &["input_values"], &["output_values"], "").is_err());
    assert!(MlxMetalKernel::new("kernel", &["a\0b"], &["output_values"], "").is_err());
    assert!(MlxMetalKernel::new("kernel", &["input_values"], &["output_values"], "a\0b").is_err());
}

#[test]
fn should_reject_a_launch_with_a_non_positive_grid() {
    let context = gpu_context();
    let kernel = MlxMetalKernel::new(
        "grid_kernel",
        &["input_values"],
        &["output_values"],
        DOUBLING_SHADER,
    )
    .expect("the kernel should build");
    let input = float_array(&[1.0], &[1]);

    let zero_grid = context.apply_metal_kernel(
        &kernel,
        &[&input],
        &single_float_output(1),
        [0, 1, 1],
        [1, 1, 1],
        &[],
    );
    let negative_group = context.apply_metal_kernel(
        &kernel,
        &[&input],
        &single_float_output(1),
        [1, 1, 1],
        [-1, 1, 1],
        &[],
    );

    assert!(zero_grid.is_err());
    assert!(negative_group.is_err());
}

#[test]
fn should_reject_a_launch_without_inputs_or_outputs() {
    let context = gpu_context();
    let kernel = MlxMetalKernel::new(
        "empty_launch",
        &["input_values"],
        &["output_values"],
        DOUBLING_SHADER,
    )
    .expect("the kernel should build");
    let input = float_array(&[1.0], &[1]);

    let without_inputs = context.apply_metal_kernel(
        &kernel,
        &[],
        &single_float_output(1),
        [1, 1, 1],
        [1, 1, 1],
        &[],
    );
    let without_outputs =
        context.apply_metal_kernel(&kernel, &[&input], &[], [1, 1, 1], [1, 1, 1], &[]);

    assert!(without_inputs.is_err());
    assert!(without_outputs.is_err());
}

#[test]
fn should_translate_a_capture_start_without_the_capture_layer() {
    // Metal records only under MTL_CAPTURE_ENABLED, which a run with the
    // capture layer deliberately opts into, so the failure path is unprovable there.
    if std::env::var_os("MTL_CAPTURE_ENABLED").is_some() {
        return;
    }
    let context = gpu_context();

    let failure = context
        .start_metal_capture("/nonexistent/capture.gputrace")
        .expect_err("capture needs the layer the environment did not insert");

    assert_translated_failure(&failure, "capture");
    assert!(context.start_metal_capture("a\0b").is_err());
}

#[test]
fn should_apply_a_caller_built_config_and_return_its_results() {
    let context = gpu_context();
    let kernel = doubling_kernel("config_doubling_kernel");
    let input = float_array(&[1.0, 2.0, 3.0], &[3]);
    let mut kernel_config =
        MlxMetalKernelConfig::new(&single_float_output(3), [3, 1, 1], [1, 1, 1], None)
            .expect("the config should build");
    kernel_config
        .set_verbose(false)
        .expect("turning logging off should be accepted");

    let outputs = context
        .apply_metal_kernel_with_config(&kernel, &[&input], &kernel_config)
        .expect("the configured launch should build");

    assert_eq!(floats_of(&outputs[0]), vec![2.0, 4.0, 6.0]);
}

#[test]
fn should_print_the_generated_shader_when_the_config_is_verbose() {
    let context = gpu_context();
    let kernel = doubling_kernel("verbose_shader_kernel");
    let input = float_array(&[1.0], &[1]);
    let mut kernel_config =
        MlxMetalKernelConfig::new(&single_float_output(1), [1, 1, 1], [1, 1, 1], None)
            .expect("the minimum positive config should build");
    kernel_config
        .set_verbose(true)
        .expect("verbose logging should be accepted");
    let _stdout_lock = std::io::stdout().lock();
    let captured_stdout = RedirectedStdout::start();

    let outputs = context
        .apply_metal_kernel_with_config(&kernel, &[&input], &kernel_config)
        .expect("the verbose launch should build");
    let launch_text = captured_stdout.finish();

    assert!(
        launch_text.contains("Generated source code for `verbose_shader_kernel`"),
        "verbose must reach the launch, got {launch_text:?}"
    );
    assert_eq!(floats_of(&outputs[0]), vec![2.0]);
}

#[test]
fn should_not_print_the_generated_shader_when_verbose_is_off() {
    let context = gpu_context();
    let kernel = doubling_kernel("quiet_shader_kernel");
    let input = float_array(&[4.0], &[1]);
    let kernel_config =
        MlxMetalKernelConfig::new(&single_float_output(1), [1, 1, 1], [1, 1, 1], None)
            .expect("the config should build");
    let _stdout_lock = std::io::stdout().lock();
    let captured_stdout = RedirectedStdout::start();

    let outputs = context
        .apply_metal_kernel_with_config(&kernel, &[&input], &kernel_config)
        .expect("the quiet launch should build");
    let launch_text = captured_stdout.finish();

    assert!(
        !launch_text.contains("Generated source code"),
        "the default config must not print the shader, got {launch_text:?}"
    );
    assert_eq!(floats_of(&outputs[0]), vec![8.0]);
}

#[test]
fn should_reject_a_kernel_config_with_a_non_positive_grid() {
    let zero_grid = MlxMetalKernelConfig::new(&single_float_output(1), [0, 1, 1], [1, 1, 1], None);
    let negative_group =
        MlxMetalKernelConfig::new(&single_float_output(1), [1, 1, 1], [-1, 1, 1], None);

    assert!(zero_grid.is_err());
    assert!(negative_group.is_err());
}

#[test]
fn should_apply_a_configured_kernel_on_an_explicit_stream() {
    let context = gpu_context();
    let kernel = doubling_kernel("graph_config_kernel");
    let input = float_array(&[1.0, 2.0], &[2]);
    let mut kernel_config =
        MlxMetalKernelConfig::new(&single_float_output(2), [2, 1, 1], [1, 1, 1], None)
            .expect("the config should build");
    kernel_config
        .set_verbose(false)
        .expect("turning logging off should be accepted");

    let outputs = apply_metal_kernel_in_graph_with_config(
        &kernel,
        &[&input],
        &kernel_config,
        context.gpu_stream(),
    )
    .expect("the explicit-stream launch should build");

    assert_eq!(floats_of(&outputs[0]), vec![2.0, 4.0]);
}

#[test]
fn should_reject_a_graph_config_launch_without_inputs() {
    let context = gpu_context();
    let kernel = doubling_kernel("empty_graph_config_launch");
    let kernel_config =
        MlxMetalKernelConfig::new(&single_float_output(1), [1, 1, 1], [1, 1, 1], None)
            .expect("the config should build");

    let without_inputs =
        apply_metal_kernel_in_graph_with_config(&kernel, &[], &kernel_config, context.gpu_stream());

    assert_eq!(without_inputs.expect_err("an empty launch is rejected"), 1);
}

#[test]
fn should_reject_a_configured_launch_without_inputs() {
    let context = gpu_context();
    let kernel = doubling_kernel("empty_config_launch");
    let kernel_config =
        MlxMetalKernelConfig::new(&single_float_output(1), [1, 1, 1], [1, 1, 1], None)
            .expect("the config should build");

    let without_inputs = context.apply_metal_kernel_with_config(&kernel, &[], &kernel_config);

    assert!(without_inputs.is_err());
}

#[test]
fn should_read_back_the_metallib_override_and_keep_computing() {
    let context = gpu_context();
    let values = float_array(&[1.0], &[1]);
    // The kernels load on first GPU use, so force that before overriding.
    assert_eq!(
        floats_of(&context.add(&values, &values).unwrap()),
        vec![2.0]
    );
    let original_path = std::ffi::CString::new(metallib_path().unwrap()).unwrap();

    set_metallib_path(c"/nonexistent/override.metallib").unwrap();
    let observed_path = metallib_path().unwrap();
    set_metallib_path(&original_path).unwrap();

    assert_eq!(observed_path, "/nonexistent/override.metallib");
    assert_eq!(
        floats_of(&context.add(&values, &values).unwrap()),
        vec![2.0]
    );
}

/// Darwin `fcntl` commands. The read end is set non-blocking so a quiet launch,
/// which prints nothing, cannot stall the test on `read`.
const F_GETFL: i32 = 3;
const F_SETFL: i32 = 4;
const O_NONBLOCK: i32 = 0x0004;

unsafe extern "C" {
    #[link_name = "dup"]
    fn duplicate_fd(file_descriptor: i32) -> i32;
    #[link_name = "dup2"]
    fn duplicate_fd_onto(file_descriptor: i32, destination_descriptor: i32) -> i32;
    #[link_name = "pipe"]
    fn open_pipe(pipe_descriptors: *mut i32) -> i32;
    #[link_name = "close"]
    fn close_fd(file_descriptor: i32) -> i32;
    #[link_name = "fflush"]
    fn flush_stdio(file: *mut std::ffi::c_void) -> i32;
    #[link_name = "fcntl"]
    fn control_fd(file_descriptor: i32, command: i32, ...) -> i32;
}

struct RedirectedStdout {
    saved_stdout_fd: i32,
    pipe_read_fd: i32,
    restored_stdout: bool,
}

impl RedirectedStdout {
    fn start() -> Self {
        let saved_stdout_fd = unsafe { duplicate_fd(1) };
        assert!(saved_stdout_fd >= 0, "dup of stdout should succeed");
        let mut pipe_descriptors = [0_i32; 2];
        let pipe_status = unsafe { open_pipe(pipe_descriptors.as_mut_ptr()) };
        assert_eq!(pipe_status, 0, "the capture pipe should open");
        let redirect_status = unsafe { duplicate_fd_onto(pipe_descriptors[1], 1) };
        assert_eq!(redirect_status, 1, "stdout should point at the pipe");
        unsafe { close_fd(pipe_descriptors[1]) };
        Self {
            saved_stdout_fd,
            pipe_read_fd: pipe_descriptors[0],
            restored_stdout: false,
        }
    }

    fn finish(mut self) -> String {
        self.restore_stdout();
        read_captured_stdout(self.pipe_read_fd)
    }

    fn restore_stdout(&mut self) {
        if self.restored_stdout {
            return;
        }
        unsafe {
            flush_stdio(std::ptr::null_mut());
            duplicate_fd_onto(self.saved_stdout_fd, 1);
            close_fd(self.saved_stdout_fd);
        }
        self.restored_stdout = true;
    }
}

impl Drop for RedirectedStdout {
    fn drop(&mut self) {
        self.restore_stdout();
        unsafe { close_fd(self.pipe_read_fd) };
    }
}

fn read_captured_stdout(pipe_read_fd: i32) -> String {
    let existing_flags = unsafe { control_fd(pipe_read_fd, F_GETFL) };
    unsafe { control_fd(pipe_read_fd, F_SETFL, existing_flags | O_NONBLOCK) };
    // MLX prints the shader through C++ `std::cout`, which writes file
    // descriptor 1, so Rust's test capture would not see it. The guard closes
    // that descriptor. An owning `File` would close it too, including on panic.
    let mut pipe_file = std::mem::ManuallyDrop::new(unsafe { File::from_raw_fd(pipe_read_fd) });
    let mut captured_bytes = Vec::new();
    let mut read_buffer = [0_u8; 4096];
    loop {
        match pipe_file.read(&mut read_buffer) {
            Ok(0) | Err(_) => break,
            Ok(read_count) => captured_bytes.extend_from_slice(&read_buffer[..read_count]),
        }
    }
    String::from_utf8_lossy(&captured_bytes).into_owned()
}
