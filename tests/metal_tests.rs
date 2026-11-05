//! Tier 7 contracts for the Metal family: custom kernel launches with verified
//! results, template arguments reaching the shader, launch validation, a shader
//! failure that defers to evaluation but still arrives as a typed error, the
//! metallib path override, and capture control.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{assert_translated_failure, float_array, floats_of, gpu_context};
use mlx_c_rust::{
    MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument, metallib_path,
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
