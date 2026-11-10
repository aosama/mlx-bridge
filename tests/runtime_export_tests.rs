//! Tier 6 contracts for exported functions: a Rust callable traced to a file
//! and replayed through the imported handle, plus graph rendering.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use std::ffi::{c_char, c_int, c_void};

use bridge_fixtures::{
    array_at, float_array, floats_of, gpu_context, scratch_file_path, vector_of,
};
use mlx_bridge::{
    GraphOutputFile, MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure,
    MlxClosureKwargs, MlxFunctionExporter, MlxImportedFunction, MlxMapStringToArray, MlxNodeNamer,
    export_function, export_function_with_keywords, export_graph_to_dot, print_graph,
};

unsafe extern "C" {
    fn fopen(path: *const c_char, mode: *const c_char) -> *mut c_void;
    fn fclose(stream: *mut c_void) -> c_int;
}

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn keyword_scaled(
    inputs: &[MlxArray],
    keywords: &MlxMapStringToArray,
) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    let scale = keywords.value_at("scale")?.ok_or(MlxCError {
        operation: "read the fixture keyword",
        description: "the scale keyword is required".to_owned(),
    })?;
    Ok(vec![context.multiply(&inputs[0], &scale)?])
}

fn refusing_callback(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    Err(MlxCError {
        operation: "trace the exported fixture",
        description: "the trace refused".to_owned(),
    })
}

#[test]
fn should_replay_an_exported_function_after_its_closure_drops() {
    gpu_context();
    let path = scratch_file_path("doubled.mlxfn");
    let function = MlxClosure::from_fn(doubled).expect("the closure should build");
    let example = float_array(&[1.0, 2.0], &[2]);

    export_function(&path, &function, &vector_of(&[&example]), false).unwrap();
    drop(function);
    let imported = MlxImportedFunction::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let outputs = imported.apply(&vector_of(&[&example])).unwrap();
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0, 4.0]);
}

#[test]
fn should_replay_a_shapeless_export_with_a_new_shape() {
    gpu_context();
    let path = scratch_file_path("doubled_shapeless.mlxfn");
    let function = MlxClosure::from_fn(doubled).expect("the closure should build");
    let example = float_array(&[1.0, 2.0], &[2]);
    let longer_input = float_array(&[1.0, 2.0, 3.0], &[3]);

    export_function(&path, &function, &vector_of(&[&example]), true).unwrap();
    let imported = MlxImportedFunction::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let outputs = imported.apply(&vector_of(&[&longer_input])).unwrap();
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0, 4.0, 6.0]);
}

#[test]
fn should_replay_an_exported_keyword_function() {
    gpu_context();
    let path = scratch_file_path("keyword_scaled.mlxfn");
    let function = MlxClosureKwargs::from_fn(keyword_scaled).expect("the closure should build");
    let example = float_array(&[2.0], &[1]);
    let mut keywords = MlxMapStringToArray::new().unwrap();
    keywords
        .insert("scale", &float_array(&[3.0], &[1]))
        .unwrap();

    export_function_with_keywords(&path, &function, &vector_of(&[&example]), &keywords, false)
        .unwrap();
    let imported = MlxImportedFunction::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let outputs = imported
        .apply_with_keywords(&vector_of(&[&example]), &keywords)
        .unwrap();
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![6.0]);
}

#[test]
fn should_replay_several_traces_recorded_through_an_exporter() {
    gpu_context();
    let path = scratch_file_path("exporter.mlxfn");
    let function = MlxClosure::from_fn(doubled).expect("the closure should build");
    let short_input = float_array(&[1.0], &[1]);
    let long_input = float_array(&[1.0, 2.0], &[2]);

    let exporter = MlxFunctionExporter::new(&path, &function, false).unwrap();
    exporter.apply(&vector_of(&[&short_input])).unwrap();
    exporter.apply(&vector_of(&[&long_input])).unwrap();
    drop(exporter);
    let imported = MlxImportedFunction::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let short_outputs = imported.apply(&vector_of(&[&short_input])).unwrap();
    let long_outputs = imported.apply(&vector_of(&[&long_input])).unwrap();
    assert_eq!(floats_of(&array_at(&short_outputs, 0)), vec![2.0]);
    assert_eq!(floats_of(&array_at(&long_outputs, 0)), vec![2.0, 4.0]);
}

#[test]
fn should_surface_a_callback_failure_during_export() {
    gpu_context();
    let path = scratch_file_path("refused.mlxfn");
    let function = MlxClosure::from_fn(refusing_callback).expect("the closure should build");
    let example = float_array(&[1.0], &[1]);

    let export_outcome = export_function(&path, &function, &vector_of(&[&example]), false);
    let _ignored_cleanup = std::fs::remove_file(&path);

    assert!(export_outcome.is_err());
}

#[test]
fn should_translate_a_missing_or_malformed_import_into_an_error() {
    gpu_context();
    let missing_path = scratch_file_path("missing.mlxfn");
    let malformed_path = scratch_file_path("malformed.mlxfn");
    std::fs::write(&malformed_path, b"not an exported function").unwrap();

    let malformed_outcome = MlxImportedFunction::load(&malformed_path);
    std::fs::remove_file(&malformed_path).unwrap();

    assert!(MlxImportedFunction::load(&missing_path).is_err());
    assert!(malformed_outcome.is_err());
    assert!(MlxImportedFunction::load("a\0b").is_err());
}

#[test]
fn should_read_back_a_node_name() {
    gpu_context();
    let node = float_array(&[1.0], &[1]);
    let mut namer = MlxNodeNamer::new().unwrap();

    namer.set_name(&node, "input_node").unwrap();

    assert_eq!(namer.name_of(&node).unwrap(), "input_node");
    assert!(namer.set_name(&node, "a\0b").is_err());
}

type GraphRenderer =
    unsafe fn(GraphOutputFile, &MlxNodeNamer, &MlxArrayVector) -> Result<(), MlxCError>;

/// Renders one named addition into a scratch file and returns the file's text.
fn render_named_addition(file_name: &str, renderer: GraphRenderer) -> String {
    let context = gpu_context();
    let path = scratch_file_path(file_name);
    let path_argument = std::ffi::CString::new(path.as_str()).unwrap();
    let input = float_array(&[1.0], &[1]);
    let output = context.add(&input, &input).unwrap();
    let mut namer = MlxNodeNamer::new().unwrap();
    namer.set_name(&input, "input_node").unwrap();

    // SAFETY: Both C strings are valid, the file opens before rendering, and
    // it stays open until the render call returns.
    let render_outcome = unsafe {
        let open_file = fopen(path_argument.as_ptr(), c"w".as_ptr());
        assert!(!open_file.is_null(), "the scratch file should open");
        let render_outcome = renderer(GraphOutputFile(open_file), &namer, &vector_of(&[&output]));
        assert_eq!(fclose(open_file), 0);
        render_outcome
    };
    let rendered_graph = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    render_outcome.unwrap();
    rendered_graph
}

#[test]
fn should_render_a_named_graph_to_a_dot_file() {
    let rendered_graph = render_named_addition("graph.dot", export_graph_to_dot);

    assert!(rendered_graph.contains("digraph"), "{rendered_graph}");
    assert!(rendered_graph.contains("input_node"), "{rendered_graph}");
}

#[test]
fn should_print_a_named_graph_to_a_file() {
    let printed_graph = render_named_addition("graph.txt", print_graph);

    assert!(printed_graph.contains("input_node"), "{printed_graph}");
}
