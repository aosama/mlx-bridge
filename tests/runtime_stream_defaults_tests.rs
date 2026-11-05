//! Tier 5 contracts for streams and lazy evaluation: explicit stream identity,
//! per-thread isolation, synchronization ordering, and graphs that only run
//! when evaluated.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, floats_of, gpu_context};
use mlx_c_rust::{
    MlxBindingsContext, MlxDeviceHandle, MlxStream, MlxStreamThreadLocal, raw, set_default_stream,
    synchronize, synchronize_all_defaults,
};

fn gpu_device() -> MlxDeviceHandle {
    MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0).expect("the GPU device should exist")
}

fn cpu_device() -> MlxDeviceHandle {
    MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_CPU, 0).expect("the CPU device should exist")
}

#[test]
fn should_identify_the_default_streams_by_device() {
    gpu_context();
    let gpu_stream = MlxStream::default_gpu().unwrap();
    let cpu_stream = MlxStream::default_cpu().unwrap();

    assert!(gpu_stream.equals(&MlxStream::default_gpu().unwrap()));
    assert!(!gpu_stream.equals(&cpu_stream));
    assert!(gpu_stream.device().unwrap().equals(&gpu_device()));
    assert!(cpu_stream.device().unwrap().equals(&cpu_device()));
}

#[test]
fn should_describe_a_stream_with_its_device() {
    gpu_context();

    let description = MlxStream::default_gpu().unwrap().to_string_lossy().unwrap();

    assert!(description.contains("gpu"), "{description}");
}

#[test]
fn should_create_a_new_distinct_stream_on_a_device() {
    gpu_context();
    let default_stream = MlxStream::default_gpu().unwrap();

    let new_stream = MlxStream::on_device(&gpu_device()).unwrap();

    assert!(!new_stream.equals(&default_stream));
    assert!(new_stream.device().unwrap().equals(&gpu_device()));
    assert_ne!(new_stream.index().unwrap(), default_stream.index().unwrap());
}

#[test]
fn should_list_a_newly_created_stream() {
    gpu_context();
    let new_stream = MlxStream::on_device(&gpu_device()).unwrap();

    let all_streams = MlxStream::all_streams().unwrap();

    let mut is_listed = false;
    for stream_index in 0..all_streams.len() {
        if all_streams
            .stream_at(stream_index)
            .unwrap()
            .equals(&new_stream)
        {
            is_listed = true;
        }
    }
    assert!(is_listed);
}

#[test]
fn should_copy_a_stream_handle_through_set() {
    gpu_context();
    let cpu_stream = MlxStream::default_cpu().unwrap();
    let mut copied_stream = MlxStream::default_gpu().unwrap();

    copied_stream.set(&cpu_stream).unwrap();

    assert!(copied_stream.equals(&cpu_stream));
}

#[test]
fn should_replace_and_restore_the_default_stream_of_a_device() {
    gpu_context();
    let original_default = MlxStream::default_for_device(&gpu_device()).unwrap();
    let replacement = MlxStream::on_device(&gpu_device()).unwrap();

    set_default_stream(&replacement).unwrap();
    let observed_default = MlxStream::default_for_device(&gpu_device()).unwrap();
    set_default_stream(&original_default).unwrap();

    assert!(observed_default.equals(&replacement));
    assert!(
        MlxStream::default_for_device(&gpu_device())
            .unwrap()
            .equals(&original_default)
    );
}

#[test]
fn should_compute_the_same_values_on_explicit_cpu_and_gpu_streams() {
    gpu_context();
    let gpu_bound = MlxBindingsContext::new().unwrap();
    let cpu_bound = MlxBindingsContext::on_cpu().unwrap();
    let values = float_array(&[1.0, 2.0], &[2]);

    let on_gpu = gpu_bound.add(&values, &values).unwrap();
    let on_cpu = cpu_bound.add(&values, &values).unwrap();

    assert!(
        gpu_bound
            .gpu_stream()
            .device()
            .unwrap()
            .equals(&gpu_device())
    );
    assert!(
        cpu_bound
            .gpu_stream()
            .device()
            .unwrap()
            .equals(&cpu_device())
    );
    assert_eq!(floats_of(&on_gpu), floats_of(&on_cpu));
}

#[test]
fn should_keep_one_thread_local_stream_stable_on_its_thread() {
    gpu_context();
    let slot = MlxStreamThreadLocal::for_device(&gpu_device()).unwrap();

    let first_read = slot.stream().unwrap();
    let second_read = slot.stream().unwrap();

    assert!(first_read.equals(&second_read));
    assert!(first_read.device().unwrap().equals(&gpu_device()));
    slot.synchronize().unwrap();
}

#[test]
fn should_copy_a_thread_local_slot_through_set() {
    gpu_context();
    let cpu_slot = MlxStreamThreadLocal::for_device(&cpu_device()).unwrap();
    let mut copied_slot = MlxStreamThreadLocal::for_device(&gpu_device()).unwrap();

    copied_slot.set(&cpu_slot).unwrap();

    assert!(
        copied_slot
            .stream()
            .unwrap()
            .equals(&cpu_slot.stream().unwrap())
    );
}

fn compute_on_a_fresh_thread() -> Vec<f32> {
    let context = gpu_context();
    let values = float_array(&[2.0, 3.0], &[2]);
    let doubled = context
        .add(&values, &values)
        .expect("the graph should build");
    floats_of(&doubled)
}

#[test]
fn should_compute_independently_on_a_spawned_thread() {
    gpu_context();

    let spawned = std::thread::spawn(compute_on_a_fresh_thread);

    assert_eq!(
        spawned.join().expect("the thread should finish"),
        vec![4.0, 6.0]
    );
}

fn impossible_reshape_fails_on_this_thread() -> bool {
    let context = gpu_context();
    let values = float_array(&[1.0], &[1]);
    context.reshape(&values, &[9]).is_err()
}

#[test]
fn should_keep_a_spawned_thread_failure_from_reaching_this_thread() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    let spawned = std::thread::spawn(impossible_reshape_fails_on_this_thread);

    assert!(spawned.join().expect("the thread should finish"));
    assert_eq!(
        floats_of(&context.add(&values, &values).unwrap()),
        vec![2.0, 4.0]
    );
}

#[test]
fn should_keep_a_graph_valid_after_its_inputs_drop() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    let doubled = context.add(&values, &values).unwrap();
    drop(values);

    assert_eq!(floats_of(&doubled), vec![2.0, 4.0]);
}

#[test]
fn should_evaluate_several_arrays_together() {
    let context = gpu_context();
    let values = float_array(&[1.0], &[1]);
    let doubled = context.add(&values, &values).unwrap();
    let tripled = context.multiply_scalar(&values, 3.0).unwrap();

    context.evaluate_arrays(&[&doubled, &tripled]).unwrap();

    assert_eq!(floats_of(&doubled), vec![2.0]);
    assert_eq!(floats_of(&tripled), vec![3.0]);
}

#[test]
fn should_evaluate_an_empty_array_list_as_a_no_op() {
    let context = gpu_context();

    context.evaluate_arrays(&[]).unwrap();
    context.async_eval_arrays(&[]).unwrap();
}

#[test]
fn should_read_final_values_after_asynchronous_evaluation_and_synchronization() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);
    let doubled = context.add(&values, &values).unwrap();

    context.async_eval_arrays(&[&doubled]).unwrap();
    synchronize(context.stream()).unwrap();
    context.stream().synchronize().unwrap();
    synchronize_all_defaults().unwrap();

    assert_eq!(floats_of(&doubled), vec![2.0, 4.0]);
}

#[test]
fn should_submit_work_to_a_context_bound_to_a_custom_stream() {
    gpu_context();
    let custom_stream = MlxStream::on_device(&gpu_device()).unwrap();
    let expected_index = custom_stream.index().unwrap();
    let context = MlxBindingsContext::on_stream(custom_stream).unwrap();
    let values = float_array(&[1.0, 2.0], &[2]);

    let doubled = context.add(&values, &values).unwrap();

    assert_eq!(context.stream().index().unwrap(), expected_index);
    assert_eq!(floats_of(&doubled), vec![2.0, 4.0]);
}

#[test]
fn should_compute_on_a_stream_registered_outside_any_thread() {
    gpu_context();
    // SAFETY: This test is the stream's only user, so nothing can race on it.
    let unsynchronized_stream = unsafe { MlxStream::new_thread_unsafe(&gpu_device()) }.unwrap();
    let context = MlxBindingsContext::on_stream(unsynchronized_stream).unwrap();
    let values = float_array(&[3.0], &[1]);

    assert_eq!(
        floats_of(&context.add(&values, &values).unwrap()),
        vec![6.0]
    );
}

#[test]
fn should_find_an_existing_default_stream_without_creating_one() {
    gpu_context();
    let default_stream = MlxStream::default_for_device(&gpu_device()).unwrap();

    let peeked = MlxStream::peek_default_for_device(&gpu_device())
        .unwrap()
        .expect("the default stream exists on this thread");

    assert!(peeked.equals(&default_stream));
}

fn peek_on_a_fresh_thread_finds_nothing() -> bool {
    // Only the handler: building a context would create the default stream.
    mlx_c_rust::install_non_terminating_error_handler();
    MlxStream::peek_default_for_device(&gpu_device())
        .expect("the peek should work")
        .is_none()
}

#[test]
fn should_report_no_default_stream_on_a_thread_that_never_created_one() {
    gpu_context();

    let spawned = std::thread::spawn(peek_on_a_fresh_thread_finds_nothing);

    assert!(spawned.join().expect("the thread should finish"));
}
