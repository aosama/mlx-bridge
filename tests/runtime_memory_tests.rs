//! Tier 1 leak probe and Tier 5 laziness evidence through the crate's own
//! memory counters. Allocations here are deliberately much larger than the
//! allocator's bookkeeping, so the comparisons measure the arrays themselves.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{array_at, gpu_context, vector_of};
use mlx_c_rust::{
    MlxArray, MlxBindingsContext, MlxCError, MlxClosure, MlxDtype, MlxMapStringToArray,
    active_memory_bytes, array_buffer_size_bytes, cache_memory_bytes, clear_allocator_cache,
    memory_limit_bytes, peak_memory_bytes, reset_peak_memory, set_cache_limit, set_memory_limit,
    synchronize,
};

/// 64 MiB of float32 elements.
const PROBE_ELEMENT_COUNT: i32 = 16 * 1024 * 1024;
const PROBE_BYTE_COUNT: usize = 64 * 1024 * 1024;
const LEAK_PROBE_CYCLE_COUNT: usize = 32;

fn settle_active_memory(context: &MlxBindingsContext) -> usize {
    synchronize(context.stream()).expect("the stream should synchronize");
    clear_allocator_cache().expect("the cache should clear");
    active_memory_bytes().expect("the counter should read")
}

fn allocate_evaluate_and_release(context: &MlxBindingsContext) {
    let values = context
        .zeros(&[PROBE_ELEMENT_COUNT], MlxDtype::Float32)
        .expect("the graph should build");
    let doubled = context
        .add(&values, &values)
        .expect("the graph should build");
    doubled.evaluate().expect("the graph should evaluate");
}

fn first_input_unchanged(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    Ok(vec![inputs[0].retain()?])
}

/// Holds the probe through every handle family that retains an array, so a
/// leak in any of their releases keeps the whole allocation alive and shows.
fn route_through_every_retaining_handle_and_release(context: &MlxBindingsContext) {
    let values = context
        .full(&[PROBE_ELEMENT_COUNT], 1.0, MlxDtype::Float32)
        .expect("the graph should build");
    values.evaluate().expect("the array should evaluate");

    let retained = values.retain().expect("the array should retain");
    let vector = vector_of(&[&values, &retained]);
    let element = array_at(&vector, 0);
    let mut map = MlxMapStringToArray::new().expect("the map should build");
    map.insert("probe", &element)
        .expect("the insert should work");
    let stored = map.value_at("probe").unwrap().expect("the key was stored");
    let mut iterator = map.iter().expect("the iterator should build");
    let iterated = iterator.next_entry().unwrap().expect("one entry exists");
    let closure = MlxClosure::from_fn(first_input_unchanged).expect("the closure should build");
    let outputs = closure
        .apply(&vector_of(&[&stored]))
        .expect("apply should work");

    array_at(&outputs, 0)
        .evaluate()
        .expect("the output should evaluate");
    iterated.1.evaluate().expect("the entry should evaluate");
}

#[test]
fn should_keep_active_memory_flat_across_create_evaluate_drop_cycles() {
    let context = gpu_context();
    allocate_evaluate_and_release(&context);
    let settled_baseline = settle_active_memory(&context);

    for _cycle_number in 0..LEAK_PROBE_CYCLE_COUNT {
        allocate_evaluate_and_release(&context);
    }

    let settled_after_cycles = settle_active_memory(&context);
    assert!(
        settled_after_cycles < settled_baseline + PROBE_BYTE_COUNT,
        "active memory grew from {settled_baseline} to {settled_after_cycles} bytes"
    );
}

#[test]
fn should_release_an_array_held_through_every_retaining_handle() {
    let context = gpu_context();
    route_through_every_retaining_handle_and_release(&context);
    let settled_baseline = settle_active_memory(&context);

    for _cycle_number in 0..LEAK_PROBE_CYCLE_COUNT {
        route_through_every_retaining_handle_and_release(&context);
    }

    let settled_after_cycles = settle_active_memory(&context);
    // A single leaked probe would exceed this margin, and repeating the cycle
    // would multiply it: the total allocated is 32 times the margin.
    assert!(
        settled_after_cycles < settled_baseline + PROBE_BYTE_COUNT,
        "active memory grew from {settled_baseline} to {settled_after_cycles} bytes"
    );
}

#[test]
fn should_count_an_evaluated_array_while_it_lives_and_release_it_on_drop() {
    let context = gpu_context();
    let baseline = settle_active_memory(&context);

    let values = context
        .full(&[PROBE_ELEMENT_COUNT], 1.0, MlxDtype::Float32)
        .expect("the graph should build");
    values.evaluate().expect("the array should evaluate");
    let while_alive = active_memory_bytes().unwrap();

    drop(values);
    let after_drop = settle_active_memory(&context);

    assert!(
        while_alive >= baseline + PROBE_BYTE_COUNT,
        "the probe must be visible: {baseline} -> {while_alive}"
    );
    assert!(after_drop < while_alive - PROBE_BYTE_COUNT / 2);
}

#[test]
fn should_allocate_no_array_storage_until_a_graph_is_evaluated() {
    let context = gpu_context();
    let baseline = settle_active_memory(&context);

    let values = context
        .full(&[PROBE_ELEMENT_COUNT], 1.0, MlxDtype::Float32)
        .expect("the graph should build");
    let lazy_doubled = context
        .add(&values, &values)
        .expect("the graph should build");
    let before_evaluation = active_memory_bytes().unwrap();

    lazy_doubled.evaluate().expect("the graph should evaluate");
    let after_evaluation = active_memory_bytes().unwrap();

    assert!(
        before_evaluation < baseline + PROBE_BYTE_COUNT / 2,
        "building the graph must not materialize it: {baseline} -> {before_evaluation}"
    );
    assert!(after_evaluation >= baseline + PROBE_BYTE_COUNT);
}

#[test]
fn should_record_and_reset_the_peak() {
    let context = gpu_context();
    allocate_evaluate_and_release(&context);
    settle_active_memory(&context);

    assert!(peak_memory_bytes().unwrap() >= PROBE_BYTE_COUNT);

    reset_peak_memory().unwrap();

    assert!(peak_memory_bytes().unwrap() < PROBE_BYTE_COUNT);
}

#[test]
fn should_return_the_previous_limit_when_the_memory_limit_changes() {
    gpu_context();
    let original_limit = memory_limit_bytes().unwrap();

    let previous_limit = set_memory_limit(original_limit / 2).unwrap();
    let lowered_limit = memory_limit_bytes().unwrap();
    set_memory_limit(original_limit).unwrap();

    assert_eq!(previous_limit, original_limit);
    assert_eq!(lowered_limit, original_limit / 2);
    assert_eq!(memory_limit_bytes().unwrap(), original_limit);
}

#[test]
fn should_return_the_previous_limit_when_the_cache_limit_changes() {
    gpu_context();
    const FIRST_CACHE_LIMIT_BYTES: usize = 1024 * 1024;
    const SECOND_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024;
    let original_limit = set_cache_limit(FIRST_CACHE_LIMIT_BYTES).unwrap();

    let previous_limit = set_cache_limit(SECOND_CACHE_LIMIT_BYTES).unwrap();
    set_cache_limit(original_limit).unwrap();

    assert_eq!(previous_limit, FIRST_CACHE_LIMIT_BYTES);
}

#[test]
fn should_empty_the_allocator_cache_when_cleared() {
    let context = gpu_context();
    allocate_evaluate_and_release(&context);

    synchronize(context.stream()).unwrap();
    clear_allocator_cache().unwrap();

    assert_eq!(cache_memory_bytes().unwrap(), 0);
}

#[test]
fn should_report_the_buffer_bytes_of_evaluated_arrays() {
    gpu_context();
    let first = MlxArray::from_f32(&[1.0, 2.0], &[2]).expect("the fixture should be valid");
    let second = MlxArray::from_f32(&[3.0], &[1]).expect("the fixture should be valid");
    first.evaluate().expect("the array should evaluate");
    second.evaluate().expect("the array should evaluate");

    assert_eq!(array_buffer_size_bytes(&vector_of(&[&first])).unwrap(), 8);
    assert_eq!(
        array_buffer_size_bytes(&vector_of(&[&first, &second])).unwrap(),
        12
    );
}

#[test]
fn should_report_no_buffer_bytes_for_an_empty_vector() {
    gpu_context();

    assert_eq!(array_buffer_size_bytes(&vector_of(&[])).unwrap(), 0);
}
