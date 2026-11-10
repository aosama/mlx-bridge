//! Tier 1 contracts for `MlxArrayVector`: construction, the empty edges,
//! ordering, and element handles that outlive the vector.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    array_at, assert_translated_failure, float_array, floats_of, gpu_context, vector_of,
};
use mlx_bridge::{MlxArray, MlxArrayVector};

fn fixture_arrays() -> (MlxArray, MlxArray) {
    (float_array(&[1.0], &[1]), float_array(&[2.0], &[1]))
}

#[test]
fn should_start_empty_and_report_no_arrays() {
    let vector = MlxArrayVector::empty("build an empty vector").expect("the vector should build");

    assert_eq!(vector.len(), 0);
    assert!(vector.is_empty());
}

#[test]
fn should_build_an_empty_vector_from_an_empty_slice() {
    let vector = vector_of(&[]);

    assert!(vector.is_empty());
}

#[test]
fn should_preserve_the_order_of_its_arrays() {
    let (first, second) = fixture_arrays();

    let vector = vector_of(&[&first, &second]);

    assert_eq!(vector.len(), 2);
    assert_eq!(floats_of(&array_at(&vector, 0)), vec![1.0]);
    assert_eq!(floats_of(&array_at(&vector, 1)), vec![2.0]);
}

#[test]
fn should_fail_with_the_named_operation_when_an_index_is_out_of_range() {
    gpu_context();
    let (first, _second) = fixture_arrays();
    let vector = vector_of(&[&first]);

    let failure = vector
        .array_at(1, "read past the vector end")
        .expect_err("index one of a single-array vector is out of range");

    assert_translated_failure(&failure, "read past the vector end");
}

#[test]
fn should_fail_to_read_any_index_of_an_empty_vector() {
    gpu_context();
    let vector = vector_of(&[]);

    assert!(vector.array_at(0, "read an empty vector").is_err());
}

#[test]
fn should_keep_an_element_readable_after_the_vector_drops() {
    let (first, _second) = fixture_arrays();
    let vector = vector_of(&[&first]);

    let element = array_at(&vector, 0);
    drop(vector);
    drop(first);

    assert_eq!(floats_of(&element), vec![1.0]);
}

#[test]
fn should_keep_source_arrays_readable_after_the_vector_drops() {
    let (first, _second) = fixture_arrays();

    drop(vector_of(&[&first]));

    assert_eq!(floats_of(&first), vec![1.0]);
}

#[test]
fn should_hold_one_array_built_through_the_single_value_constructor() {
    let (first, _second) = fixture_arrays();

    let vector = MlxArrayVector::from_array(&first).expect("the vector should build");

    assert_eq!(vector.len(), 1);
}

#[test]
fn should_append_one_array_and_many_arrays_in_call_order() {
    let (first, second) = fixture_arrays();
    let mut vector = vector_of(&[]);

    vector.append_array(&first).expect("the append should work");
    vector
        .append_arrays(&[second, float_array(&[3.0], &[1])])
        .expect("the bulk append should work");

    assert_eq!(vector.len(), 3);
    assert_eq!(floats_of(&array_at(&vector, 2)), vec![3.0]);
}

#[test]
fn should_replace_its_contents_through_the_setters() {
    let (first, second) = fixture_arrays();
    let mut vector = vector_of(&[&first, &second]);

    vector.set_one(&first).expect("the replacement should work");
    assert_eq!(vector.len(), 1);

    vector
        .set_arrays(&[first, second])
        .expect("the bulk replacement should work");
    assert_eq!(vector.len(), 2);

    vector.set_arrays(&[]).expect("clearing should work");
    assert!(vector.is_empty());
}

#[test]
fn should_copy_the_source_contents_independently_through_set() {
    let (first, second) = fixture_arrays();
    let source = vector_of(&[&first]);
    let mut copy = vector_of(&[]);

    copy.set(&source).expect("the copy should work");
    copy.append_array(&second).expect("the append should work");

    assert_eq!(copy.len(), 2);
    assert_eq!(source.len(), 1, "growing the copy must not grow the source");
}
