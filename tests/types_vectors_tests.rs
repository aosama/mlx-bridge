//! Tier 1 contracts for the typed vector handles: integer, string, stream,
//! and nested array vectors, including empty edges and UTF-8 round trips.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, gpu_context, vector_of};
use mlx_bridge::{MlxStream, MlxVectorInt, MlxVectorStream, MlxVectorString, MlxVectorVectorArray};

#[test]
fn should_round_trip_integers_in_order() {
    let vector = MlxVectorInt::from_values(&[3, -1, 7]).expect("the vector should build");

    assert_eq!(vector.len(), 3);
    assert_eq!(vector.value_at(0).unwrap(), 3);
    assert_eq!(vector.value_at(1).unwrap(), -1);
    assert_eq!(vector.value_at(2).unwrap(), 7);
}

#[test]
fn should_report_empty_integer_vectors_built_either_way() {
    assert!(MlxVectorInt::empty().is_empty());
    assert!(MlxVectorInt::from_values(&[]).unwrap().is_empty());
}

#[test]
fn should_fail_to_read_past_the_end_of_an_integer_vector() {
    gpu_context();
    let vector = MlxVectorInt::from_value(1).expect("the vector should build");

    assert_eq!(vector.len(), 1);
    assert!(vector.value_at(1).is_err());
}

#[test]
fn should_grow_and_replace_an_integer_vector() {
    let mut vector = MlxVectorInt::from_values(&[1]).expect("the vector should build");

    vector.append_one(2).expect("the append should work");
    vector
        .append_values(&[3, 4])
        .expect("the append should work");
    assert_eq!(vector.len(), 4);
    assert_eq!(vector.value_at(3).unwrap(), 4);

    vector
        .set_values(&[9])
        .expect("the replacement should work");
    assert_eq!(vector.len(), 1);

    vector.set_one(8).expect("the replacement should work");
    assert_eq!(vector.value_at(0).unwrap(), 8);
}

#[test]
fn should_copy_an_integer_vector_independently_through_set() {
    let source = MlxVectorInt::from_values(&[1, 2]).expect("the vector should build");
    let mut copy = MlxVectorInt::empty();

    copy.set(&source).expect("the copy should work");
    copy.append_one(3).expect("the append should work");

    assert_eq!(copy.len(), 3);
    assert_eq!(source.len(), 2);
}

#[test]
fn should_round_trip_utf8_strings_in_order() {
    let vector =
        MlxVectorString::from_strings(&["plain", "héllo", "日本語", ""]).expect("should build");

    assert_eq!(vector.len(), 4);
    assert_eq!(vector.string_at(0).unwrap(), "plain");
    assert_eq!(vector.string_at(1).unwrap(), "héllo");
    assert_eq!(vector.string_at(2).unwrap(), "日本語");
    assert_eq!(vector.string_at(3).unwrap(), "");
}

#[test]
fn should_reject_a_string_with_an_interior_null_byte() {
    let failure =
        MlxVectorString::from_strings(&["a\0b"]).expect_err("C strings cannot hold a null byte");

    assert!(failure.description.contains("null byte"), "{failure}");
}

#[test]
fn should_reject_an_appended_string_with_an_interior_null_byte() {
    let mut vector = MlxVectorString::empty();

    assert!(vector.append_one("a\0b").is_err());
    assert!(
        vector.is_empty(),
        "a rejected append must not change length"
    );
}

#[test]
fn should_fail_to_read_past_the_end_of_a_string_vector() {
    gpu_context();
    let vector = MlxVectorString::empty();

    assert!(vector.is_empty());
    assert!(vector.string_at(0).is_err());
}

#[test]
fn should_grow_and_replace_a_string_vector() {
    let mut vector = MlxVectorString::empty();

    vector.append_one("first").expect("the append should work");
    vector
        .set_strings(&["a", "b"])
        .expect("the set should work");
    assert_eq!(vector.len(), 2);

    vector.set_one("only").expect("the set should work");
    assert_eq!(vector.len(), 1);
    assert_eq!(vector.string_at(0).unwrap(), "only");
}

#[test]
fn should_copy_a_string_vector_independently_through_set() {
    let source = MlxVectorString::from_strings(&["kept"]).expect("should build");
    let mut copy = MlxVectorString::empty();

    copy.set(&source).expect("the copy should work");
    copy.append_one("extra").expect("the append should work");

    assert_eq!(source.len(), 1);
    assert_eq!(copy.len(), 2);
}

#[test]
fn should_round_trip_streams_by_identity() {
    let gpu_stream = MlxStream::default_gpu().expect("the GPU stream should exist");
    let cpu_stream = MlxStream::default_cpu().expect("the CPU stream should exist");

    let mut vector =
        MlxVectorStream::from_streams(&[&gpu_stream]).expect("the vector should build");
    vector
        .append_stream(&cpu_stream)
        .expect("the append should work");

    assert_eq!(vector.len(), 2);
    assert!(vector.stream_at(0).unwrap().equals(&gpu_stream));
    assert!(vector.stream_at(1).unwrap().equals(&cpu_stream));
}

#[test]
fn should_fail_to_read_any_index_of_an_empty_stream_vector() {
    gpu_context();
    let vector = MlxVectorStream::empty();

    assert!(vector.is_empty());
    assert!(vector.stream_at(0).is_err());
}

#[test]
fn should_copy_a_stream_vector_through_set() {
    let gpu_stream = MlxStream::default_gpu().expect("the GPU stream should exist");
    let source = MlxVectorStream::from_streams(&[&gpu_stream]).expect("should build");
    let mut copy = MlxVectorStream::empty();

    copy.set(&source).expect("the copy should work");

    assert_eq!(copy.len(), 1);
}

#[test]
fn should_round_trip_nested_array_vectors() {
    let array = float_array(&[1.0], &[1]);
    let inner_vectors = [vector_of(&[&array, &array]), vector_of(&[])];

    let nested = MlxVectorVectorArray::from_vectors(&inner_vectors).expect("should build");

    assert_eq!(nested.len(), 2);
    assert_eq!(nested.vector_at(0).unwrap().len(), 2);
    assert!(nested.vector_at(1).unwrap().is_empty());
}

#[test]
fn should_fail_to_read_any_index_of_an_empty_nested_vector() {
    gpu_context();
    let nested = MlxVectorVectorArray::empty();

    assert!(nested.is_empty());
    assert!(nested.vector_at(0).is_err());
}

#[test]
fn should_copy_a_nested_vector_through_set() {
    let source = MlxVectorVectorArray::from_vectors(&[vector_of(&[])]).expect("should build");
    let mut copy = MlxVectorVectorArray::empty();

    copy.set(&source).expect("the copy should work");

    assert_eq!(copy.len(), 1);
}

#[test]
fn should_build_and_grow_a_string_vector_from_one_and_many_strings() {
    let mut vector = MlxVectorString::from_string("first").expect("should build");

    vector
        .append_strings(&["second", "third"])
        .expect("the append should work");

    assert_eq!(vector.len(), 3);
    assert_eq!(vector.string_at(2).unwrap(), "third");
    assert!(vector.append_strings(&["a\0b"]).is_err());
}

#[test]
fn should_build_replace_and_grow_a_stream_vector() {
    let gpu_stream = MlxStream::default_gpu().expect("the GPU stream should exist");
    let cpu_stream = MlxStream::default_cpu().expect("the CPU stream should exist");

    let mut vector = MlxVectorStream::from_stream(&gpu_stream).expect("should build");
    vector
        .append_streams(&[&cpu_stream, &gpu_stream])
        .expect("the append should work");
    assert_eq!(vector.len(), 3);
    assert!(vector.stream_at(1).unwrap().equals(&cpu_stream));

    vector
        .set_stream(&cpu_stream)
        .expect("the replacement should work");
    assert_eq!(vector.len(), 1);

    vector
        .set_streams(&[&gpu_stream, &cpu_stream])
        .expect("the replacement should work");
    assert_eq!(vector.len(), 2);
}

#[test]
fn should_build_replace_and_grow_a_nested_vector() {
    let array = float_array(&[1.0], &[1]);
    let one_array = vector_of(&[&array]);
    let two_arrays = vector_of(&[&array, &array]);

    let mut nested = MlxVectorVectorArray::from_vector(&one_array).expect("should build");
    nested
        .append_vector(&two_arrays)
        .expect("the append should work");
    assert_eq!(nested.len(), 2);
    assert_eq!(nested.vector_at(1).unwrap().len(), 2);

    nested
        .append_vectors(&[vector_of(&[]), vector_of(&[])])
        .expect("the append should work");
    assert_eq!(nested.len(), 4);

    nested
        .set_vector(&two_arrays)
        .expect("the replacement should work");
    assert_eq!(nested.len(), 1);

    nested
        .set_vectors(&[vector_of(&[]), vector_of(&[])])
        .expect("the replacement should work");
    assert_eq!(nested.len(), 2);
}
