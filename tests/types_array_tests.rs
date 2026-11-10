//! Tier 1 ownership contracts for `MlxArray`: handle independence, copy-out
//! semantics, and host reads that stay inside the array's allocation whatever
//! its element type or storage layout.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    assert_floats_close, bools_of, float_array, floats_of, gpu_context, int_array, ints_of,
};
use mlx_bridge::{MlxArray, MlxDtype};

#[test]
fn should_keep_a_retained_handle_readable_after_its_source_drops() {
    let source = float_array(&[1.0, 2.0, 3.0], &[3]);
    let retained = source.retain().expect("a live array should retain");

    drop(source);

    assert_eq!(floats_of(&retained), vec![1.0, 2.0, 3.0]);
}

#[test]
fn should_read_the_same_values_through_source_and_retained_handles() {
    let source = float_array(&[4.0, 5.0], &[2]);
    let retained = source.retain().expect("a live array should retain");

    assert_eq!(floats_of(&source), floats_of(&retained));
    assert_eq!(source.shape(), retained.shape());
    assert_eq!(source.dtype(), retained.dtype());
}

#[test]
fn should_leave_the_source_untouched_when_a_retained_handle_is_replaced() {
    let source = float_array(&[7.0], &[1]);
    let mut retained = source.retain().expect("a live array should retain");

    retained
        .set_f32(9.0)
        .expect("the scalar setter should work");

    assert_eq!(floats_of(&source), vec![7.0]);
    assert_eq!(floats_of(&retained), vec![9.0]);
}

#[test]
fn should_snapshot_host_values_when_an_array_is_constructed() {
    let mut host_values = vec![1.0_f32, 2.0];
    let array = float_array(&host_values, &[2]);

    host_values[0] = 99.0;

    assert_eq!(floats_of(&array), vec![1.0, 2.0]);
}

#[test]
fn should_return_host_copies_that_do_not_alias_array_storage() {
    let array = float_array(&[1.0, 2.0], &[2]);

    let mut first_copy = floats_of(&array);
    first_copy[0] = 99.0;

    assert_eq!(floats_of(&array), vec![1.0, 2.0]);
}

#[test]
fn should_keep_array_metadata_and_values_after_its_context_drops() {
    let context = gpu_context();
    let array = context
        .multiply_scalar(&float_array(&[1.0, 2.0], &[2]), 3.0)
        .expect("the graph should build");

    drop(context);

    assert_eq!(array.shape(), vec![2]);
    assert_eq!(array.dtype(), MlxDtype::Float32);
    assert_eq!(floats_of(&array), vec![3.0, 6.0]);
}

#[test]
fn should_treat_an_empty_shape_as_a_single_scalar() {
    let scalar = float_array(&[2.5], &[]);

    assert_eq!(scalar.shape(), Vec::<i32>::new());
    assert_eq!(scalar.element_count(), 1);
    assert_eq!(floats_of(&scalar), vec![2.5]);
}

#[test]
fn should_read_a_zero_sized_array_as_an_empty_vector() {
    let zero_sized = float_array(&[], &[0]);

    assert!(!zero_sized.is_empty(), "zero-sized is not an empty handle");
    assert_eq!(zero_sized.element_count(), 0);
    assert_eq!(floats_of(&zero_sized), Vec::<f32>::new());
}

#[test]
fn should_reject_a_shape_that_disagrees_with_the_value_count() {
    let failure =
        MlxArray::from_f32(&[1.0, 2.0, 3.0], &[2, 2]).expect_err("3 values cannot fill 2x2");

    assert_eq!(failure.operation, "create an MLX array");
}

#[test]
fn should_reject_a_negative_dimension() {
    assert!(MlxArray::from_i32(&[1, 2], &[-1, -2]).is_err());
    assert!(MlxArray::from_u32(&[1, 2], &[-2]).is_err());
}

#[test]
fn should_report_an_empty_handle_and_refuse_every_read_of_it() {
    gpu_context();
    let empty = MlxArray::empty();

    assert!(empty.is_empty());
    assert!(empty.evaluate().is_err());
    assert!(empty.retain().is_err());
    assert!(empty.to_vec_f32().is_err());
    assert!(empty.to_vec_bool().is_err());
    assert!(empty.item_u32().is_err());
    assert!(empty.item_bool().is_err());
    assert!(empty.is_row_contiguous().is_err());
    assert!(empty.require_populated("read an empty handle").is_err());
}

#[test]
fn should_refuse_a_wider_read_than_the_element_type_stores() {
    let single_precision = float_array(&[1.0, 2.0], &[2]);

    // A float64 read of float32 storage would run past the allocation.
    let failure = single_precision
        .to_vec_f64()
        .expect_err("float32 storage cannot satisfy a float64 read");

    assert_eq!(failure.description, "array dtype is Float32, not Float64");
}

#[test]
fn should_refuse_every_vector_read_with_the_wrong_element_type() {
    let single_precision = float_array(&[1.5], &[1]);

    assert!(single_precision.to_vec_i32().is_err());
    assert!(single_precision.to_vec_i64().is_err());
    assert!(single_precision.to_vec_i16().is_err());
    assert!(single_precision.to_vec_i8().is_err());
    assert!(single_precision.to_vec_u8().is_err());
    assert!(single_precision.to_vec_u16().is_err());
    assert!(single_precision.to_vec_u32().is_err());
    assert!(single_precision.to_vec_u64().is_err());
    assert!(single_precision.to_vec_bool().is_err());
    assert!(single_precision.to_vec_f16_bits().is_err());
    assert!(single_precision.to_vec_bf16_bits().is_err());
    assert!(int_array(&[1], &[1]).to_vec_f32().is_err());
}

#[test]
fn should_refuse_every_scalar_read_with_the_wrong_element_type() {
    let single_precision = float_array(&[1.5], &[1]);

    // MLX's item<T>() reinterprets rather than converts: an unguarded uint32
    // read of 1.5 returns 1069547520, the float's raw bits.
    assert!(single_precision.item_u32().is_err());
    assert!(single_precision.item_f64().is_err());
    assert!(MlxArray::from_scalar_i32(1).item_f32().is_err());
    assert!(single_precision.item_i64().is_err());
    assert!(single_precision.item_i32().is_err());
    assert!(single_precision.item_i16().is_err());
    assert!(single_precision.item_i8().is_err());
    assert!(single_precision.item_u8().is_err());
    assert!(single_precision.item_u16().is_err());
    assert!(single_precision.item_u64().is_err());
    assert!(single_precision.item_bool().is_err());
    assert!(single_precision.item_complex().is_err());
    assert!(single_precision.item_f16_bits().is_err());
    assert!(single_precision.item_bf16_bits().is_err());
}

#[test]
fn should_refuse_a_scalar_read_of_a_multiple_element_array() {
    gpu_context();
    let failure = MlxArray::from_u32(&[1, 2], &[2])
        .expect("the fixture should be valid")
        .item_u32()
        .expect_err("two elements are not a scalar");

    assert!(failure.description.contains("size 1"), "{failure}");
}

#[test]
fn should_read_a_reversed_view_in_its_logical_order() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);
    values.evaluate().expect("the parent should evaluate");

    let reversed = context.flip(&values).expect("flip should build");

    // The reversed view's data pointer addresses the buffer's last element,
    // so a raw forward read returned [4, 0, 0, 0] from beyond the allocation.
    assert_eq!(floats_of(&reversed), vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn should_read_a_transposed_view_in_its_logical_order() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let transposed = context.transpose(&values).expect("transpose should build");

    assert_eq!(transposed.shape(), vec![3, 2]);
    assert_eq!(floats_of(&transposed), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn should_read_a_strided_slice_without_its_skipped_elements() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
    values.evaluate().expect("the parent should evaluate");

    let every_second = context
        .slice(&values, &[0], &[6], &[2])
        .expect("the strided slice should build");

    assert_eq!(floats_of(&every_second), vec![1.0, 3.0, 5.0]);
}

#[test]
fn should_read_a_contiguous_offset_slice_from_its_own_start() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);
    values.evaluate().expect("the parent should evaluate");

    let tail = context
        .slice(&values, &[2], &[4], &[1])
        .expect("the offset slice should build");

    assert_eq!(floats_of(&tail), vec![3.0, 4.0]);
}

#[test]
fn should_read_reversed_views_of_integer_element_types() {
    let context = gpu_context();
    let signed_values = int_array(&[1, 2, 3], &[3]);
    let unsigned_values = MlxArray::from_u32(&[1, 2, 3], &[3]).expect("the fixture is valid");

    let reversed_signed = context.flip(&signed_values).expect("flip should build");
    let reversed_unsigned = context.flip(&unsigned_values).expect("flip should build");
    let bytes = context
        .astype(&unsigned_values, MlxDtype::UInt8)
        .expect("the cast should build");
    let reversed_bytes = context.flip(&bytes).expect("flip should build");

    assert_eq!(ints_of(&reversed_signed), vec![3, 2, 1]);
    assert_eq!(reversed_unsigned.to_vec_u32().unwrap(), vec![3, 2, 1]);
    assert_eq!(reversed_bytes.to_vec_u8().unwrap(), vec![3, 2, 1]);
}

#[test]
fn should_report_contiguity_for_views_and_their_materializations() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let transposed = context.transpose(&values).expect("transpose should build");
    let materialized = context
        .build_contiguous_row_major_copy(&transposed)
        .expect("the copy should build");

    assert!(values.is_row_contiguous().unwrap());
    assert!(!transposed.is_row_contiguous().unwrap());
    assert!(materialized.is_row_contiguous().unwrap());
}

#[test]
fn should_refuse_a_direct_transfer_of_a_view_and_materialize_through_the_context() {
    let context = gpu_context();
    let values = MlxArray::from_u32(&[1, 2, 3], &[3]).expect("the fixture is valid");

    let reversed = context.flip(&values).expect("flip should build");
    reversed.evaluate().expect("the view should evaluate");

    let failure = reversed
        .copy_evaluated_u32_values()
        .expect_err("a view has no single host transfer");
    assert!(failure.description.contains("non-contiguous"), "{failure}");
    assert_eq!(context.copy_u32_values(&reversed).unwrap(), vec![3, 2, 1]);
}

#[test]
fn should_transfer_an_evaluated_contiguous_array_directly() {
    let values = MlxArray::from_u32(&[7, 8], &[2]).expect("the fixture is valid");
    values.evaluate().expect("the array should evaluate");

    assert_eq!(values.copy_evaluated_u32_values().unwrap(), vec![7, 8]);
}

#[test]
fn should_refuse_a_direct_byte_transfer_of_a_view_and_of_the_wrong_element_type() {
    let context = gpu_context();
    let bytes = context
        .astype(&int_array(&[1, 2], &[2]), MlxDtype::UInt8)
        .expect("the cast should build");
    let reversed = context.flip(&bytes).expect("flip should build");

    assert!(reversed.copy_evaluated_u8_values().is_err());
    assert!(
        float_array(&[1.0], &[1])
            .copy_evaluated_u8_values()
            .is_err()
    );
    assert_eq!(bytes.copy_evaluated_u8_values().unwrap(), vec![1, 2]);
    assert_eq!(context.copy_u8_values(&reversed).unwrap(), vec![2, 1]);
}

#[test]
fn should_transfer_a_zero_sized_array_as_an_empty_vector() {
    let zero_sized = MlxArray::from_u32(&[], &[0]).expect("the fixture is valid");

    assert_eq!(
        zero_sized.copy_evaluated_u32_values().unwrap(),
        Vec::<u32>::new()
    );
}

#[test]
fn should_round_trip_every_scalar_constructor() {
    assert!(MlxArray::from_bool(true).item_bool().unwrap());
    assert_eq!(MlxArray::from_scalar_i32(-3).item_i32().unwrap(), -3);
    assert_eq!(MlxArray::from_scalar_f32(1.5).item_f32().unwrap(), 1.5);
    assert_eq!(MlxArray::from_scalar_f64(2.5).item_f64().unwrap(), 2.5);
    assert_eq!(
        MlxArray::from_scalar_complex(1.0, -2.0)
            .item_complex()
            .unwrap(),
        (1.0, -2.0)
    );
}

#[test]
fn should_replace_a_value_and_its_element_type_through_the_setters() {
    let mut array = MlxArray::from_scalar_f32(1.0);

    array.set_i32(4).expect("the setter should work");
    assert_eq!(array.item_i32().unwrap(), 4);
    array.set_bool(true).expect("the setter should work");
    assert!(array.item_bool().unwrap());
    array.set_f64(0.5).expect("the setter should work");
    assert_eq!(array.item_f64().unwrap(), 0.5);
    array.set_complex(3.0, 4.0).expect("the setter should work");
    assert_eq!(array.item_complex().unwrap(), (3.0, 4.0));
}

#[test]
fn should_round_trip_integer_element_types_through_their_typed_readers() {
    let context = gpu_context();
    let source = int_array(&[1, 2], &[2]);

    let bytes = context.astype(&source, MlxDtype::UInt8).unwrap();
    let shorts = context.astype(&source, MlxDtype::UInt16).unwrap();
    let longs = context.astype(&source, MlxDtype::UInt64).unwrap();
    let signed_bytes = context.astype(&source, MlxDtype::Int8).unwrap();
    let signed_shorts = context.astype(&source, MlxDtype::Int16).unwrap();
    let signed_longs = context.astype(&source, MlxDtype::Int64).unwrap();

    assert_eq!(bytes.to_vec_u8().unwrap(), vec![1, 2]);
    assert_eq!(shorts.to_vec_u16().unwrap(), vec![1, 2]);
    assert_eq!(longs.to_vec_u64().unwrap(), vec![1, 2]);
    assert_eq!(signed_bytes.to_vec_i8().unwrap(), vec![1, 2]);
    assert_eq!(signed_shorts.to_vec_i16().unwrap(), vec![1, 2]);
    assert_eq!(signed_longs.to_vec_i64().unwrap(), vec![1, 2]);
}

#[test]
fn should_read_half_precision_values_as_their_raw_bits() {
    let context = gpu_context();
    let one = float_array(&[1.0], &[1]);

    let half = context.astype(&one, MlxDtype::Float16).unwrap();
    let brain_float = context.astype(&one, MlxDtype::BFloat16).unwrap();

    // IEEE binary16 and bfloat16 encodings of 1.0.
    assert_eq!(half.to_vec_f16_bits().unwrap(), vec![0x3C00]);
    assert_eq!(half.item_f16_bits().unwrap(), 0x3C00);
    assert_eq!(brain_float.to_vec_bf16_bits().unwrap(), vec![0x3F80]);
    assert_eq!(brain_float.item_bf16_bits().unwrap(), 0x3F80);
}

#[test]
fn should_read_a_boolean_comparison_result() {
    let context = gpu_context();
    let left = float_array(&[1.0, 2.0], &[2]);
    let right = float_array(&[1.0, 3.0], &[2]);

    let matches = context.equal(&left, &right).expect("equal should build");

    assert_eq!(matches.dtype(), MlxDtype::Bool);
    assert_eq!(bools_of(&matches), vec![true, false]);
}

#[test]
fn should_report_sizes_consistently_with_the_element_type() {
    let values = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    assert_eq!(values.element_count(), 6);
    assert_eq!(values.item_size(), 4);
    assert_eq!(values.byte_count(), 24);
    assert_eq!(values.strides(), vec![3, 1]);
    assert_eq!(MlxArray::byte_size_of_dtype(MlxDtype::Float64), 8);
    assert_eq!(MlxArray::byte_size_of_dtype(MlxDtype::Bool), 1);
    assert_eq!(MlxArray::byte_size_of_dtype(MlxDtype::Complex64), 8);
}

#[test]
fn should_resolve_dimensions_from_either_end_and_report_zero_out_of_range() {
    gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    assert_eq!(values.dimension(0), 2);
    assert_eq!(values.dimension(-1), 3);
    assert_eq!(values.dimension(7), 0);
}

#[test]
fn should_take_ownership_of_a_managed_buffer_for_the_array_lifetime() {
    let mut host_bytes = Vec::new();
    for host_value in [1.0_f32, 2.0] {
        host_bytes.extend_from_slice(&host_value.to_ne_bytes());
    }

    // SAFETY: The buffer moves into the array and holds exactly two float32
    // values, matching the shape and element type below.
    let managed = unsafe { MlxArray::from_managed_buffer(host_bytes, &[2], MlxDtype::Float32) }
        .expect("the managed buffer should wrap");

    assert_floats_close(&floats_of(&managed), &[1.0, 2.0], 0.0);
}

#[test]
fn should_reject_a_managed_buffer_whose_size_disagrees_with_the_shape() {
    // SAFETY: The size check rejects this buffer before MLX ever sees it.
    let failure = unsafe { MlxArray::from_managed_buffer(vec![0_u8; 4], &[2], MlxDtype::Float32) }
        .expect_err("four bytes cannot hold two float32 values");

    assert!(failure.description.contains("byte count"), "{failure}");
}

#[test]
fn should_report_a_reversed_view_with_a_negative_stride() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);
    values.evaluate().expect("the parent should evaluate");

    let reversed = context.flip(&values).expect("flip should build");
    reversed.evaluate().expect("the view should evaluate");

    assert_eq!(reversed.strides(), vec![-1]);
}
