//! Tier 1 contracts for the typed array constructors: each stores its element
//! type's exact bytes, so every value round-trips through the matching reader.

use mlx_c_rust::{MlxArray, MlxDtype};

#[test]
fn should_round_trip_unsigned_values_including_their_extremes() {
    let bytes = MlxArray::from_u8(&[0, u8::MAX], &[2]).unwrap();
    let shorts = MlxArray::from_u16(&[0, u16::MAX], &[2]).unwrap();
    let longs = MlxArray::from_u64(&[0, u64::MAX], &[2]).unwrap();

    assert_eq!(bytes.dtype(), MlxDtype::UInt8);
    assert_eq!(bytes.to_vec_u8().unwrap(), vec![0, u8::MAX]);
    assert_eq!(shorts.to_vec_u16().unwrap(), vec![0, u16::MAX]);
    assert_eq!(longs.to_vec_u64().unwrap(), vec![0, u64::MAX]);
}

#[test]
fn should_round_trip_signed_values_including_their_extremes() {
    let bytes = MlxArray::from_i8(&[i8::MIN, i8::MAX], &[2]).unwrap();
    let shorts = MlxArray::from_i16(&[i16::MIN, i16::MAX], &[2]).unwrap();
    let longs = MlxArray::from_i64(&[i64::MIN, i64::MAX], &[2]).unwrap();

    assert_eq!(bytes.dtype(), MlxDtype::Int8);
    assert_eq!(bytes.to_vec_i8().unwrap(), vec![i8::MIN, i8::MAX]);
    assert_eq!(shorts.to_vec_i16().unwrap(), vec![i16::MIN, i16::MAX]);
    assert_eq!(longs.to_vec_i64().unwrap(), vec![i64::MIN, i64::MAX]);
}

#[test]
fn should_keep_double_precision_a_single_precision_array_would_lose() {
    const BEYOND_SINGLE_PRECISION: f64 = 1.000_000_000_1;

    let doubles = MlxArray::from_f64(&[BEYOND_SINGLE_PRECISION], &[1]).unwrap();

    assert_eq!(doubles.dtype(), MlxDtype::Float64);
    assert_eq!(doubles.to_vec_f64().unwrap(), vec![BEYOND_SINGLE_PRECISION]);
}

#[test]
fn should_round_trip_boolean_values() {
    let flags = MlxArray::from_bools(&[true, false, true], &[3]).unwrap();

    assert_eq!(flags.dtype(), MlxDtype::Bool);
    assert_eq!(flags.to_vec_bool().unwrap(), vec![true, false, true]);
}

#[test]
fn should_round_trip_half_precision_bits_unchanged() {
    const HALF_ONE_BITS: u16 = 0x3C00;
    const BROAD_ONE_BITS: u16 = 0x3F80;

    let halves = MlxArray::from_f16_bits(&[HALF_ONE_BITS], &[1]).unwrap();
    let brain_floats = MlxArray::from_bf16_bits(&[BROAD_ONE_BITS], &[1]).unwrap();

    assert_eq!(halves.dtype(), MlxDtype::Float16);
    assert_eq!(halves.to_vec_f16_bits().unwrap(), vec![HALF_ONE_BITS]);
    assert_eq!(brain_floats.dtype(), MlxDtype::BFloat16);
    assert_eq!(
        brain_floats.to_vec_bf16_bits().unwrap(),
        vec![BROAD_ONE_BITS]
    );
}

#[test]
fn should_round_trip_complex_values_with_their_parts_in_order() {
    let complex_values = MlxArray::from_complex(&[(1.0, -2.0), (3.0, 4.0)], &[2]).unwrap();

    assert_eq!(complex_values.dtype(), MlxDtype::Complex64);
    assert_eq!(
        complex_values.to_vec_complex().unwrap(),
        vec![(1.0, -2.0), (3.0, 4.0)]
    );
}

#[test]
fn should_lay_values_out_in_row_major_order() {
    let matrix = MlxArray::from_i64(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();

    assert_eq!(matrix.shape(), vec![2, 3]);
    assert_eq!(matrix.to_vec_i64().unwrap(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn should_accept_a_scalar_and_a_zero_sized_shape() {
    assert_eq!(
        MlxArray::from_i64(&[9], &[]).unwrap().item_i64().unwrap(),
        9
    );
    assert_eq!(
        MlxArray::from_u8(&[], &[0]).unwrap().to_vec_u8().unwrap(),
        Vec::<u8>::new()
    );
}

#[test]
fn should_reject_a_shape_that_disagrees_with_the_value_count_for_every_type() {
    assert!(MlxArray::from_u8(&[1], &[2]).is_err());
    assert!(MlxArray::from_u16(&[1], &[2]).is_err());
    assert!(MlxArray::from_u64(&[1], &[2]).is_err());
    assert!(MlxArray::from_i8(&[1], &[2]).is_err());
    assert!(MlxArray::from_i16(&[1], &[2]).is_err());
    assert!(MlxArray::from_i64(&[1], &[2]).is_err());
    assert!(MlxArray::from_f64(&[1.0], &[2]).is_err());
    assert!(MlxArray::from_bools(&[true], &[2]).is_err());
    assert!(MlxArray::from_f16_bits(&[1], &[2]).is_err());
    assert!(MlxArray::from_bf16_bits(&[1], &[2]).is_err());
    assert!(MlxArray::from_complex(&[(1.0, 1.0)], &[2]).is_err());
}
