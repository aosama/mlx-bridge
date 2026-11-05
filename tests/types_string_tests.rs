//! Tier 1 contracts for `MlxString`: UTF-8 round trips and the empty edge.

use mlx_c_rust::MlxString;

#[test]
fn should_round_trip_utf8_text() {
    let string: MlxString = "日本語 héllo".parse().expect("the string should build");

    assert_eq!(string.to_string_lossy(), "日本語 héllo");
}

#[test]
fn should_read_a_new_empty_string_as_empty_text() {
    assert_eq!(MlxString::empty().to_string_lossy(), "");
}

#[test]
fn should_reject_text_with_an_interior_null_byte() {
    assert!("a\0b".parse::<MlxString>().is_err());
}

#[test]
fn should_keep_a_copy_readable_after_its_source_drops() {
    let source: MlxString = "kept".parse().expect("the string should build");
    let mut copy = MlxString::empty();

    copy.set(&source).expect("the copy should work");
    drop(source);

    assert_eq!(copy.to_string_lossy(), "kept");
}
