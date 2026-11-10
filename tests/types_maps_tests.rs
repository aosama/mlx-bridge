//! Tier 1 contracts for the string-keyed maps: absent keys and exhausted
//! iterators are values rather than failures, and entries outlive their map.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, floats_of, gpu_context};
use mlx_bridge::{MlxMapStringToArray, MlxMapStringToString};

#[test]
fn should_return_a_stored_array_by_key() {
    let mut map = MlxMapStringToArray::new().expect("the map should build");

    map.insert("weights", &float_array(&[1.0], &[1]))
        .expect("the insert should work");

    let stored = map
        .value_at("weights")
        .unwrap()
        .expect("the key was stored");
    assert_eq!(floats_of(&stored), vec![1.0]);
}

#[test]
fn should_report_an_absent_array_key_as_none() {
    gpu_context();
    let map = MlxMapStringToArray::new().expect("the map should build");

    assert!(map.value_at("missing").unwrap().is_none());
}

#[test]
fn should_replace_the_array_stored_under_an_existing_key() {
    let mut map = MlxMapStringToArray::new().expect("the map should build");

    map.insert("key", &float_array(&[1.0], &[1])).unwrap();
    map.insert("key", &float_array(&[2.0], &[1])).unwrap();

    assert_eq!(floats_of(&map.value_at("key").unwrap().unwrap()), vec![2.0]);
}

#[test]
fn should_keep_a_read_array_valid_after_the_map_drops() {
    let mut map = MlxMapStringToArray::new().expect("the map should build");
    map.insert("key", &float_array(&[3.0], &[1])).unwrap();

    let stored = map.value_at("key").unwrap().unwrap();
    drop(map);

    assert_eq!(floats_of(&stored), vec![3.0]);
}

#[test]
fn should_finish_array_iteration_with_none_after_every_entry() {
    let mut map = MlxMapStringToArray::new().expect("the map should build");
    map.insert("first", &float_array(&[1.0], &[1])).unwrap();
    map.insert("second", &float_array(&[2.0], &[1])).unwrap();

    let mut iterator = map.iter().expect("the iterator should build");
    let mut seen_keys = Vec::new();
    while let Some((key, _array)) = iterator.next_entry().unwrap() {
        seen_keys.push(key);
    }
    seen_keys.sort();

    assert_eq!(seen_keys, vec!["first", "second"]);
    assert!(
        iterator.next_entry().unwrap().is_none(),
        "an exhausted iterator stays exhausted"
    );
}

#[test]
fn should_finish_iterating_an_empty_array_map_immediately() {
    let map = MlxMapStringToArray::new().expect("the map should build");

    let mut iterator = map.iter().expect("the iterator should build");

    assert!(iterator.next_entry().unwrap().is_none());
}

#[test]
fn should_reject_an_array_key_with_an_interior_null_byte() {
    let mut map = MlxMapStringToArray::new().expect("the map should build");

    assert!(map.insert("a\0b", &float_array(&[1.0], &[1])).is_err());
    assert!(map.value_at("a\0b").is_err());
}

#[test]
fn should_copy_an_array_map_through_set() {
    let mut source = MlxMapStringToArray::new().expect("the map should build");
    source.insert("key", &float_array(&[1.0], &[1])).unwrap();
    let mut copy = MlxMapStringToArray::new().expect("the map should build");

    copy.set(&source).expect("the copy should work");
    drop(source);

    assert!(copy.value_at("key").unwrap().is_some());
}

#[test]
fn should_round_trip_utf8_keys_and_values() {
    let mut map = MlxMapStringToString::new().expect("the map should build");

    map.insert("modèle", "日本語")
        .expect("the insert should work");

    assert_eq!(map.value_at("modèle").unwrap().unwrap(), "日本語");
}

#[test]
fn should_store_an_empty_string_distinctly_from_an_absent_key() {
    let mut map = MlxMapStringToString::new().expect("the map should build");

    map.insert("present", "").expect("the insert should work");

    assert_eq!(map.value_at("present").unwrap().as_deref(), Some(""));
    assert_eq!(map.value_at("absent").unwrap(), None);
}

#[test]
fn should_finish_string_iteration_with_none_after_every_entry() {
    let mut map = MlxMapStringToString::new().expect("the map should build");
    map.insert("first", "1").unwrap();
    map.insert("second", "2").unwrap();

    let mut iterator = map.iter().expect("the iterator should build");
    let mut seen_entries = Vec::new();
    while let Some(entry) = iterator.next_entry().unwrap() {
        seen_entries.push(entry);
    }
    seen_entries.sort();

    assert_eq!(
        seen_entries,
        vec![
            ("first".to_owned(), "1".to_owned()),
            ("second".to_owned(), "2".to_owned())
        ]
    );
}

#[test]
fn should_finish_iterating_an_empty_string_map_immediately() {
    let map = MlxMapStringToString::new().expect("the map should build");

    let mut iterator = map.iter().expect("the iterator should build");

    assert!(iterator.next_entry().unwrap().is_none());
}

#[test]
fn should_reject_string_entries_with_an_interior_null_byte() {
    let mut map = MlxMapStringToString::new().expect("the map should build");

    assert!(map.insert("a\0b", "value").is_err());
    assert!(map.insert("key", "a\0b").is_err());
    assert!(map.value_at("a\0b").is_err());
}

#[test]
fn should_copy_a_string_map_through_set() {
    let mut source = MlxMapStringToString::new().expect("the map should build");
    source.insert("key", "value").unwrap();
    let mut copy = MlxMapStringToString::new().expect("the map should build");

    copy.set(&source).expect("the copy should work");
    drop(source);

    assert_eq!(copy.value_at("key").unwrap().unwrap(), "value");
}
