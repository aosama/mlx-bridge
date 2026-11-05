//! Contracts for the process-wide print format. One test exercises the whole
//! cycle because the setting is global to this test binary.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, gpu_context};
use mlx_c_rust::{MlxArray, MlxPrintFormat, print_format, set_print_format};

#[test]
fn should_apply_and_restore_the_print_format() {
    gpu_context();
    let original_format = print_format().unwrap();
    let values = float_array(&[1.23456], &[1]);

    set_print_format(MlxPrintFormat {
        precision: 2,
        capitalize_bool: true,
    })
    .unwrap();
    let observed_format = print_format().unwrap();
    let rendered_float = values.to_string_lossy().unwrap();
    let rendered_bool = MlxArray::from_bool(true).to_string_lossy().unwrap();
    set_print_format(MlxPrintFormat {
        precision: -1,
        capitalize_bool: false,
    })
    .unwrap();
    let rendered_with_default_precision = values.to_string_lossy().unwrap();
    set_print_format(original_format).unwrap();

    assert_eq!(observed_format.precision, 2);
    assert!(observed_format.capitalize_bool);
    assert!(rendered_float.contains("1.23"), "{rendered_float}");
    assert!(!rendered_float.contains("1.234"), "{rendered_float}");
    assert!(rendered_bool.contains("True"), "{rendered_bool}");
    assert!(
        rendered_with_default_precision.contains("1.23456"),
        "a negative precision selects MLX's default: {rendered_with_default_precision}"
    );
    assert_eq!(print_format().unwrap(), original_format);
}
