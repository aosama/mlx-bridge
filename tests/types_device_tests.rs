//! Tier 5 contracts for `MlxDeviceHandle` and `MlxDeviceInfo`: device identity,
//! the default device, and typed information entries.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::gpu_context;
use mlx_bridge::{MlxDeviceHandle, raw};

#[test]
fn should_make_the_gpu_the_default_device_on_apple_silicon() {
    gpu_context();
    let gpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0).unwrap();

    let default_device = MlxDeviceHandle::default_device().unwrap();

    assert!(default_device.equals(&gpu_device));
    assert_eq!(default_device.index().unwrap(), 0);
    assert_eq!(
        default_device.device_type().unwrap(),
        raw::mlx_device_type__MLX_GPU
    );
    assert!(default_device.to_string_lossy().unwrap().contains("gpu"));
}

#[test]
fn should_distinguish_the_cpu_device_and_report_it_available() {
    gpu_context();
    let cpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_CPU, 0).unwrap();
    let gpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0).unwrap();

    assert!(!cpu_device.equals(&gpu_device));
    assert!(cpu_device.is_available().unwrap());
    assert!(gpu_device.is_available().unwrap());
    assert!(MlxDeviceHandle::count_of_type(raw::mlx_device_type__MLX_GPU).unwrap() >= 1);
}

#[test]
fn should_replace_and_restore_the_default_device() {
    gpu_context();
    let original_default = MlxDeviceHandle::default_device().unwrap();
    let cpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_CPU, 0).unwrap();

    cpu_device.set_as_default().unwrap();
    let observed_default = MlxDeviceHandle::default_device().unwrap();
    original_default.set_as_default().unwrap();

    assert!(observed_default.equals(&cpu_device));
    assert!(
        MlxDeviceHandle::default_device()
            .unwrap()
            .equals(&original_default)
    );
}

#[test]
fn should_copy_a_device_handle_through_set() {
    gpu_context();
    let cpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_CPU, 0).unwrap();
    let mut copied_device = MlxDeviceHandle::default_device().unwrap();

    copied_device.set(&cpu_device).unwrap();

    assert!(copied_device.equals(&cpu_device));
}

#[test]
fn should_describe_the_gpu_through_typed_information_entries() {
    gpu_context();
    let gpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0).unwrap();

    let information = gpu_device.information().unwrap();

    assert!(information.has_key("device_name").unwrap());
    assert!(information.is_string("device_name").unwrap());
    assert!(!information.string_value("device_name").unwrap().is_empty());
    assert!(!information.is_string("memory_size").unwrap());
    assert!(information.size_value("memory_size").unwrap() > 0);
    assert!(!information.keys().unwrap().is_empty());
}

#[test]
fn should_report_an_absent_information_key() {
    gpu_context();
    let gpu_device = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0).unwrap();

    let information = gpu_device.information().unwrap();

    assert!(!information.has_key("not-a-key").unwrap());
    assert!(information.string_value("not-a-key").is_err());
    assert!(information.size_value("not-a-key").is_err());
    assert!(information.has_key("a\0b").is_err());
}
