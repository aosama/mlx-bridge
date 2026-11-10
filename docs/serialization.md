# Serialization

This guide covers safetensors, single arrays, GGUF, and custom byte transports. All paths are UTF-8 strings, and a path containing a null byte is rejected before reaching MLX.

## Safetensors

`save_safetensors` writes a name-to-array map plus string metadata, and `load_safetensors` reads both back on the stream you give it.

```rust
use mlx_bridge::{MlxArray, MlxCError, MlxMapStringToArray, MlxMapStringToString, MlxStream, install_non_terminating_error_handler, load_safetensors, save_safetensors};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let path = std::env::temp_dir().join(format!("mlx_docs_{}_weights.safetensors", std::process::id()));
    let path = path.to_str().expect("the temporary path is UTF-8");

    let mut arrays = MlxArrayMapFixture::weights()?;
    arrays.insert("bias", &MlxArray::from_f32(&[0.5], &[1])?)?;
    let mut metadata = MlxMapStringToString::new()?;
    metadata.insert("format", "mlx")?;

    save_safetensors(path, &arrays, &metadata)?;
    let (loaded_arrays, loaded_metadata) = load_safetensors(path, &MlxStream::default_cpu()?)?;
    std::fs::remove_file(path).expect("the file was just written");

    let weights = loaded_arrays.value_at("weights")?.expect("the key was saved");
    assert_eq!(weights.to_vec_f32()?, vec![1.0, 2.0]);
    assert_eq!(loaded_metadata.value_at("format")?.as_deref(), Some("mlx"));
    Ok(())
}

struct MlxArrayMapFixture;

impl MlxArrayMapFixture {
    fn weights() -> Result<MlxMapStringToArray, MlxCError> {
        let mut arrays = MlxMapStringToArray::new()?;
        arrays.insert("weights", &MlxArray::from_f32(&[1.0, 2.0], &[2])?)?;
        Ok(arrays)
    }
}
```

## Single arrays

`save_to_file` and `load_from_file` handle one array in NumPy's `.npy` format.

```rust
use mlx_bridge::{MlxArray, MlxCError, MlxStream, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let path = std::env::temp_dir().join(format!("mlx_docs_{}_single.npy", std::process::id()));
    let path = path.to_str().expect("the temporary path is UTF-8");

    MlxArray::from_f32(&[3.0, 4.0, 5.0, 6.0], &[2, 2])?.save_to_file(path)?;
    let loaded = MlxArray::load_from_file(path, &MlxStream::default_cpu()?)?;
    std::fs::remove_file(path).expect("the file was just written");

    assert_eq!(loaded.shape(), vec![2, 2]);
    assert_eq!(loaded.to_vec_f32()?, vec![3.0, 4.0, 5.0, 6.0]);
    Ok(())
}
```

## GGUF

`MlxIoGguf` holds arrays and typed metadata: arrays, strings, and string lists. Reading an absent entry is an error, so use the `has_metadata_*` probes when a key is optional. They return `Ok(false)` for an absent key.

```rust
use mlx_bridge::{MlxArray, MlxCError, MlxIoGguf, MlxStream, MlxVectorString, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let path = std::env::temp_dir().join(format!("mlx_docs_{}_model.gguf", std::process::id()));
    let path = path.to_str().expect("the temporary path is UTF-8");

    let mut gguf = MlxIoGguf::new()?;
    gguf.set_array("weights", &MlxArray::from_f32(&[1.0, 2.0], &[2])?)?;
    gguf.set_metadata_string("general.name", "fixture")?;
    gguf.set_metadata_string_vector("tokenizer.tokens", &MlxVectorString::from_strings(&["a", "b"])?)?;
    gguf.save_to_file(path)?;

    let loaded = MlxIoGguf::load_from_file(path, &MlxStream::default_cpu()?)?;
    std::fs::remove_file(path).expect("the file was just written");

    assert_eq!(loaded.array_at("weights")?.to_vec_f32()?, vec![1.0, 2.0]);
    assert_eq!(loaded.metadata_string_at("general.name")?, "fixture");
    assert!(!loaded.has_metadata_string("absent")?);
    assert!(loaded.array_at("absent").is_err());
    Ok(())
}
```

## Custom byte transports

`MlxIoReader` and `MlxIoWriter` read and write through callbacks you supply, so you can serialize to memory, a network, or an encrypted store. You provide a descriptor pointer and a vtable of C functions. Construction is `unsafe` because the compiler cannot check that the descriptor stays valid or that the functions behave.

Every vtable slot must hold a function, because MLX-C calls them without checking, and a missing one crashes the process. The `free` slot runs when the handle drops: if you own the descriptor, as below, give it a function that does nothing. A transport that reports unhealthy through `good` makes saves and loads return an error.

```rust
use std::ffi::{c_char, c_int, c_void};

use mlx_bridge::{MlxArray, MlxCError, MlxIoReader, MlxIoWriter, MlxStream, install_non_terminating_error_handler, load_array_from_reader, raw, save_array_to_writer};

struct MemoryTransport {
    stored_bytes: Vec<u8>,
    cursor_position: usize,
}

/// # Safety
/// The descriptor must point to a live `MemoryTransport`.
unsafe fn transport_of<'transport>(descriptor: *mut c_void) -> &'transport mut MemoryTransport {
    unsafe { &mut *descriptor.cast::<MemoryTransport>() }
}

unsafe extern "C" fn is_healthy(_descriptor: *mut c_void) -> bool {
    true
}

unsafe extern "C" fn tell(descriptor: *mut c_void) -> usize {
    unsafe { transport_of(descriptor) }.cursor_position
}

unsafe extern "C" fn seek(descriptor: *mut c_void, offset: i64, whence: c_int) -> c_int {
    let transport = unsafe { transport_of(descriptor) };
    let origin = match whence {
        0 => 0,
        1 => transport.cursor_position as i64,
        2 => transport.stored_bytes.len() as i64,
        _ => return -1,
    };
    transport.cursor_position = (origin + offset) as usize;
    0
}

unsafe extern "C" fn read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    requested_byte_count: usize,
    byte_offset: usize,
) -> usize {
    let transport = unsafe { transport_of(descriptor) };
    let remaining_byte_count = transport.stored_bytes.len().saturating_sub(byte_offset);
    let copied_byte_count = requested_byte_count.min(remaining_byte_count);
    unsafe {
        std::ptr::copy_nonoverlapping(
            transport.stored_bytes[byte_offset..].as_ptr(),
            destination.cast::<u8>(),
            copied_byte_count,
        );
    }
    copied_byte_count
}

unsafe extern "C" fn read(descriptor: *mut c_void, destination: *mut c_char, requested_byte_count: usize) -> usize {
    let start_position = unsafe { transport_of(descriptor) }.cursor_position;
    let copied_byte_count = unsafe { read_at_offset(descriptor, destination, requested_byte_count, start_position) };
    unsafe { transport_of(descriptor) }.cursor_position = start_position + copied_byte_count;
    copied_byte_count
}

unsafe extern "C" fn write(descriptor: *mut c_void, source: *const c_char, written_byte_count: usize) -> usize {
    let transport = unsafe { transport_of(descriptor) };
    let written_bytes = unsafe { std::slice::from_raw_parts(source.cast::<u8>(), written_byte_count) };
    let end_position = transport.cursor_position + written_byte_count;
    if transport.stored_bytes.len() < end_position {
        transport.stored_bytes.resize(end_position, 0);
    }
    transport.stored_bytes[transport.cursor_position..end_position].copy_from_slice(written_bytes);
    transport.cursor_position = end_position;
    written_byte_count
}

unsafe extern "C" fn label(_descriptor: *mut c_void) -> *const c_char {
    c"memory-transport".as_ptr()
}

// This program owns the transport, so dropping a handle has nothing to release.
unsafe extern "C" fn free_nothing(_descriptor: *mut c_void) {}

fn memory_vtable() -> raw::mlx_io_vtable {
    raw::mlx_io_vtable {
        is_open: Some(is_healthy),
        good: Some(is_healthy),
        tell: Some(tell),
        seek: Some(seek),
        read: Some(read),
        read_at_offset: Some(read_at_offset),
        write: Some(write),
        label: Some(label),
        free: Some(free_nothing),
    }
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let mut storage = MemoryTransport { stored_bytes: Vec::new(), cursor_position: 0 };
    let descriptor: *mut c_void = std::ptr::from_mut(&mut storage).cast();

    // SAFETY: The storage outlives both handles and the vtable matches it.
    let writer = unsafe { MlxIoWriter::from_parts(descriptor, memory_vtable()) }?;
    save_array_to_writer(&writer, &MlxArray::from_f32(&[7.0, 8.0], &[2])?)?;
    drop(writer);

    storage.cursor_position = 0;
    let descriptor: *mut c_void = std::ptr::from_mut(&mut storage).cast();
    let reader = unsafe { MlxIoReader::from_parts(descriptor, memory_vtable()) }?;
    let loaded = load_array_from_reader(&reader, &MlxStream::default_cpu()?)?;

    assert_eq!(loaded.to_vec_f32()?, vec![7.0, 8.0]);
    Ok(())
}
```

`save_safetensors_to_writer` and `load_safetensors_from_reader` do the same for safetensors.
