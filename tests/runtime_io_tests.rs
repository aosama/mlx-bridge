//! Tier 6 contracts for serialization: safetensors and array round trips
//! through both the file path and the reader/writer vtable, a failing
//! transport proving error propagation, and GGUF round trips.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use std::ffi::{c_char, c_int, c_void};

use bridge_fixtures::{float_array, floats_of, gpu_context, scratch_file_path};
use mlx_bridge::{
    MlxArray, MlxIoGguf, MlxIoReader, MlxIoWriter, MlxMapStringToArray, MlxMapStringToString,
    MlxStream, MlxVectorString, load_array_from_reader, load_safetensors,
    load_safetensors_from_reader, raw, save_array_to_writer, save_safetensors,
    save_safetensors_to_writer,
};

const SEEK_FROM_START: c_int = 0;
const SEEK_FROM_CURRENT: c_int = 1;
const SEEK_FROM_END: c_int = 2;

/// An in-memory byte transport standing behind the MLX-C vtable.
struct MemoryTransport {
    stored_bytes: Vec<u8>,
    cursor_position: usize,
    is_healthy: bool,
}

impl MemoryTransport {
    fn holding(stored_bytes: Vec<u8>) -> Self {
        Self {
            stored_bytes,
            cursor_position: 0,
            is_healthy: true,
        }
    }

    fn as_descriptor(&mut self) -> *mut c_void {
        std::ptr::from_mut(self).cast()
    }
}

/// # Safety
/// The descriptor must point to the live `MemoryTransport` the test owns.
unsafe fn transport_of<'transport>(descriptor: *mut c_void) -> &'transport mut MemoryTransport {
    // SAFETY: Every vtable below receives the descriptor from `as_descriptor`
    // and the test keeps the transport alive past the reader or writer.
    unsafe { &mut *descriptor.cast::<MemoryTransport>() }
}

unsafe extern "C" fn transport_is_healthy(descriptor: *mut c_void) -> bool {
    unsafe { transport_of(descriptor) }.is_healthy
}

unsafe extern "C" fn transport_tell(descriptor: *mut c_void) -> usize {
    unsafe { transport_of(descriptor) }.cursor_position
}

unsafe extern "C" fn transport_seek(descriptor: *mut c_void, offset: i64, whence: c_int) -> c_int {
    let transport = unsafe { transport_of(descriptor) };
    let origin = match whence {
        SEEK_FROM_START => 0,
        SEEK_FROM_CURRENT => transport.cursor_position as i64,
        SEEK_FROM_END => transport.stored_bytes.len() as i64,
        _ => return -1,
    };
    transport.cursor_position = (origin + offset) as usize;
    0
}

unsafe extern "C" fn transport_read(
    descriptor: *mut c_void,
    destination: *mut c_char,
    requested_byte_count: usize,
) -> usize {
    let transport = unsafe { transport_of(descriptor) };
    let start_position = transport.cursor_position;
    let copied_byte_count = unsafe {
        transport_read_at_offset(
            descriptor,
            destination,
            requested_byte_count,
            start_position,
        )
    };
    transport.cursor_position = start_position + copied_byte_count;
    copied_byte_count
}

unsafe extern "C" fn transport_read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    requested_byte_count: usize,
    byte_offset: usize,
) -> usize {
    let transport = unsafe { transport_of(descriptor) };
    let remaining_byte_count = transport.stored_bytes.len().saturating_sub(byte_offset);
    let copied_byte_count = requested_byte_count.min(remaining_byte_count);
    // SAFETY: MLX provides a destination for `requested_byte_count` bytes and
    // the copied range stays inside the stored bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            transport.stored_bytes[byte_offset..].as_ptr(),
            destination.cast::<u8>(),
            copied_byte_count,
        );
    }
    copied_byte_count
}

unsafe extern "C" fn transport_write(
    descriptor: *mut c_void,
    source: *const c_char,
    written_byte_count: usize,
) -> usize {
    let transport = unsafe { transport_of(descriptor) };
    // SAFETY: MLX provides a source of `written_byte_count` readable bytes.
    let written_bytes =
        unsafe { std::slice::from_raw_parts(source.cast::<u8>(), written_byte_count) };
    let end_position = transport.cursor_position + written_byte_count;
    if transport.stored_bytes.len() < end_position {
        transport.stored_bytes.resize(end_position, 0);
    }
    transport.stored_bytes[transport.cursor_position..end_position].copy_from_slice(written_bytes);
    transport.cursor_position = end_position;
    written_byte_count
}

unsafe extern "C" fn transport_label(_descriptor: *mut c_void) -> *const c_char {
    c"memory-transport".as_ptr()
}

/// The test owns each transport, so releasing the handle frees nothing.
unsafe extern "C" fn transport_free(_descriptor: *mut c_void) {}

fn memory_vtable() -> raw::mlx_io_vtable {
    raw::mlx_io_vtable {
        is_open: Some(transport_is_healthy),
        good: Some(transport_is_healthy),
        tell: Some(transport_tell),
        seek: Some(transport_seek),
        read: Some(transport_read),
        read_at_offset: Some(transport_read_at_offset),
        write: Some(transport_write),
        label: Some(transport_label),
        free: Some(transport_free),
    }
}

fn reader_over(transport: &mut MemoryTransport) -> MlxIoReader {
    // SAFETY: The transport outlives the reader and the vtable slots follow
    // the MLX-C contract for it.
    unsafe { MlxIoReader::from_parts(transport.as_descriptor(), memory_vtable()) }
        .expect("the reader should build")
}

fn writer_over(transport: &mut MemoryTransport) -> MlxIoWriter {
    // SAFETY: The transport outlives the writer and the vtable slots follow
    // the MLX-C contract for it.
    unsafe { MlxIoWriter::from_parts(transport.as_descriptor(), memory_vtable()) }
        .expect("the writer should build")
}

fn cpu_stream() -> MlxStream {
    MlxStream::default_cpu().expect("the CPU stream should exist")
}

fn weights_and_metadata() -> (MlxMapStringToArray, MlxMapStringToString) {
    let mut arrays = MlxMapStringToArray::new().unwrap();
    arrays
        .insert("weights", &float_array(&[1.0, 2.0], &[2]))
        .unwrap();
    let mut metadata = MlxMapStringToString::new().unwrap();
    metadata.insert("format", "mlx").unwrap();
    (arrays, metadata)
}

fn assert_holds_weights_and_metadata(
    arrays: &MlxMapStringToArray,
    metadata: &MlxMapStringToString,
) {
    let weights = arrays
        .value_at("weights")
        .unwrap()
        .expect("the key was saved");
    assert_eq!(floats_of(&weights), vec![1.0, 2.0]);
    assert_eq!(metadata.value_at("format").unwrap().unwrap(), "mlx");
}

#[test]
fn should_round_trip_safetensors_through_a_file() {
    gpu_context();
    let path = scratch_file_path("round_trip.safetensors");
    let (arrays, metadata) = weights_and_metadata();

    save_safetensors(&path, &arrays, &metadata).unwrap();
    let (loaded_arrays, loaded_metadata) = load_safetensors(&path, &cpu_stream()).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_holds_weights_and_metadata(&loaded_arrays, &loaded_metadata);
}

#[test]
fn should_round_trip_safetensors_through_the_vtable_transport() {
    gpu_context();
    let (arrays, metadata) = weights_and_metadata();
    let mut storage = MemoryTransport::holding(Vec::new());

    save_safetensors_to_writer(&writer_over(&mut storage), &arrays, &metadata).unwrap();
    storage.cursor_position = 0;
    let (loaded_arrays, loaded_metadata) =
        load_safetensors_from_reader(&reader_over(&mut storage), &cpu_stream()).unwrap();

    assert!(!storage.stored_bytes.is_empty());
    assert_holds_weights_and_metadata(&loaded_arrays, &loaded_metadata);
}

#[test]
fn should_round_trip_one_array_through_a_file() {
    gpu_context();
    let path = scratch_file_path("round_trip.npy");
    let array = float_array(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);

    array.save_to_file(&path).unwrap();
    let loaded = MlxArray::load_from_file(&path, &cpu_stream()).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.shape(), vec![2, 2]);
    assert_eq!(floats_of(&loaded), vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn should_round_trip_one_array_through_the_vtable_transport() {
    gpu_context();
    let array = float_array(&[7.0, 8.0], &[2]);
    let mut storage = MemoryTransport::holding(Vec::new());

    save_array_to_writer(&writer_over(&mut storage), &array).unwrap();
    storage.cursor_position = 0;
    let loaded = load_array_from_reader(&reader_over(&mut storage), &cpu_stream()).unwrap();

    assert_eq!(floats_of(&loaded), vec![7.0, 8.0]);
}

#[test]
fn should_translate_a_failing_writer_into_an_error() {
    gpu_context();
    let (arrays, metadata) = weights_and_metadata();
    let mut failing_storage = MemoryTransport::holding(Vec::new());
    failing_storage.is_healthy = false;

    let failure =
        save_safetensors_to_writer(&writer_over(&mut failing_storage), &arrays, &metadata)
            .expect_err("an unhealthy transport must fail the save");

    assert_eq!(
        failure.operation,
        "save MLX safetensors maps through a writer"
    );
    assert!(!failure.description.is_empty());
}

#[test]
fn should_translate_a_failing_reader_into_an_error() {
    gpu_context();
    let mut failing_storage = MemoryTransport::holding(vec![1, 2, 3]);
    failing_storage.is_healthy = false;

    assert!(load_array_from_reader(&reader_over(&mut failing_storage), &cpu_stream()).is_err());
}

#[test]
fn should_expose_the_transport_descriptor_and_label() {
    gpu_context();
    let mut storage = MemoryTransport::holding(Vec::new());
    let expected_descriptor = storage.as_descriptor();

    let reader = reader_over(&mut storage);

    assert_eq!(reader.descriptor().unwrap(), expected_descriptor);
    assert_eq!(
        reader.to_string_lossy().unwrap(),
        c"memory-transport".to_str().unwrap()
    );
}

#[test]
fn should_translate_a_missing_file_into_an_error() {
    gpu_context();
    let path = scratch_file_path("missing.safetensors");

    assert!(load_safetensors(&path, &cpu_stream()).is_err());
    assert!(MlxArray::load_from_file(&path, &cpu_stream()).is_err());
    assert!(MlxIoGguf::load_from_file(&path, &cpu_stream()).is_err());
}

#[test]
fn should_reject_a_path_with_an_interior_null_byte() {
    gpu_context();
    let (arrays, metadata) = weights_and_metadata();

    assert!(save_safetensors("a\0b", &arrays, &metadata).is_err());
    assert!(load_safetensors("a\0b", &cpu_stream()).is_err());
}

#[test]
fn should_round_trip_gguf_arrays_and_metadata_through_a_file() {
    gpu_context();
    let path = scratch_file_path("round_trip.gguf");
    let mut gguf = MlxIoGguf::new().unwrap();
    gguf.set_array("weights", &float_array(&[1.0, 2.0], &[2]))
        .unwrap();
    gguf.set_metadata_string("general.name", "fixture").unwrap();
    gguf.set_metadata_array("general.scale", &float_array(&[0.5], &[1]))
        .unwrap();

    gguf.save_to_file(&path).unwrap();
    let loaded = MlxIoGguf::load_from_file(&path, &cpu_stream()).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(
        floats_of(&loaded.array_at("weights").unwrap()),
        vec![1.0, 2.0]
    );
    assert_eq!(
        loaded.metadata_string_at("general.name").unwrap(),
        "fixture"
    );
    assert_eq!(
        floats_of(&loaded.metadata_array_at("general.scale").unwrap()),
        vec![0.5]
    );
    assert_eq!(loaded.array_keys().unwrap().len(), 1);
}

#[test]
fn should_report_gguf_metadata_presence_by_kind() {
    gpu_context();
    let mut gguf = MlxIoGguf::new().unwrap();

    gguf.set_metadata_string("text", "value").unwrap();

    assert!(gguf.has_metadata_string("text").unwrap());
    assert!(!gguf.has_metadata_array("text").unwrap());
    assert!(!gguf.has_metadata_string("absent").unwrap());
    assert!(!gguf.has_metadata_string_vector("absent").unwrap());
}

#[test]
fn should_fail_to_read_an_absent_gguf_entry() {
    gpu_context();
    let gguf = MlxIoGguf::new().unwrap();

    assert!(gguf.array_at("absent").is_err());
    assert!(gguf.metadata_string_at("absent").is_err());
    assert!(gguf.metadata_array_at("absent").is_err());
}

#[test]
fn should_round_trip_gguf_string_vector_metadata() {
    gpu_context();
    let mut gguf = MlxIoGguf::new().unwrap();
    let tokens = MlxVectorString::from_strings(&["first", "second"]).unwrap();

    gguf.set_metadata_string_vector("tokenizer.tokens", &tokens)
        .unwrap();

    let stored_tokens = gguf.metadata_string_vector_at("tokenizer.tokens").unwrap();
    assert!(gguf.has_metadata_string_vector("tokenizer.tokens").unwrap());
    assert_eq!(stored_tokens.len(), 2);
    assert_eq!(stored_tokens.string_at(1).unwrap(), "second");
    assert!(gguf.metadata_string_vector_at("absent").is_err());
}

#[test]
fn should_expose_the_writer_descriptor_and_label() {
    gpu_context();
    let mut storage = MemoryTransport::holding(Vec::new());
    let expected_descriptor = storage.as_descriptor();

    let writer = writer_over(&mut storage);

    assert_eq!(writer.descriptor().unwrap(), expected_descriptor);
    assert_eq!(writer.to_string_lossy().unwrap(), "memory-transport");
}
