//! The complete MLX stream domain: streams, thread-local slots, and the
//! process-wide stream registry.

use crate::MlxCError;
use crate::error::check_status;
use crate::raw;
use crate::types::device::MlxDeviceHandle;
use crate::types::stream::MlxStream;
use crate::types::vectors::MlxVectorStream;

/// Per-thread MLX stream slot for one device.
#[derive(Debug)]
pub struct MlxStreamThreadLocal(raw::mlx_stream_thread_local);

impl MlxStreamThreadLocal {
    /// Creates the thread-local slot for one device's stream.
    pub fn for_device(device: &MlxDeviceHandle) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX thread-local stream slot";
        // SAFETY: The device handle is live and the returned slot enters
        // RAII ownership.
        let raw_slot = unsafe { raw::mlx_stream_thread_local_new(device.raw()) };
        if raw_slot.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty thread-local stream handle".to_owned(),
            });
        }
        Ok(Self(raw_slot))
    }

    /// Copies the source slot handle into this slot.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned slot value.
        let status = unsafe { raw::mlx_stream_thread_local_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX thread-local stream slot")
    }

    /// Copies the slot's current stream into an owned handle.
    pub fn stream(&self) -> Result<MlxStream, MlxCError> {
        let operation: &'static str = "read the MLX thread-local stream";
        let mut raw_stream = raw::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: `self` owns a live slot and the output pointer is valid
        // writable storage for one stream handle.
        let status = unsafe { raw::mlx_stream_from_thread_local(&mut raw_stream, self.0) };
        check_status(status, operation)?;
        MlxStream::from_live_raw(raw_stream, operation)
    }

    /// Synchronizes the slot's current stream.
    pub fn synchronize(&self) -> Result<(), MlxCError> {
        // SAFETY: `self` owns a live slot handle.
        let status = unsafe { raw::mlx_synchronize_thread_local(self.0) };
        check_status(status, "synchronize the MLX thread-local stream")
    }
}

impl Drop for MlxStreamThreadLocal {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live slot exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_stream_thread_local_free(self.0);
        }
    }
}

impl MlxStream {
    /// Creates an owned stream on the given device.
    pub fn on_device(device: &MlxDeviceHandle) -> Result<Self, MlxCError> {
        // SAFETY: The device handle is live and the returned stream enters
        // RAII ownership.
        let raw_stream = unsafe { raw::mlx_stream_new_device(device.raw()) };
        Self::from_live_raw(raw_stream, "create an MLX stream on a device")
    }

    /// Creates a stream registered globally rather than per thread, so it can
    /// be evaluated from any thread.
    ///
    /// # Safety
    /// MLX applies no synchronization to these streams, so the caller must rule
    /// out data races on it.
    pub unsafe fn new_thread_unsafe(device: &MlxDeviceHandle) -> Result<Self, MlxCError> {
        // SAFETY: The device handle is live and the returned stream enters
        // RAII ownership; the caller carries the synchronization contract.
        let raw_stream = unsafe { raw::mlx_stream_new_thread_unsafe(device.raw()) };
        Self::from_live_raw(raw_stream, "create a thread-unsafe MLX stream")
    }

    /// The device's default stream on this thread if one exists yet, without
    /// creating it.
    pub fn peek_default_for_device(device: &MlxDeviceHandle) -> Result<Option<Self>, MlxCError> {
        let operation: &'static str = "peek at the MLX default stream";
        let mut raw_stream = raw::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        let mut has_stream = false;
        // SAFETY: The device handle is live and both output pointers are valid
        // writable storage; the stream is written only when one exists.
        let status = unsafe {
            raw::mlxrs_peek_default_stream(&mut raw_stream, &mut has_stream, device.raw())
        };
        check_status(status, operation)?;
        if !has_stream {
            return Ok(None);
        }
        Ok(Some(Self::from_live_raw(raw_stream, operation)?))
    }

    /// Renders the stream description text.
    pub fn to_string_lossy(&self) -> Result<String, MlxCError> {
        let operation: &'static str = "describe an MLX stream";
        let mut output = crate::MlxString::empty();
        // SAFETY: `self` owns a live stream and the output pointer is valid
        // writable storage for one string handle.
        let status = unsafe { raw::mlx_stream_tostring(output.raw_mut(), self.0) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output.to_string_lossy())
    }

    /// Copies the source stream handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned stream value.
        let status = unsafe { raw::mlx_stream_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX stream")
    }

    /// Whether both handles reference the same underlying stream.
    #[must_use]
    pub fn equals(&self, other: &Self) -> bool {
        // SAFETY: Both handles are live; equality is a pure query.
        unsafe { raw::mlx_stream_equal(self.0, other.0) }
    }

    /// The device this stream submits work to.
    pub fn device(&self) -> Result<MlxDeviceHandle, MlxCError> {
        let operation: &'static str = "read the MLX stream device";
        let mut raw_device = raw::mlx_device {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: `self` owns a live stream and the output pointer is valid
        // writable storage for one device handle.
        let status = unsafe { raw::mlx_stream_get_device(&mut raw_device, self.0) };
        check_status(status, operation)?;
        MlxDeviceHandle::from_live_raw(raw_device, operation)
    }

    /// The stream's index on its device.
    pub fn index(&self) -> Result<i32, MlxCError> {
        let operation: &'static str = "read the MLX stream index";
        let mut stream_index: i32 = 0;
        // SAFETY: `self` owns a live stream and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_stream_get_index(&mut stream_index, self.0) };
        check_status(status, operation)?;
        Ok(stream_index)
    }

    /// Synchronizes this stream's pending work.
    pub fn synchronize(&self) -> Result<(), MlxCError> {
        // SAFETY: `self` owns a live stream handle.
        let status = unsafe { raw::mlx_synchronize(self.0) };
        check_status(status, "synchronize an MLX stream")
    }

    /// The device's default stream.
    pub fn default_for_device(device: &MlxDeviceHandle) -> Result<Self, MlxCError> {
        let operation: &'static str = "acquire the MLX device default stream";
        let mut raw_stream = raw::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: The device handle is live and the output pointer is valid
        // writable storage for one stream handle.
        let status = unsafe { raw::mlx_get_default_stream(&mut raw_stream, device.raw()) };
        check_status(status, operation)?;
        Self::from_live_raw(raw_stream, operation)
    }

    /// Every stream created in this process.
    pub fn all_streams() -> Result<MlxVectorStream, MlxCError> {
        let operation: &'static str = "list the MLX streams";
        let mut output = MlxVectorStream::empty();
        // SAFETY: The output pointer is valid writable storage for one
        // stream vector handle.
        let status = unsafe { raw::mlx_get_streams(output.raw_mut()) };
        check_status(status, operation)?;
        Ok(output)
    }
}

/// Frees every stream created in this process except the defaults in use.
pub fn clear_streams() -> Result<(), MlxCError> {
    // SAFETY: Clearing takes no inputs and reports status through the
    // captured-error machinery.
    let status = unsafe { raw::mlx_clear_streams() };
    check_status(status, "clear the MLX streams")
}

/// Replaces the process's default stream.
pub fn set_default_stream(stream: &MlxStream) -> Result<(), MlxCError> {
    // SAFETY: The stream handle is live.
    let status = unsafe { raw::mlx_set_default_stream(stream.raw()) };
    check_status(status, "set the MLX default stream")
}

/// Synchronizes work across all devices' default streams.
pub fn synchronize_all_defaults() -> Result<(), MlxCError> {
    // SAFETY: Synchronization takes no inputs and reports status through the
    // captured-error machinery.
    let status = unsafe { raw::mlx_synchronize_default() };
    check_status(status, "synchronize the MLX default streams")
}
