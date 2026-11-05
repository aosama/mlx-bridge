//! JACCL initialization with a Rust all-gather callback.
//!
//! The callback's arguments travel inside the payload so the panic boundary can
//! run a named function: the trampoline parks the call, then
//! `run_payload_closure` invokes `run_pending_all_gather` over the payload.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void};

use crate::error::check_status;
use crate::raw;
use crate::runtime::platform::MlxDistributedGroup;
use crate::types::closures::{drop_payload, run_payload_closure};
use crate::{MlxCError, bindings::support::c_string_argument};

/// One all-gather invocation's borrowed C arguments.
struct PendingAllGather {
    rank: i32,
    world_size: i32,
    source: *const c_char,
    destination: *mut c_char,
    contributed_byte_count: usize,
}

struct AllGatherPayload<F> {
    callback: F,
    pending_call: Option<PendingAllGather>,
}

/// Initializes JACCL with `all_gather` as its metadata side channel.
///
/// `all_gather(rank, world_size, source, destination)` receives this rank's
/// bytes in `source` and must fill `destination`, which holds
/// `world_size * source.len()` bytes ordered by rank. MLX owns the callback from
/// this call on and may hold it for as long as cached groups live, so call
/// `clear_distributed_backends` first to replace one.
///
/// # Errors
/// Returns the captured MLX-C description when initialization fails, or the
/// callback's own failure when it fails during initialization.
pub fn initialize_distributed_with_all_gather<F>(
    strict: bool,
    backend: &str,
    all_gather: F,
) -> Result<MlxDistributedGroup, MlxCError>
where
    F: FnMut(i32, i32, &[u8], &mut [u8]) -> Result<(), MlxCError> + Send + 'static,
{
    let operation: &'static str = "initialize MLX distributed with an all-gather callback";
    let backend_argument = c_string_argument(backend, operation)?;
    let payload = Box::into_raw(Box::new(RefCell::new(AllGatherPayload {
        callback: all_gather,
        pending_call: None,
    })));
    let mut group = MlxDistributedGroup::empty();
    // SAFETY: The backend name outlives the call, the output pointer is valid
    // writable storage, and the trampoline and destructor match the payload's
    // type. MLX owns the payload from entry, failure included, so it is not
    // touched again here.
    let status = unsafe {
        raw::mlx_distributed_init_with_all_gather(
            group.raw_mut(),
            strict,
            backend_argument.as_ptr(),
            Some(all_gather_trampoline::<F>),
            payload.cast(),
            Some(drop_payload::<AllGatherPayload<F>>),
        )
    };
    check_status(status, operation)?;
    group.require_populated(operation)?;
    Ok(group)
}

/// The panic-bounded body: rebuilds the byte slices and runs the callback.
fn run_pending_all_gather<F>(payload: &mut AllGatherPayload<F>) -> Result<(), MlxCError>
where
    F: FnMut(i32, i32, &[u8], &mut [u8]) -> Result<(), MlxCError>,
{
    let operation: &'static str = "run an MLX all-gather callback";
    let Some(call) = payload.pending_call.take() else {
        return Err(MlxCError {
            operation,
            description: "the all-gather trampoline parked no call".to_owned(),
        });
    };
    let Ok(world_size) = usize::try_from(call.world_size) else {
        return Err(MlxCError {
            operation,
            description: "MLX reported a negative world size".to_owned(),
        });
    };
    let Some(destination_byte_count) = world_size.checked_mul(call.contributed_byte_count) else {
        return Err(MlxCError {
            operation,
            description: "the gathered size exceeds the address space".to_owned(),
        });
    };
    let mut no_destination_bytes = [0_u8; 0];
    // SAFETY: MLX lends `contributed_byte_count` readable source bytes and
    // `world_size * contributed_byte_count` writable destination bytes for this
    // synchronous call; a null pointer is only valid for an empty range, which
    // takes the empty slice instead because a slice may never be null.
    let (source_bytes, destination_bytes) = unsafe {
        let source_bytes: &[u8] = if call.source.is_null() {
            &[]
        } else {
            std::slice::from_raw_parts(call.source.cast::<u8>(), call.contributed_byte_count)
        };
        let destination_bytes: &mut [u8] = if call.destination.is_null() {
            &mut no_destination_bytes
        } else {
            std::slice::from_raw_parts_mut(call.destination.cast::<u8>(), destination_byte_count)
        };
        (source_bytes, destination_bytes)
    };
    (payload.callback)(call.rank, call.world_size, source_bytes, destination_bytes)
}

unsafe extern "C" fn all_gather_trampoline<F>(
    rank: c_int,
    world_size: c_int,
    source: *const c_char,
    destination: *mut c_char,
    contributed_byte_count: usize,
    payload: *mut c_void,
) -> c_int
where
    F: FnMut(i32, i32, &[u8], &mut [u8]) -> Result<(), MlxCError> + Send + 'static,
{
    // SAFETY: The payload was created for exactly this `F` by the initializer,
    // and MLX keeps it alive while it can still call this trampoline.
    let payload_cell = unsafe { &*payload.cast::<RefCell<AllGatherPayload<F>>>() };
    match payload_cell.try_borrow_mut() {
        Ok(mut writable_payload) => {
            writable_payload.pending_call = Some(PendingAllGather {
                rank,
                world_size,
                source,
                destination,
                contributed_byte_count,
            });
        }
        Err(_) => {
            crate::error::set_closure_error(MlxCError {
                operation: "run an MLX all-gather callback",
                description: "the all-gather callback was re-entered".to_owned(),
            });
            return 1;
        }
    }
    let Some(outcome) = run_payload_closure::<AllGatherPayload<F>, Result<(), MlxCError>>(
        payload,
        "run an MLX all-gather callback",
        run_pending_all_gather::<F>,
    ) else {
        return 1;
    };
    match outcome {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::take_closure_error;

    fn copy_source_to_own_rank(
        rank: i32,
        world_size: i32,
        source: &[u8],
        destination: &mut [u8],
    ) -> Result<(), MlxCError> {
        assert_eq!(world_size, 2);
        assert_eq!(destination.len(), 2 * source.len());
        let start = source.len() * usize::try_from(rank).expect("the rank is non-negative");
        destination[start..start + source.len()].copy_from_slice(source);
        Ok(())
    }

    fn refusing_all_gather(
        _rank: i32,
        _world_size: i32,
        _source: &[u8],
        _destination: &mut [u8],
    ) -> Result<(), MlxCError> {
        Err(MlxCError {
            operation: "run the fixture all-gather",
            description: "the side channel refused".to_owned(),
        })
    }

    fn panicking_all_gather(
        _rank: i32,
        _world_size: i32,
        _source: &[u8],
        _destination: &mut [u8],
    ) -> Result<(), MlxCError> {
        panic!("the side channel panicked")
    }

    type FixtureAllGather = fn(i32, i32, &[u8], &mut [u8]) -> Result<(), MlxCError>;

    fn fixture_payload(callback: FixtureAllGather) -> RefCell<AllGatherPayload<FixtureAllGather>> {
        RefCell::new(AllGatherPayload {
            callback,
            pending_call: None,
        })
    }

    /// Drives the trampoline the way MLX-C does, from raw pointers.
    fn invoke_trampoline(
        payload: &RefCell<AllGatherPayload<FixtureAllGather>>,
        rank: i32,
        world_size: i32,
        source: *const u8,
        destination: *mut u8,
        byte_count: usize,
    ) -> c_int {
        let payload_pointer = std::ptr::from_ref(payload).cast_mut().cast::<c_void>();
        // SAFETY: The payload has the trampoline's exact type, and each test
        // passes buffers matching the byte count, or null for an empty range.
        unsafe {
            all_gather_trampoline::<FixtureAllGather>(
                rank,
                world_size,
                source.cast(),
                destination.cast(),
                byte_count,
                payload_pointer,
            )
        }
    }

    #[test]
    fn should_hand_the_callback_its_rank_and_both_buffers() {
        let payload = fixture_payload(copy_source_to_own_rank);
        let source = [7_u8, 8];
        let mut destination = [0_u8; 4];

        let status = invoke_trampoline(
            &payload,
            1,
            2,
            source.as_ptr(),
            destination.as_mut_ptr(),
            source.len(),
        );

        assert_eq!(status, 0);
        assert_eq!(destination, [0, 0, 7, 8]);
    }

    #[test]
    fn should_accept_null_buffers_for_an_empty_gather() {
        let payload = fixture_payload(copy_source_to_own_rank);

        let status = invoke_trampoline(&payload, 0, 2, std::ptr::null(), std::ptr::null_mut(), 0);

        assert_eq!(status, 0);
    }

    #[test]
    fn should_park_a_callback_failure_for_the_status_check() {
        let payload = fixture_payload(refusing_all_gather);

        let status = invoke_trampoline(&payload, 0, 2, std::ptr::null(), std::ptr::null_mut(), 0);

        assert_eq!(status, 1);
        let failure = take_closure_error().expect("the failure should be parked");
        assert_eq!(failure.operation, "run the fixture all-gather");
    }

    #[test]
    fn should_stop_a_callback_panic_at_the_ffi_boundary() {
        let payload = fixture_payload(panicking_all_gather);

        let status = invoke_trampoline(&payload, 0, 2, std::ptr::null(), std::ptr::null_mut(), 0);

        assert_eq!(status, 1);
        let failure = take_closure_error().expect("the panic should be parked");
        assert_eq!(failure.description, "the side channel panicked");
    }

    #[test]
    fn should_refuse_a_negative_world_size() {
        let payload = fixture_payload(copy_source_to_own_rank);

        let status = invoke_trampoline(&payload, 0, -1, std::ptr::null(), std::ptr::null_mut(), 0);

        assert_eq!(status, 1);
        let failure = take_closure_error().expect("the failure should be parked");
        assert!(failure.description.contains("negative"), "{failure}");
    }
}
