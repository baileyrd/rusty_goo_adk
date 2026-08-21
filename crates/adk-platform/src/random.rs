//! Random-number provider — a scoped, overridable source of randomness.
//!
//! Ports `google.adk.platform._random`'s `ContextVar`-backed provider-swap
//! pattern (capabilities C0002, C0006). The source installs a *callable*
//! (not a fixed instance) that is re-evaluated on every [`get_random`]
//! call, and warns callers to close over an existing generator if they want
//! sequence continuity across calls rather than a fresh one each time —
//! this crate preserves both the callable-provider shape and that warning.
//!
//! **Adaptation, disclosed rather than silent**: the source scopes the
//! override per `asyncio` task via `contextvars.ContextVar`. No async
//! runtime has been chosen for this port yet (deferred to whichever later
//! phase first needs one — see `capability-manifest.md` phase notes), so
//! this starts thread-scoped via [`std::thread_local`] instead. This keeps
//! the source's real invariant — an override is scoped, not a bare global —
//! while the exact propagation boundary (thread vs. async task) necessarily
//! differs until an async runtime lands. Revisit then.
//!
//! **Adaptation**: the source's default provider is Python's
//! general-purpose (non-cryptographic) `random.Random`, a Mersenne
//! Twister. Bit-for-bit output parity with a different language's PRNG
//! isn't meaningful, so this crate hand-rolls a small xorshift128+
//! generator instead of taking a new external dependency for something
//! this narrowly scoped — the capability being preserved is the
//! swappable/scoped *provider pattern*, not a specific bit stream.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

/// A small, non-cryptographic PRNG (xorshift128+). Not suitable for secrets.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 2],
}

impl Rng {
    /// Deterministic construction from a single 64-bit seed (SplitMix64
    /// spreads it across the two xorshift128+ words).
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = move || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng {
            state: [next(), next()],
        }
    }

    /// Seeds from a coarse, non-cryptographic entropy source (wall clock +
    /// a stack address). Fine for jitter/sampling; never use for secrets.
    pub fn from_entropy() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos() as u64;
        let stack_addr = &nanos as *const u64 as u64;
        Self::from_seed(nanos ^ stack_addr.rotate_left(17))
    }

    /// Next 64 bits of randomness.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state[0];
        let y = self.state[1];
        self.state[0] = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.state[1] = x;
        x.wrapping_add(y)
    }

    /// Next value in `[0, 1)`, using the top 53 bits (matches an `f64`
    /// mantissa's precision).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Next integer in `[low, high)`. Panics if `low >= high`.
    pub fn gen_range(&mut self, low: u64, high: u64) -> u64 {
        assert!(low < high, "gen_range: low must be < high");
        low + (self.next_u64() % (high - low))
    }
}

/// A handle to a shared, mutable `Rng` — cloning the handle (not the
/// state) is how [`get_random`] returns "the same instance" across calls,
/// mirroring the source's `get_random() is get_random()` guarantee after
/// `reset_random_provider()`.
pub type SharedRng = Rc<RefCell<Rng>>;

type Provider = Rc<dyn Fn() -> SharedRng>;

thread_local! {
    static DEFAULT_RANDOM: RefCell<Option<SharedRng>> = const { RefCell::new(None) };
    static RANDOM_PROVIDER: RefCell<Option<Provider>> = const { RefCell::new(None) };
}

/// Installs a callable provider, evaluated on every [`get_random`] call.
///
/// Close over an existing [`SharedRng`] (e.g. `Rc::new(RefCell::new(Rng::from_seed(42)))`)
/// if you want a stable sequence across calls — constructing a fresh `Rng`
/// inside the closure on every call is almost never what you want, exactly
/// as the source's docstring warns.
pub fn set_random_provider<F>(provider: F)
where
    F: Fn() -> SharedRng + 'static,
{
    RANDOM_PROVIDER.with(|p| *p.borrow_mut() = Some(Rc::new(provider)));
}

/// Restores the default provider. Resetting returns to the *same*
/// long-lived default instance, not a freshly reseeded one — matching the
/// source exactly (see `test_reset_random_provider` in the source's own
/// test suite, ported as [`tests::reset_random_provider_returns_same_instance`]
/// in this crate).
pub fn reset_random_provider() {
    RANDOM_PROVIDER.with(|p| *p.borrow_mut() = None);
}

/// Returns the currently active `Rng` handle.
pub fn get_random() -> SharedRng {
    let installed = RANDOM_PROVIDER.with(|p| p.borrow().clone());
    if let Some(provider) = installed {
        return provider();
    }
    DEFAULT_RANDOM.with(|d| {
        let mut d = d.borrow_mut();
        if d.is_none() {
            *d = Some(Rc::new(RefCell::new(Rng::from_entropy())));
        }
        Rc::clone(d.as_ref().expect("just initialized above"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity test for capability C0002: `reset_random_provider` restores
    /// the same default instance, not a fresh one.
    #[test]
    fn reset_random_provider_returns_same_instance() {
        reset_random_provider();
        let first = get_random();
        let second = get_random();
        assert!(
            Rc::ptr_eq(&first, &second),
            "get_random() should return the same instance on repeat calls"
        );
        reset_random_provider();
        let third = get_random();
        assert!(
            Rc::ptr_eq(&first, &third),
            "reset_random_provider() should restore the same long-lived default, not a new one"
        );
    }

    /// Parity test for capability C0002: the provider is a callable
    /// re-evaluated per call, not a fixed instance snapshotted at
    /// `set_random_provider` time.
    #[test]
    fn provider_is_reevaluated_on_every_call() {
        let call_count = Rc::new(RefCell::new(0u32));
        let counted = Rc::clone(&call_count);
        set_random_provider(move || {
            *counted.borrow_mut() += 1;
            Rc::new(RefCell::new(Rng::from_seed(1)))
        });

        let _ = get_random();
        let _ = get_random();
        assert_eq!(*call_count.borrow(), 2, "provider should run on every call");

        reset_random_provider();
    }

    /// Parity test for capability C0002: a provider closing over one
    /// `SharedRng` instance gives sequence continuity across calls.
    #[test]
    fn provider_closing_over_shared_rng_gives_continuity() {
        let shared = Rc::new(RefCell::new(Rng::from_seed(7)));
        let handle_for_provider = Rc::clone(&shared);
        set_random_provider(move || Rc::clone(&handle_for_provider));

        let a = get_random();
        let first_value = a.borrow_mut().next_u64();
        let b = get_random();
        let second_value = b.borrow_mut().next_u64();

        let mut reference = Rng::from_seed(7);
        assert_eq!(first_value, reference.next_u64());
        assert_eq!(second_value, reference.next_u64());

        reset_random_provider();
    }

    #[test]
    fn from_seed_is_deterministic() {
        let mut a = Rng::from_seed(42);
        let mut b = Rng::from_seed(42);
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn next_f64_is_in_unit_range() {
        let mut rng = Rng::from_seed(1);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn gen_range_stays_in_bounds() {
        let mut rng = Rng::from_seed(2);
        for _ in 0..1000 {
            let v = rng.gen_range(10, 20);
            assert!((10..20).contains(&v));
        }
    }
}
