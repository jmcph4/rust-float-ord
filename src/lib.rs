//! Order floating point numbers, into this ordering:
//!
//!    NaN | -Infinity | x < 0 | -0 | +0 | x > 0 | +Infinity | NaN

use num::traits::{Bounded, Zero};
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::transmute;
use std::ops::Deref;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// A wrapper for floats, that implements total equality and ordering
/// and hashing.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    any(test, feature = "bytemuck"),
    derive(bytemuck::Zeroable, bytemuck::Pod)
)]
#[repr(transparent)]
pub struct FloatOrd<T>(pub T);

#[cfg(any(test, feature = "bytemuck"))]
unsafe impl<T> bytemuck::TransparentWrapper<T> for FloatOrd<T> {}

macro_rules! float_ord_impl {
    ($f:ident, $i:ident, $n:expr) => {
        impl FloatOrd<$f> {
            fn convert(self) -> $i {
                let u: $i = self.0.to_bits();
                let bit = 1 << ($n - 1);
                if u & bit == 0 {
                    u | bit
                } else {
                    !u
                }
            }
        }
        impl From<$f> for FloatOrd<$f> {
            fn from(from: $f) -> Self {
                FloatOrd(from)
            }
        }
        impl PartialEq for FloatOrd<$f> {
            fn eq(&self, other: &Self) -> bool {
                self.convert() == other.convert()
            }
        }
        impl Eq for FloatOrd<$f> {}
        impl PartialOrd for FloatOrd<$f> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.convert().partial_cmp(&other.convert())
            }
        }
        impl Ord for FloatOrd<$f> {
            fn cmp(&self, other: &Self) -> Ordering {
                self.convert().cmp(&other.convert())
            }
        }
        impl Hash for FloatOrd<$f> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.convert().hash(state);
            }
        }
        impl Deref for FloatOrd<$f> {
            type Target = $f;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl Add for FloatOrd<$f> {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                FloatOrd(self.0 + other.0)
            }
        }
        impl Sub for FloatOrd<$f> {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                FloatOrd(self.0 - other.0)
            }
        }
        impl Mul for FloatOrd<$f> {
            type Output = Self;

            fn mul(self, other: Self) -> Self {
                FloatOrd(self.0 * other.0)
            }
        }
        impl Div for FloatOrd<$f> {
            type Output = Self;

            fn div(self, other: Self) -> Self {
                FloatOrd(self.0 / other.0)
            }
        }
        impl Rem for FloatOrd<$f> {
            type Output = Self;

            fn rem(self, other: Self) -> Self {
                FloatOrd(self.0 % other.0)
            }
        }
        impl fmt::Display for FloatOrd<$f> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
        impl Zero for FloatOrd<$f> {
            fn zero() -> Self {
                Self(0.0)
            }

            fn is_zero(&self) -> bool {
                self.0 == 0.0
            }
        }
        impl Bounded for FloatOrd<$f> {
            fn max_value() -> Self {
                Self(<$f>::MAX)
            }

            fn min_value() -> Self {
                Self(<$f>::MIN)
            }
        }
    };
}

float_ord_impl!(f32, u32, 32);
float_ord_impl!(f64, u64, 64);

/// Sort a slice of floats.
///
/// # Allocation behavior
///
/// This routine uses a quicksort implementation that does not heap allocate.
///
/// # Example
///
/// ```
/// let mut v = [-5.0, 4.0, 1.0, -3.0, 2.0];
///
/// float_ord::sort(&mut v);
/// assert!(v == [-5.0, -3.0, 1.0, 2.0, 4.0]);
/// ```
pub fn sort<T>(v: &mut [T])
where
    FloatOrd<T>: Ord,
{
    let v_: &mut [FloatOrd<T>] = unsafe { transmute(v) };
    v_.sort_unstable();
}

#[cfg(test)]
mod tests {
    extern crate bytemuck;
    extern crate rand;
    extern crate std;

    use self::rand::{thread_rng, Rng};
    use self::std::collections::hash_map::DefaultHasher;
    use self::std::hash::{Hash, Hasher};
    use self::std::iter;
    use self::std::prelude::v1::*;
    use super::FloatOrd;

    #[test]
    fn test_ord() {
        assert!(FloatOrd(1.0f64) < FloatOrd(2.0f64));
        assert!(FloatOrd(2.0f32) > FloatOrd(1.0f32));
        assert!(FloatOrd(1.0f64) == FloatOrd(1.0f64));
        assert!(FloatOrd(1.0f32) == FloatOrd(1.0f32));
        assert!(FloatOrd(0.0f64) > FloatOrd(-0.0f64));
        assert!(FloatOrd(0.0f32) > FloatOrd(-0.0f32));
        assert!(FloatOrd(::std::f64::NAN) == FloatOrd(::std::f64::NAN));
        assert!(FloatOrd(::std::f32::NAN) == FloatOrd(::std::f32::NAN));
        assert!(FloatOrd(-::std::f64::NAN) < FloatOrd(::std::f64::NAN));
        assert!(FloatOrd(-::std::f32::NAN) < FloatOrd(::std::f32::NAN));
        assert!(FloatOrd(-::std::f64::INFINITY) < FloatOrd(::std::f64::INFINITY));
        assert!(FloatOrd(-::std::f32::INFINITY) < FloatOrd(::std::f32::INFINITY));
        assert!(FloatOrd(::std::f64::INFINITY) < FloatOrd(::std::f64::NAN));
        assert!(FloatOrd(::std::f32::INFINITY) < FloatOrd(::std::f32::NAN));
        assert!(FloatOrd(-::std::f64::NAN) < FloatOrd(::std::f64::INFINITY));
        assert!(FloatOrd(-::std::f32::NAN) < FloatOrd(::std::f32::INFINITY));
        let float1 = FloatOrd(1.0f64);
        assert!(float1 == FloatOrd(*float1));
    }

    #[test]
    fn test_ord_numbers() {
        let mut rng = thread_rng();
        for n in 0..16 {
            for l in 0..16 {
                let v = iter::repeat(())
                    .map(|()| rng.gen())
                    .map(|x: f64| x % (1 << l) as i64 as f64)
                    .take(1 << n)
                    .collect::<Vec<_>>();
                assert!(v
                    .windows(2)
                    .all(|w| (w[0] <= w[1]) == (FloatOrd(w[0]) <= FloatOrd(w[1]))));
            }
        }
    }

    #[test]
    fn test_from() {
        assert!(FloatOrd(3.145f64) == 3.145f64.into());
        assert!(FloatOrd(3.145f64) != 2.718f64.into());
    }

    fn hash<F: Hash>(f: F) -> u64 {
        let mut hasher = DefaultHasher::new();
        f.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_hash() {
        assert_ne!(hash(FloatOrd(0.0f64)), hash(FloatOrd(-0.0f64)));
        assert_ne!(hash(FloatOrd(0.0f32)), hash(FloatOrd(-0.0f32)));
        assert_eq!(hash(FloatOrd(-0.0f64)), hash(FloatOrd(-0.0f64)));
        assert_eq!(hash(FloatOrd(0.0f32)), hash(FloatOrd(0.0f32)));
        assert_ne!(
            hash(FloatOrd(::std::f64::NAN)),
            hash(FloatOrd(-::std::f64::NAN))
        );
        assert_ne!(
            hash(FloatOrd(::std::f32::NAN)),
            hash(FloatOrd(-::std::f32::NAN))
        );
        assert_eq!(
            hash(FloatOrd(::std::f64::NAN)),
            hash(FloatOrd(::std::f64::NAN))
        );
        assert_eq!(
            hash(FloatOrd(-::std::f32::NAN)),
            hash(FloatOrd(-::std::f32::NAN))
        );
    }

    #[test]
    fn test_sort_numbers() {
        let mut rng = thread_rng();
        for n in 0..16 {
            for l in 0..16 {
                let mut v = iter::repeat(())
                    .map(|()| rng.gen())
                    .map(|x: f64| x % (1 << l) as i64 as f64)
                    .take(1 << n)
                    .collect::<Vec<_>>();
                let mut v1 = v.clone();

                super::sort(&mut v);
                assert!(v.windows(2).all(|w: &[f64]| w[0] <= w[1]));

                v1.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!(v1.windows(2).all(|w| w[0] <= w[1]));

                v1.sort_by(|a, b| b.partial_cmp(a).unwrap());
                assert!(v1.windows(2).all(|w| w[0] >= w[1]));
            }
        }

        let mut v = [5.0];
        super::sort(&mut v);
        assert!(v == [5.0]);
    }

    #[test]
    fn test_sort_nan() {
        let nan = ::std::f64::NAN;
        let mut v = [-1.0, 5.0, 0.0, -0.0, nan, 1.5, nan, 3.7];
        super::sort(&mut v);
        assert!(v[0] == -1.0);
        assert!(v[1] == 0.0 && v[1].is_sign_negative());
        assert!(v[2] == 0.0 && !v[2].is_sign_negative());
        assert!(v[3] == 1.5);
        assert!(v[4] == 3.7);
        assert!(v[5] == 5.0);
        assert!(v[6].is_nan());
        assert!(v[7].is_nan());
    }

    #[test]
    fn test_bytemuck() {
        use tests::bytemuck::TransparentWrapper;

        assert!(FloatOrd::wrap(0.0f32) > FloatOrd(-0.0f32));
        assert!(FloatOrd(0.0f32) > FloatOrd::wrap(-0.0f32));
        assert!(FloatOrd::wrap(::std::f64::NAN) == FloatOrd(::std::f64::NAN));
        assert!(FloatOrd(::std::f64::NAN) == FloatOrd::wrap(::std::f64::NAN));
        assert!(FloatOrd::wrap(::std::f32::NAN) == FloatOrd(::std::f32::NAN));
        assert!(FloatOrd(::std::f32::NAN) == FloatOrd::wrap(::std::f32::NAN));
        assert!(FloatOrd::wrap(-::std::f64::NAN) < FloatOrd(::std::f64::NAN));
        assert!(FloatOrd(-::std::f64::NAN) < FloatOrd::wrap(::std::f64::NAN));
        assert!(FloatOrd::wrap(-::std::f32::NAN) < FloatOrd(::std::f32::NAN));
        assert!(FloatOrd(-::std::f32::NAN) < FloatOrd::wrap(::std::f32::NAN));
        assert!(FloatOrd::wrap(-::std::f64::INFINITY) < FloatOrd(::std::f64::INFINITY));
        assert!(FloatOrd(-::std::f64::INFINITY) < FloatOrd::wrap(::std::f64::INFINITY));
        assert!(FloatOrd::wrap(-::std::f32::INFINITY) < FloatOrd(::std::f32::INFINITY));
        assert!(FloatOrd(-::std::f32::INFINITY) < FloatOrd::wrap(::std::f32::INFINITY));
        assert!(FloatOrd::wrap(::std::f64::INFINITY) < FloatOrd(::std::f64::NAN));
        assert!(FloatOrd(::std::f64::INFINITY) < FloatOrd::wrap(::std::f64::NAN));
        assert!(FloatOrd::wrap(::std::f32::INFINITY) < FloatOrd(::std::f32::NAN));
        assert!(FloatOrd(::std::f32::INFINITY) < FloatOrd::wrap(::std::f32::NAN));
        assert!(FloatOrd::wrap(-::std::f64::NAN) < FloatOrd(::std::f64::INFINITY));
        assert!(FloatOrd(-::std::f64::NAN) < FloatOrd::wrap(::std::f64::INFINITY));
        assert!(FloatOrd::wrap(-::std::f32::NAN) < FloatOrd(::std::f32::INFINITY));
        assert!(FloatOrd(-::std::f32::NAN) < FloatOrd::wrap(::std::f32::INFINITY));
    }

    #[test]
    fn test_sort_bytemuck() {
        let nan = ::std::f64::NAN;
        let mut v = [-1.0, 5.0, 0.0, -0.0, nan, 1.5, nan, 3.7];
        let floatord_slice: &mut [FloatOrd<f64>] = bytemuck::cast_slice_mut(&mut v);
        floatord_slice.sort_unstable();
        assert!(v[0] == -1.0);
        assert!(v[1] == 0.0 && v[1].is_sign_negative());
        assert!(v[2] == 0.0 && !v[2].is_sign_negative());
        assert!(v[3] == 1.5);
        assert!(v[4] == 3.7);
        assert!(v[5] == 5.0);
        assert!(v[6].is_nan());
        assert!(v[7].is_nan());
    }
}
