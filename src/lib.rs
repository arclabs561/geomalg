//! Geometric algebra primitives for the real Clifford algebra `Cl(P, Q)`.
//!
//! A [`Cl<P, Q>`] value is a multivector stored as `2^(P + Q)` `f64`
//! coefficients, one per basis blade. Blade indices are bitmasks of basis
//! vectors in canonical (increasing) order: index `0` is the scalar blade,
//! `1` is `e1`, `2` is `e2`, `3` is `e1e2`, and so on. Basis vector `j`
//! squares to `+1` when `j < P` and to `-1` otherwise.
//!
//! ```
//! use geomalg::Cl;
//! use std::f64::consts::FRAC_PI_2;
//!
//! // Rotate e1 into e2 by pi/2 in the e1^e2 plane.
//! let r = Cl::<2, 0>::rotor(0, 1, FRAC_PI_2);
//! let e1 = Cl::<2, 0>::basis(0);
//! let rotated = r.sandwich(&e1);
//!
//! assert!((rotated.coefficient(0b10) - 1.0).abs() < 1e-12);
//! assert!(rotated.coefficient(0b01).abs() < 1e-12);
//! ```
//!
//! # Scope
//!
//! This is a primitive, not a symbolic algebra system. It provides a fixed
//! metric signature, dense `f64` coefficients, the geometric product, grade
//! projection, reversal, and rotor/Pin sandwich actions. It does not do
//! symbolic simplification, arbitrary metrics, or automatic differentiation.

#![warn(missing_docs)]
#![deny(unsafe_code)]

use std::fmt;
use std::ops;

/// Maximum number of basis vectors supported by the `usize` blade bitmask.
const MAX_DIM: usize = 63;

/// A real multivector in the Clifford algebra `Cl(P, Q)`.
///
/// The first `P` basis vectors square to `+1` and the remaining `Q` square to
/// `-1`. Coefficients are stored in blade order, where blade `i` is the bitmask
/// of the basis vectors it contains.
///
/// # Examples
///
/// ```
/// use geomalg::Cl;
///
/// let e1 = Cl::<2, 0>::basis(0);
/// let e2 = Cl::<2, 0>::basis(1);
/// assert_eq!(e1.geometric(&e2), e2.geometric(&e1).neg());
/// ```
#[derive(Clone, PartialEq)]
pub struct Cl<const P: usize, const Q: usize> {
    coeffs: Vec<f64>,
}

#[allow(clippy::should_implement_trait)]
impl<const P: usize, const Q: usize> Cl<P, Q> {
    /// Number of basis blades, `2^(P + Q)`.
    ///
    /// # Panics
    ///
    /// Panics when `P + Q > 63`, which the `usize` blade bitmask cannot index.
    pub fn blade_count() -> usize {
        let dim = P + Q;
        assert!(
            dim <= MAX_DIM,
            "Cl(P, Q) supports at most {MAX_DIM} basis vectors, got {dim}"
        );
        1usize << dim
    }

    /// The additive identity.
    pub fn zero() -> Self {
        Self {
            coeffs: vec![0.0; Self::blade_count()],
        }
    }

    /// A pure scalar `value` times the identity blade.
    pub fn scalar(value: f64) -> Self {
        let mut out = Self::zero();
        out.coeffs[0] = value;
        out
    }

    /// The unit basis vector `index`, i.e. `e1` for `0`, `e2` for `1`, and so
    /// on.
    ///
    /// # Panics
    ///
    /// Panics when `index >= P + Q`.
    pub fn basis(index: usize) -> Self {
        assert!(
            index < P + Q,
            "basis vector index {index} is out of range for Cl({P}, {Q})"
        );
        Self::blade(1 << index)
    }

    /// The unit basis blade at bitmask `mask`, where bit `j` selects `e(j+1)`.
    ///
    /// Blade `0` is the scalar, blade `1` is `e1`, blade `2` is `e2` and blade
    /// `3` is `e1e2`.
    ///
    /// # Panics
    ///
    /// Panics when `mask >= 2^(P + Q)`.
    pub fn blade(mask: usize) -> Self {
        assert!(
            mask < Self::blade_count(),
            "blade mask {mask} is out of range for Cl({P}, {Q})"
        );
        let mut out = Self::zero();
        out.coeffs[mask] = 1.0;
        out
    }

    /// A grade-1 multivector from one coefficient per basis vector.
    ///
    /// # Panics
    ///
    /// Panics when `values.len() != P + Q`.
    pub fn vector(values: &[f64]) -> Self {
        assert!(
            values.len() == P + Q,
            "vector needs {} coefficients, got {}",
            P + Q,
            values.len()
        );
        let mut out = Self::zero();
        for (axis, &value) in values.iter().enumerate() {
            out.coeffs[1 << axis] = value;
        }
        out
    }

    /// Coefficient of the blade at `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index >= 2^(P + Q)`.
    pub fn coefficient(&self, index: usize) -> f64 {
        self.coeffs[index]
    }

    /// All coefficients in blade order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    /// Sum of two multivectors.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(&other.coeffs)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Difference of two multivectors.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(&other.coeffs)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Additive inverse of a multivector.
    pub fn neg(&self) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|c| -c).collect(),
        }
    }

    /// Multiply every coefficient by the scalar `factor`.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|c| c * factor).collect(),
        }
    }

    /// The full geometric product `self * other`.
    ///
    /// Blades are multiplied one pair at a time, with the sign coming from the
    /// permutation that sorts the concatenated vectors and from the metric for
    /// every vector that appears twice.
    pub fn geometric(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            for (j, &b) in other.coeffs.iter().enumerate() {
                if b == 0.0 {
                    continue;
                }
                let (index, sign) = Self::blade_product(i, j);
                out.coeffs[index] += sign * a * b;
            }
        }
        out
    }

    /// Reversal: reverse the vector order inside every blade.
    ///
    /// A grade-`k` blade gains the sign `(-1)^(k(k - 1)/2)`.
    pub fn reverse(&self) -> Self {
        let mut out = self.clone();
        for (index, value) in out.coeffs.iter_mut().enumerate() {
            let grade = index.count_ones();
            if (grade * grade.saturating_sub(1) / 2) % 2 == 1 {
                *value = -*value;
            }
        }
        out
    }

    /// Project onto grade `k`, discarding every other grade.
    pub fn grade(&self, k: u32) -> Self {
        let mut out = Self::zero();
        for (index, &value) in self.coeffs.iter().enumerate() {
            if index.count_ones() == k {
                out.coeffs[index] = value;
            }
        }
        out
    }

    /// Euclidean norm of the multivector: `sqrt(|<self * reverse(self)>_0|)`.
    pub fn norm(&self) -> f64 {
        let square = self.geometric(&self.reverse()).coeffs[0];
        square.abs().sqrt()
    }

    /// Unit multivector in the same direction, or `None` when the norm is zero
    /// or not finite.
    pub fn normalized(&self) -> Option<Self> {
        let norm = self.norm();
        if norm == 0.0 || !norm.is_finite() {
            None
        } else {
            Some(self.scale(1.0 / norm))
        }
    }

    /// Rotor `exp(-(angle / 2) * e_a e_b)` in the plane spanned by basis
    /// vectors `plane_a` and `plane_b`.
    ///
    /// In a positive-definite plane this sandwich-rotates a vector by `angle`,
    /// turning `e_a` toward `e_b`. In a mixed-signature plane the exponential
    /// becomes the corresponding hyperbolic motion.
    ///
    /// # Panics
    ///
    /// Panics when an index is out of range or the two indices are equal.
    pub fn rotor(plane_a: usize, plane_b: usize, angle: f64) -> Self {
        let blades = Self::blade_count();
        assert!(
            plane_a < blades && plane_b < blades,
            "rotor plane index out of range for Cl({P}, {Q})"
        );
        assert!(
            plane_a != plane_b,
            "rotor needs two distinct plane basis vectors"
        );

        let bivector = Self::basis(plane_a).geometric(&Self::basis(plane_b));
        let square = bivector.geometric(&bivector).coeffs[0];
        let t = -angle / 2.0;

        let (cos_part, sin_part) = if square < 0.0 {
            let lambda = (-square).sqrt();
            ((lambda * t).cos(), (lambda * t).sin() / lambda)
        } else if square > 0.0 {
            let mu = square.sqrt();
            ((mu * t).cosh(), (mu * t).sinh() / mu)
        } else {
            (1.0, t)
        };

        Self::scalar(cos_part).add(&bivector.scale(sin_part))
    }

    /// Sandwich action `self * v * reverse(self)`.
    ///
    /// With a unit rotor this is a rotation; with a unit Pin it is a
    /// reflection or rotation-reflection.
    pub fn sandwich(&self, v: &Self) -> Self {
        self.geometric(v).geometric(&self.reverse())
    }

    /// Product of two basis blades: resulting blade index and `+/-1` sign.
    fn blade_product(a: usize, b: usize) -> (usize, f64) {
        let mut inversions = 0u32;
        let mut rest = a;
        while rest != 0 {
            let bit = rest.trailing_zeros();
            inversions += (b & ((1usize << bit) - 1)).count_ones();
            rest &= rest - 1;
        }

        let mut negative = inversions % 2 == 1;
        let mut shared = a & b;
        while shared != 0 {
            let bit = shared.trailing_zeros() as usize;
            if bit >= P {
                negative = !negative;
            }
            shared &= shared - 1;
        }

        (a ^ b, if negative { -1.0 } else { 1.0 })
    }
}

impl<const P: usize, const Q: usize> ops::Add for Cl<P, Q> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Cl::add(&self, &rhs)
    }
}

impl<const P: usize, const Q: usize> ops::Sub for Cl<P, Q> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Cl::sub(&self, &rhs)
    }
}

impl<const P: usize, const Q: usize> ops::Neg for Cl<P, Q> {
    type Output = Self;

    fn neg(self) -> Self {
        Cl::neg(&self)
    }
}

impl<const P: usize, const Q: usize> ops::Neg for &Cl<P, Q> {
    type Output = Cl<P, Q>;

    fn neg(self) -> Cl<P, Q> {
        Cl::neg(self)
    }
}

impl<const P: usize, const Q: usize> ops::Mul for Cl<P, Q> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.geometric(&rhs)
    }
}

impl<const P: usize, const Q: usize> ops::Mul<f64> for Cl<P, Q> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl<const P: usize, const Q: usize> fmt::Display for Cl<P, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut wrote = false;
        for (index, &value) in self.coeffs.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let negative = value.is_sign_negative();
            if wrote {
                write!(f, " {} ", if negative { "-" } else { "+" })?;
            } else if negative {
                write!(f, "-")?;
            }
            let magnitude = value.abs();
            if index == 0 || magnitude != 1.0 {
                write!(f, "{magnitude}")?;
            }
            if index != 0 {
                write!(f, "e")?;
                for bit in 0..(P + Q) {
                    if index & (1 << bit) != 0 {
                        write!(f, "{}", bit + 1)?;
                    }
                }
            }
            wrote = true;
        }
        if !wrote {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl<const P: usize, const Q: usize> fmt::Debug for Cl<P, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cl<{P}, {Q}>({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::Cl;

    #[test]
    fn blade_product_signs_follow_the_metric() {
        // e2 * e1 = -e1 * e2.
        assert_eq!(Cl::<2, 0>::blade_product(0b10, 0b01), (0b11, -1.0));
        // e1 * e1 = 1, e2 * e2 = 1.
        assert_eq!(Cl::<2, 0>::blade_product(0b01, 0b01), (0, 1.0));
        // In Cl(1, 3), e2 * e2 = -1.
        assert_eq!(Cl::<1, 3>::blade_product(0b10, 0b10), (0, -1.0));
        // Mixed-signature swap: e2 e1 = -e1 e2 still holds, and e2 e2 = -1.
        assert_eq!(Cl::<1, 3>::blade_product(0b10, 0b01), (0b11, -1.0));
    }

    #[test]
    fn display_and_debug_are_readable() {
        let m = Cl::<2, 0>::scalar(1.0)
            .add(&Cl::<2, 0>::basis(0).scale(2.0))
            .sub(&Cl::<2, 0>::blade(3));
        assert_eq!(format!("{m}"), "1 + 2e1 - e12");
        assert_eq!(format!("{:?}", Cl::<2, 0>::zero()), "Cl<2, 0>(0)");
    }

    #[test]
    fn normalized_rejects_zero_and_non_finite() {
        assert_eq!(Cl::<2, 0>::zero().normalized(), None);
        assert_eq!(Cl::<2, 0>::scalar(f64::NAN).normalized(), None);
        assert!(Cl::<2, 0>::vector(&[3.0, 4.0]).normalized().is_some());
    }
}
