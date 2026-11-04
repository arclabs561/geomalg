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
use std::sync::OnceLock;

/// Maximum number of basis vectors supported by the `usize` blade bitmask.
const MAX_DIM: usize = 63;

/// Blade counts at or below this use the unrolled, fixed-size product kernel.
/// 64 blades is `Cl(6, 0)`, well past the dimensions this crate is used at.
const STACK_BLADES: usize = 64;

/// Number of `(P, Q)` cache slots. `P + Q <= 63` bounds both keys by 63, so the
/// slot index `P * 64 + Q` is unique and in range.
const TABLE_SLOTS: usize = 64 * 64;

/// Precomputed blade-product signs for one metric signature.
///
/// Every `Cl::<P, Q>` product needs the sign and resulting blade of each blade
/// pair. Computing them per term dominated the product; a table built once per
/// signature makes the product a single pass over the non-zero coefficients.
///
/// Get the table for a signature with [`Cl::table`] and reuse it with
/// [`Cl::geometric_with`] when a loop performs many products:
///
/// ```
/// use geomalg::Cl;
///
/// let table = Cl::<2, 0>::table();
/// let a = Cl::<2, 0>::basis(0);
/// let b = Cl::<2, 0>::basis(1);
/// assert_eq!(a.geometric_with(&b, table), a.geometric(&b));
/// ```
pub struct ProductTable {
    positive: usize,
    negative: usize,
    blade_count: usize,
    /// `signs[a * n + mask]` is the `+/-1` sign of blade `a` times blade
    /// `a ^ mask`. Storing by *result* mask keeps the hot loop's writes
    /// contiguous.
    signs: Box<[f64]>,
}

impl ProductTable {
    fn build<const P: usize, const Q: usize>() -> Self {
        let n = 1usize << (P + Q);
        let mut signs = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                let (mask, sign) = Cl::<P, Q>::blade_product(a, b);
                signs[a * n + mask] = sign;
            }
        }
        Self {
            positive: P,
            negative: Q,
            blade_count: n,
            signs: signs.into_boxed_slice(),
        }
    }

    /// The metric signature this table was built for.
    pub fn signature(&self) -> (usize, usize) {
        (self.positive, self.negative)
    }

    /// Number of basis blades, `2^(P + Q)`.
    pub fn blade_count(&self) -> usize {
        self.blade_count
    }

    #[inline]
    fn matches<const P: usize, const Q: usize>(&self) -> bool {
        self.positive == P && self.negative == Q
    }

    /// Accumulate `a * b` into the already-zeroed `out`.
    ///
    /// The summation order is the reference order: `a` ascending, then `b`
    /// ascending, with each term evaluated as `(sign * a_i) * b_j`. A given
    /// output blade is reached by exactly one `b` index per `a` index, so
    /// walking the result blade instead of `b` keeps that order intact.
    #[inline]
    fn accumulate(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        // A right operand with no zero coefficient needs no per-term test, and
        // the fixed-size kernels then keep the accumulator in registers.
        if b.iter().all(|&value| value != 0.0) {
            match self.blade_count {
                4 => self.dense::<4>(a, b, out),
                8 => self.dense::<8>(a, b, out),
                16 => self.dense::<16>(a, b, out),
                32 => self.dense::<32>(a, b, out),
                _ => self.sparse(a, b, out),
            }
        } else {
            self.sparse(a, b, out);
        }
    }

    /// Fully inlined product for a small, fully non-zero right operand.
    ///
    /// The loops are bounded by the constant `N`, so the indices and the
    /// `k ^ i` gather fold to constants, the accumulator stays in registers,
    /// and the destructuring below removes the per-term bounds checks.
    #[inline]
    fn dense<const N: usize>(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        let (Some(a), Some(b), Some(out)) = (fixed::<N>(a), fixed::<N>(b), fixed_mut::<N>(out))
        else {
            return;
        };
        let mut acc = [0.0f64; N];
        for i in 0..N {
            let ai = a[i];
            if ai == 0.0 {
                continue;
            }
            let row = &self.signs[i * N..i * N + N];
            for k in 0..N {
                acc[k] += row[k] * ai * b[k ^ i];
            }
        }
        out.copy_from_slice(&acc);
    }

    /// General product: skips zero coefficients on both sides.
    fn sparse(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = self.blade_count;
        for (i, &ai) in a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            let row = &self.signs[i * n..i * n + n];
            for (j, &bj) in b.iter().enumerate() {
                if bj == 0.0 {
                    continue;
                }
                let k = i ^ j;
                out[k] += row[k] * ai * bj;
            }
        }
    }
}

/// One cache slot per signature. A `static` inside a generic function would be
/// shared by every signature, so the cache is a fixed array indexed by the
/// signature instead.
static TABLES: [OnceLock<ProductTable>; TABLE_SLOTS] = [const { OnceLock::new() }; TABLE_SLOTS];

/// Convert a slice to a fixed-size reference, or `None` when the length is
/// wrong. Used once per product call, never per term.
#[inline]
fn fixed<const N: usize>(slice: &[f64]) -> Option<&[f64; N]> {
    slice.try_into().ok()
}

/// Mutable counterpart of [`fixed`].
#[inline]
fn fixed_mut<const N: usize>(slice: &mut [f64]) -> Option<&mut [f64; N]> {
    slice.try_into().ok()
}

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

    /// The precomputed blade product table for this signature, built once and
    /// shared by every value of the type.
    ///
    /// Pass it to [`Cl::geometric_with`] when a loop performs many products, so
    /// the table is not looked up per product.
    ///
    /// # Panics
    ///
    /// Panics when `P + Q > 63`, as [`Cl::blade_count`] does.
    pub fn table() -> &'static ProductTable {
        let dim = P + Q;
        assert!(
            dim <= MAX_DIM,
            "Cl(P, Q) supports at most {MAX_DIM} basis vectors, got {dim}"
        );
        TABLES[P * 64 + Q].get_or_init(ProductTable::build::<P, Q>)
    }

    /// The full geometric product `self * other`.
    ///
    /// Blades are multiplied one pair at a time, with the sign coming from the
    /// permutation that sorts the concatenated vectors and from the metric for
    /// every vector that appears twice. The sign and resulting blade index of
    /// each pair come from a table computed once per `(P, Q)`, so the product
    /// is a single pass over the non-zero coefficients.
    pub fn geometric(&self, other: &Self) -> Self {
        self.geometric_with(other, Self::table())
    }

    /// The full geometric product, using an explicit [`ProductTable`].
    ///
    /// Identical to [`Cl::geometric`], but a caller doing many products can
    /// hoist the table lookup. The table must have been built for `Cl(P, Q)`.
    ///
    /// # Panics
    ///
    /// Panics when `table` belongs to a different signature.
    pub fn geometric_with(&self, other: &Self, table: &ProductTable) -> Self {
        assert!(
            table.matches::<P, Q>(),
            "product table is for Cl({}, {}), not Cl({P}, {Q})",
            table.positive,
            table.negative
        );
        let mut out = Self::zero();
        table.accumulate(&self.coeffs, &other.coeffs, &mut out.coeffs);
        out
    }

    /// The full geometric product `self * other`, written into `out`.
    ///
    /// Equivalent to `*out = self.geometric(other)` but reuses the allocation
    /// already held by `out`, so a loop of products allocates nothing.
    pub fn geometric_into(&self, other: &Self, out: &mut Self) {
        out.coeffs.fill(0.0);
        Self::table().accumulate(&self.coeffs, &other.coeffs, &mut out.coeffs);
    }

    /// Add `other` into `self` in place, without allocating.
    pub fn add_assign(&mut self, other: &Self) {
        for (lhs, &rhs) in self.coeffs.iter_mut().zip(&other.coeffs) {
            *lhs += rhs;
        }
    }

    /// Multiply every coefficient by `factor` in place, without allocating.
    pub fn scale_assign(&mut self, factor: f64) {
        for value in &mut self.coeffs {
            *value *= factor;
        }
    }

    /// Copy `a` into `out` with every blade reversed.
    #[inline]
    fn reverse_into(a: &[f64], out: &mut [f64]) {
        for (index, (src, dst)) in a.iter().zip(out).enumerate() {
            let grade = index.count_ones();
            *dst = if (grade * grade.saturating_sub(1) / 2) % 2 == 1 {
                -*src
            } else {
                *src
            };
        }
    }

    /// Reversal: reverse the vector order inside every blade.
    ///
    /// A grade-`k` blade gains the sign `(-1)^(k(k - 1)/2)`.
    pub fn reverse(&self) -> Self {
        let mut out = self.clone();
        Self::reverse_into(&self.coeffs, &mut out.coeffs);
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

        let (mask, sign) = Self::blade_product(1 << plane_a, 1 << plane_b);
        // `(e_a e_b)^2` is a scalar, so its own table entry is the square.
        let (_, square) = Self::blade_product(mask, mask);
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

        let mut out = Self::zero();
        out.coeffs[0] = cos_part;
        out.coeffs[mask] = sign * sin_part;
        out
    }

    /// Exponential of a simple bivector: `exp(B)` for a pure grade-2 `B` whose
    /// square is a scalar.
    ///
    /// `B^2 = s` gives `cos(sqrt(-s)) + sin(sqrt(-s)) / sqrt(-s) * B` for
    /// `s < 0`, `cosh(sqrt(s)) + sinh(sqrt(s)) / sqrt(s) * B` for `s > 0`, and
    /// `1 + B` for `s = 0`. With `B = -(angle / 2) * e_a e_b` this is
    /// [`Cl::rotor`]; any other plane, such as a boost toward a general
    /// direction, works the same way, which is how to build a rotor that is not
    /// aligned with a coordinate plane.
    ///
    /// Returns `None` when `self` has parts outside grade 2, or when its square
    /// is not a scalar (a non-simple bivector, possible from four dimensions
    /// up). Both checks use a tolerance of `1e-12` relative to the bivector's
    /// coefficient scale.
    pub fn exp_bivector(&self) -> Option<Self> {
        let scale = self.coeffs.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        let tol = 1e-12 * scale.max(1.0);
        for (mask, &c) in self.coeffs.iter().enumerate() {
            if mask.count_ones() != 2 && c.abs() > tol {
                return None;
            }
        }
        let square = self.geometric(self);
        let square_tol = 1e-12 * (scale * scale).max(1.0);
        if square.coeffs[1..].iter().any(|c| c.abs() > square_tol) {
            return None;
        }
        let s = square.coeffs[0];
        let (cos_part, sin_part) = if s < -square_tol {
            let lambda = (-s).sqrt();
            (lambda.cos(), lambda.sin() / lambda)
        } else if s > square_tol {
            let mu = s.sqrt();
            (mu.cosh(), mu.sinh() / mu)
        } else {
            (1.0, 1.0)
        };
        let mut out = self.grade(2).scale(sin_part);
        out.coeffs[0] = cos_part;
        Some(out)
    }

    /// Sandwich action `self * v * reverse(self)`.
    ///
    /// With a unit rotor this is a rotation; with a unit Pin it is a
    /// reflection or rotation-reflection.
    pub fn sandwich(&self, v: &Self) -> Self {
        let n = self.coeffs.len();
        let table = Self::table();
        let mut out = Self::zero();
        if n <= STACK_BLADES {
            // The intermediate `self * v` and the reversal are scratch: only
            // the result reaches the heap.
            let mut mid = [0.0f64; STACK_BLADES];
            let mut reversed = [0.0f64; STACK_BLADES];
            let mid = &mut mid[..n];
            let reversed = &mut reversed[..n];
            table.accumulate(&self.coeffs, &v.coeffs, mid);
            Self::reverse_into(&self.coeffs, reversed);
            table.accumulate(mid, reversed, &mut out.coeffs);
        } else {
            let mid = self.geometric(v);
            table.accumulate(&mid.coeffs, &self.reverse().coeffs, &mut out.coeffs);
        }
        out
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

/// A rotor with its grade-1 action precomputed.
///
/// Applying a rotor to a vector is linear, so the sandwich product for each
/// basis vector is computed once. [`Rotor::apply_vector`] then costs `N * N`
/// multiply-adds instead of two geometric products.
#[derive(Clone, Debug)]
pub struct Rotor<const P: usize, const Q: usize> {
    even: Cl<P, Q>,
    action: Vec<f64>,
}

impl<const P: usize, const Q: usize> Rotor<P, Q> {
    /// Builds the rotor for the `plane_a`/`plane_b` plane and precomputes its
    /// action on grade-1 vectors.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Cl::rotor`].
    pub fn new(plane_a: usize, plane_b: usize, angle: f64) -> Self {
        Self::with_action(Cl::<P, Q>::rotor(plane_a, plane_b, angle))
    }

    /// Wraps any rotor multivector and precomputes its action on grade-1
    /// vectors, for rotors that are not aligned with a coordinate plane (for
    /// example [`Cl::exp_bivector`] of a general bivector, or a product of
    /// rotors).
    ///
    /// Returns `None` unless `even` has only even-grade parts and
    /// `even * reverse(even)` is `1` (both within `1e-9`).
    pub fn from_even(even: Cl<P, Q>) -> Option<Self> {
        let tol = 1e-9;
        if even
            .coeffs
            .iter()
            .enumerate()
            .any(|(mask, c)| mask.count_ones() % 2 == 1 && c.abs() > tol)
        {
            return None;
        }
        let unit = even.geometric(&even.reverse());
        if (unit.coeffs[0] - 1.0).abs() > tol || unit.coeffs[1..].iter().any(|c| c.abs() > tol) {
            return None;
        }
        Some(Self::with_action(even))
    }

    fn with_action(even: Cl<P, Q>) -> Self {
        let n = P + Q;
        let mut action = vec![0.0; n * n];
        for j in 0..n {
            let image = even.sandwich(&Cl::<P, Q>::basis(j));
            for i in 0..n {
                action[i * n + j] = image.coefficient(1 << i);
            }
        }
        Self { even, action }
    }

    /// The normalized even multivector behind this rotor.
    pub fn even(&self) -> &Cl<P, Q> {
        &self.even
    }

    /// Applies the rotor to an arbitrary multivector via the sandwich product.
    pub fn apply(&self, v: &Cl<P, Q>) -> Cl<P, Q> {
        self.even.sandwich(v)
    }

    /// Applies the rotor to a grade-1 vector using the precomputed action.
    ///
    /// # Panics
    ///
    /// Panics when `v.len() != P + Q`.
    pub fn apply_vector(&self, v: &[f64]) -> Vec<f64> {
        let n = P + Q;
        assert_eq!(v.len(), n, "vector length must be P + Q");
        let mut out = vec![0.0; n];
        self.apply_vectors_into(v, &mut out);
        out
    }

    /// Applies the rotor to `input.len() / N` vectors, writing to `out`.
    ///
    /// # Panics
    ///
    /// Panics when the lengths are not whole multiples of `P + Q` or differ.
    pub fn apply_vectors(&self, input: &[f64], out: &mut [f64]) {
        let n = P + Q;
        assert_eq!(input.len() % n, 0, "input must hold whole vectors");
        assert_eq!(out.len(), input.len(), "out must match input length");
        for (src, dst) in input.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            self.apply_vectors_into(src, dst);
        }
    }

    fn apply_vectors_into(&self, src: &[f64], dst: &mut [f64]) {
        let n = P + Q;
        for (i, slot) in dst.iter_mut().enumerate() {
            let row = &self.action[i * n..i * n + n];
            *slot = row.iter().zip(src).map(|(a, b)| a * b).sum();
        }
    }
}

#[cfg(test)]
mod rotor_tests {
    use super::*;

    #[test]
    fn vector_action_matches_the_sandwich_product() {
        let rotor = Rotor::<2, 0>::new(0, 1, std::f64::consts::FRAC_PI_3);
        for v in [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, -1.3]] {
            let image = rotor.apply(&Cl::<2, 0>::vector(&v));
            let actual = rotor.apply_vector(&v);
            for (i, a) in actual.iter().enumerate() {
                let b = image.coefficient(1 << i);
                assert!((a - b).abs() < 1e-12, "component {i}: {a} != {b}");
            }
        }
    }

    #[test]
    fn batch_matches_the_single_call() {
        let rotor = Rotor::<3, 1>::new(0, 1, 0.9);
        let mut input = Vec::new();
        for k in 0..16 {
            for j in 0..4 {
                input.push(((k * 4 + j) as f64 * 0.37).sin());
            }
        }
        let mut out = vec![0.0; input.len()];
        rotor.apply_vectors(&input, &mut out);
        for (chunk_in, chunk_out) in input.chunks_exact(4).zip(out.chunks_exact(4)) {
            let single = rotor.apply_vector(chunk_in);
            for (a, b) in chunk_out.iter().zip(&single) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exp_bivector_reproduces_coordinate_plane_rotors() {
        for (a, b) in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)] {
            for angle in [-2.3, -0.4, 0.0, 0.7, 1.9] {
                let (mask, sign) = Cl::<3, 1>::blade_product(1 << a, 1 << b);
                let mut bivector = Cl::<3, 1>::zero();
                bivector.coeffs[mask] = -sign * angle / 2.0;
                let via_exp = bivector.exp_bivector().expect("simple bivector");
                let direct = Cl::<3, 1>::rotor(a, b, angle);
                for (x, y) in via_exp.coefficients().iter().zip(direct.coefficients()) {
                    assert!((x - y).abs() < 1e-14, "plane ({a},{b}) angle {angle}");
                }
            }
        }
    }

    #[test]
    fn exp_bivector_rejects_non_bivectors_and_non_simple_ones() {
        assert!(Cl::<3, 1>::vector(&[1.0, 0.0, 0.0, 0.0])
            .exp_bivector()
            .is_none());
        // e1e2 + e3e4 squares to -2 + 2 e1e2e3e4, which is not a scalar.
        let mut non_simple = Cl::<4, 0>::zero();
        non_simple.coeffs[0b0011] = 1.0;
        non_simple.coeffs[0b1100] = 1.0;
        assert!(non_simple.exp_bivector().is_none());
    }

    #[test]
    fn boost_toward_a_general_direction_from_even() {
        // Boost by rapidity theta along u = (0.6, 0.8, 0) in Cl(3, 1), time = e4.
        let theta = 0.9;
        let (ux, uy) = (0.6, 0.8);
        let mut bivector = Cl::<3, 1>::zero();
        for (axis, weight) in [(0usize, ux), (1usize, uy)] {
            let (mask, sign) = Cl::<3, 1>::blade_product(1 << axis, 1 << 3);
            bivector.coeffs[mask] += -sign * weight * theta / 2.0;
        }
        let even = bivector.exp_bivector().expect("simple bivector");
        let rotor = Rotor::from_even(even).expect("unit rotor");

        let minkowski = |v: &[f64]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - v[3] * v[3];
        for v in [
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.3, -0.2, 0.5, 1.4],
            vec![1.0, 0.0, 0.0, 0.0],
        ] {
            let out = rotor.apply_vector(&v);
            let image = rotor.apply(&Cl::<3, 1>::vector(&v));
            for (i, a) in out.iter().enumerate() {
                assert!((a - image.coefficient(1 << i)).abs() < 1e-12);
            }
            assert!((minkowski(&out) - minkowski(&v)).abs() < 1e-12);
        }
        // The time axis goes to cosh(theta) in time and sinh(theta) along +/-u.
        let t = rotor.apply_vector(&[0.0, 0.0, 0.0, 1.0]);
        assert!((t[3] - theta.cosh()).abs() < 1e-12);
        assert!((t[0].abs() - ux * theta.sinh()).abs() < 1e-12);
        assert!((t[1].abs() - uy * theta.sinh()).abs() < 1e-12);
        assert!(t[0] * t[1] > 0.0 && t[2].abs() < 1e-12);
    }

    #[test]
    fn from_even_rejects_odd_and_non_unit_input() {
        assert!(Rotor::<3, 1>::from_even(Cl::vector(&[1.0, 0.0, 0.0, 0.0])).is_none());
        assert!(Rotor::<3, 1>::from_even(Cl::scalar(2.0)).is_none());
        assert!(Rotor::<3, 1>::from_even(Cl::scalar(1.0)).is_some());
    }

    #[test]
    fn rotor_preserves_vector_norm() {
        let rotor = Rotor::<2, 0>::new(0, 1, 1.1);
        let out = rotor.apply_vector(&[3.0, 4.0]);
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 5.0).abs() < 1e-12);
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
