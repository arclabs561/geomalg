# geomalg

Geometric algebra primitives: multivectors, geometric product, grade
projection, Pin/Spin actions, and metric signatures.

`Cl<P, Q>` is a real Clifford algebra with a fixed metric: the first `P` basis
vectors square to `+1` and the remaining `Q` square to `-1`. Coefficients are
plain `f64`, one per basis blade, so a multivector is a value you can clone,
compare, and pass across a boundary without a trait object or an allocator
registry.

## Install

```toml
geomalg = "0.1"
```

```rust
use geomalg::Cl;
use std::f64::consts::FRAC_PI_2;

// Rotate e1 into e2 by pi/2 in the e1^e2 plane.
let rotor = Cl::<2, 0>::rotor(0, 1, FRAC_PI_2);
let rotated = rotor.sandwich(&Cl::<2, 0>::basis(0));

assert!((rotated.coefficient(0b10) - 1.0).abs() < 1e-12);
```

## What it is

- `Cl<P, Q>`: a multivector over `2^(P + Q)` basis blades. Blade `i` is the
  bitmask of the basis vectors it contains, in canonical increasing order, so
  `1` is `e1`, `2` is `e2` and `3` is `e1e2`.
- Construction: `zero`, `scalar`, `basis(i)` for basis vector `i`,
  `blade(mask)` for any blade bitmask, and `vector(&[f64])` for grade 1.
- Products: `geometric` (full product, metric-aware blade signs), `reverse`,
  `grade(k)`, `norm`, `normalized` (returns `None` for zero or non-finite
  norms), plus `add`, `sub`, `neg` and `scale`.
- Actions: `rotor(a, b, angle)` builds `exp(-(angle / 2) e_a e_b)` in the even
  subalgebra, and `sandwich(v)` computes `r v reverse(r)` for rotor and Pin
  actions. A positive-definite plane rotates by the requested angle.

## Boundary

This is a primitive, not a symbolic algebra system. There is no symbolic
simplification, no arbitrary or variable metric, no automatic differentiation,
and no general tensor or exterior-calculus layer. Blade indices are `usize`
bitmasks, so `P + Q` is capped at 63, and the coefficient type is fixed to
`f64` rather than generic.

If a consumer needs conformal geometry, arbitrary signatures per value, or
symbolic coefficients, that belongs in the consumer or in a later, separate
crate. This crate stays a dense numeric multivector.

## Performance

Release build, `cargo bench --bench product`, best of five batches, Apple M3 Max
(aarch64-apple-darwin, 16 cores), shared machine so compare ratios within a run
rather than absolute values across days.

| Operation | Cl(3,1) | Cl(2,0) |
| --- | --- | --- |
| `geometric` (dense × dense) | 145 ns | 35 ns |
| `geometric_into` (dense) | 114 ns | 13 ns |
| `Rotor::new` | 43 ns | 43 ns |
| `Rotor::apply_vector`, one call | 27 ns/point | — |
| `Rotor::apply_vectors`, batch of 1,000 | 9 ns/point | — |

Not claimed: other CPU tiers, threads, cold caches, or a multi-machine
comparison. Sparse inputs are slower than dense ones at Cl(3,1) (134 ns), because
the zero check costs more than it saves at that size.

## Testing

`just check` runs formatting, Clippy with `-D warnings`, the test suite and
docs. The tests cover basis-vector squares and anticommutation in `Cl(2, 0)`,
a negative-square vector in `Cl(1, 3)`, associativity in both signatures,
`v * reverse(v)` as the norm squared, rotor and Pin sandwiches, grade
projection, and `normalized` rejecting zero and non-finite input.

## License

MIT OR Apache-2.0.
