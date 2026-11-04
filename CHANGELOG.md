# Changelog

## Unreleased

### Added

- `Cl::exp_bivector`: the exponential of a simple bivector (pure grade 2,
  square a scalar), with the trigonometric, hyperbolic and parabolic cases.
  Returns `None` for other input. `Cl::rotor` is the coordinate-plane case.
- `Rotor::from_even`: wraps any unit even multivector and precomputes its
  grade-1 action, so rotors outside coordinate planes (a boost toward a
  general direction, a product of rotors) get the fast `apply_vector` path.
  Returns `None` for odd parts or `R * reverse(R) != 1`.

## 0.1.1

### Added

- `ProductTable` and `Cl::table()`, with `geometric_with` and `geometric_into`
  taking the table or a caller-supplied output buffer.
- `Rotor` with a precomputed grade-1 action: `apply` for the sandwich product,
  `apply_vector` for one vector and `apply_vectors` for a flat batch.

### Changed

- `geometric` uses the per-signature table and skips zero coefficients.
  Release benchmarks: Cl(3,1) product 240 -> 145 ns, `geometric_into` 114 ns,
  batched rotor action 9.0 ns per point (26.6 ns single).

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

Initial release.

### Added

- `Cl<P, Q>` multivectors over `2^(P + Q)` basis blades with `f64`
  coefficients, bitmask blade indices, and a fixed metric signature: vectors
  `0..P` square to `+1`, the rest to `-1`.
- Constructors `zero`, `scalar`, `basis` (by vector index), `blade` (by
  bitmask) and `vector` (grade 1), with `coefficient` and `coefficients`
  accessors.
- Full geometric product with metric-aware blade signs, plus `add`, `sub`,
  `neg`, `scale`, `reverse`, `grade(k)`, `norm` and `normalized`.
- `rotor(a, b, angle)` as the exponential of `-(angle / 2) e_a e_b`, and the
  `sandwich` action for rotor and Pin transforms.
- `Display`/`Debug` as readable blade sums, and exact-`f64` `PartialEq`.
- `normalized` returns `None` for zero or non-finite norms.
