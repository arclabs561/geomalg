# Changelog

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
