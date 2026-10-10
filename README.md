# geomalg

Local draft spot. Not a crate yet.

Purpose: geometric algebra primitives: multivectors, geometric product, grade
projection, Pin/Spin actions, and metric signatures.

Crates.io exact-name check with `cargo info geomalg`: no crate found on
2026-06-26. PyPI and npm also returned 404 for `geomalg` on the same date.

First proof: represent a rotation or small multivector operation that removes
awkward conversion code from a geometry example.

Current verdict: keep this README-only. `skel/examples/rotor_rotation.rs`
already proves the tiny Cl(2, 0) rotor locally, and `skel::lie` owns SO/SE
geometry. A standalone crate is premature unless multivectors, metric
signatures, or sandwich actions recur outside that example.

Extraction rule: add a crate only when a geometry or Burn example needs the
primitive. Do not start with a broad symbolic algebra package.
