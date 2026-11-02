//! Release-mode microbenchmarks for the `geomalg` products.
//!
//! Run with `cargo bench` (or `just bench`). Every case reports `ns/op` on
//! stdout so CI logs carry the numbers. The harness is deliberately a plain
//! `Instant` loop with no dependency: calibrate a batch until one batch takes at
//! least [`MIN_BATCH`], then keep the best of [`ROUNDS`] batches.
//!
//! `dense`/`sparse` occupancy means "every coefficient of the multivector is
//! non-zero" and "a handful of coefficients are non-zero"; the sparse case is
//! what a rotor or a vector actually looks like.

use geomalg::Cl;
use std::hint::black_box;
use std::time::{Duration, Instant};

const MIN_BATCH: Duration = Duration::from_millis(50);
const ROUNDS: u32 = 5;
const MAX_BATCH: u64 = 20_000_000;

/// Best `ns/op` of [`ROUNDS`] batches after calibrating the batch size.
fn time<F, R>(mut op: F) -> f64
where
    F: FnMut() -> R,
{
    let mut batch = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..batch {
            black_box(op());
        }
        let elapsed = start.elapsed();
        if elapsed >= MIN_BATCH || batch >= MAX_BATCH {
            break;
        }
        let target = MIN_BATCH.as_nanos() as f64;
        let factor = (target / elapsed.as_nanos().max(1) as f64).max(2.0);
        batch = ((batch as f64 * factor) as u64)
            .clamp(batch + 1, MAX_BATCH)
            .min(MAX_BATCH);
    }

    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..batch {
            black_box(op());
        }
        let ns = start.elapsed().as_nanos() as f64 / batch as f64;
        best = best.min(ns);
    }
    best
}

fn report(case: &str, ns: f64) {
    println!("{case:<34} {ns:>9.1} ns/op");
}

/// A deterministic pseudo-random value in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Every coefficient non-zero.
fn dense<const P: usize, const Q: usize>(seed: u64) -> Cl<P, Q> {
    let mut rng = Lcg(seed);
    let mut out = Cl::<P, Q>::zero();
    for mask in 0..Cl::<P, Q>::blade_count() {
        out = out.add(&Cl::<P, Q>::blade(mask).scale(rng.next()));
    }
    out
}

/// A grade-1 vector plus one bivector: the shape a rotor or a point actually
/// has.
fn sparse<const P: usize, const Q: usize>(seed: u64) -> Cl<P, Q> {
    let mut rng = Lcg(seed);
    let vector: Vec<f64> = (0..P + Q).map(|_| rng.next()).collect();
    Cl::<P, Q>::vector(&vector).add(&Cl::<P, Q>::blade(0b11).scale(rng.next()))
}

fn bench_geometric<const P: usize, const Q: usize>(case: &str, seed: u64) {
    let dense_a = dense::<P, Q>(seed);
    let dense_b = dense::<P, Q>(seed + 1);
    let sparse_a = sparse::<P, Q>(seed + 2);
    let sparse_b = sparse::<P, Q>(seed + 3);

    report(
        &format!("{case} geometric dense*dense"),
        time(|| dense_a.geometric(&dense_b)),
    );
    report(
        &format!("{case} geometric sparse*sparse"),
        time(|| sparse_a.geometric(&sparse_b)),
    );
    // Same product into a reused destination: kernel cost with no allocation
    // and no table lookup per call.
    let mut out = Cl::<P, Q>::zero();
    report(
        &format!("{case} geometric_into dense"),
        time(|| {
            dense_a.geometric_into(&dense_b, &mut out);
            black_box(&out);
        }),
    );
}

fn main() {
    println!("geomalg benchmarks (release, best of {ROUNDS} batches)");
    println!("{:─<48}", "");
    report("table lookup Cl(3,1)", time(Cl::<3, 1>::table));
    report("alloc Cl(3,1)::zero", time(Cl::<3, 1>::zero));
    println!("{:─<48}", "");
    bench_geometric::<2, 0>("Cl(2,0)", 0x9e37_79b9_7f4a_7c15);
    bench_geometric::<3, 0>("Cl(3,0)", 0xc2b2_ae3d_27d4_eb4f);
    bench_geometric::<3, 1>("Cl(3,1)", 0x1656_67b1_9e37_79f9);
    bench_geometric::<1, 3>("Cl(1,3)", 0x27d4_eb2f_1656_67b1);
    println!("{:─<48}", "");

    let angle = 0.37;
    report(
        "Cl(2,0) rotor",
        time(|| Cl::<2, 0>::rotor(0, 1, black_box(angle))),
    );
    report(
        "Cl(3,1) rotor",
        time(|| Cl::<3, 1>::rotor(0, 1, black_box(angle))),
    );
    println!("{:─<48}", "");

    // Sandwich is the operation a consumer actually pays for: geometric with a
    // dense rotor body twice, plus the reversal.
    let rotor2 = Cl::<2, 0>::rotor(0, 1, angle);
    let point2 = sparse::<2, 0>(0xabcd);
    report("Cl(2,0) sandwich", time(|| rotor2.sandwich(&point2)));

    let rotor31 = Cl::<3, 1>::rotor(0, 1, angle);
    let point31 = sparse::<3, 1>(0xabce);
    report("Cl(3,1) sandwich", time(|| rotor31.sandwich(&point31)));

    // A dense multivector sandwich, i.e. the worst case for occupancy.
    let dense31 = dense::<3, 1>(0xabcf);
    report(
        "Cl(3,1) sandwich dense",
        time(|| rotor31.sandwich(&dense31)),
    );

    // Batched rotor action: the realistic transport pattern (one rotor, many
    // points), compared against the per-call path and the matrix baseline.
    println!("{:─<48}", "");
    let rotor = geomalg::Rotor::<3, 1>::new(0, 1, angle);
    let mut points = Vec::new();
    for k in 0..1000 {
        for j in 0..4 {
            points.push(((k * 4 + j) as f64 * 0.13).sin());
        }
    }
    let mut out = vec![0.0; points.len()];
    let batch_ns = time(|| {
        rotor.apply_vectors(&points, &mut out);
        black_box(&out);
    });
    println!(
        "Cl(3,1) rotor apply_vectors x1000          {:>7.1} ns/point",
        batch_ns / 1000.0
    );
    let loop_ns = time(|| {
        for chunk in points.chunks_exact(4) {
            black_box(rotor.apply_vector(chunk));
        }
    });
    println!(
        "Cl(3,1) rotor apply_vector loop x1000      {:>7.1} ns/point",
        loop_ns / 1000.0
    );
    println!("{:─<48}", "");
}
