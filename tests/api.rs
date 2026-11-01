//! Behavioural tests for the `geomalg` public API.

use geomalg::Cl;
use std::f64::consts::FRAC_PI_2;

type Cl2 = Cl<2, 0>;
type Cl13 = Cl<1, 3>;

/// Compare two multivectors coefficient by coefficient.
fn assert_close<const P: usize, const Q: usize>(a: &Cl<P, Q>, b: &Cl<P, Q>, tol: f64) {
    assert_eq!(a.coefficients().len(), b.coefficients().len());
    for (index, (x, y)) in a.coefficients().iter().zip(b.coefficients()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "blade {index}: {x} != {y} (tolerance {tol})"
        );
    }
}

#[test]
fn cl2_basis_vectors_square_to_one() {
    let e1 = Cl2::basis(0);
    let e2 = Cl2::basis(1);

    assert_eq!(e1.geometric(&e1), Cl2::scalar(1.0));
    assert_eq!(e2.geometric(&e2), Cl2::scalar(1.0));
}

#[test]
fn cl2_basis_vectors_anticommute() {
    let e1 = Cl2::basis(0);
    let e2 = Cl2::basis(1);

    assert_eq!(e1.geometric(&e2), e2.geometric(&e1).neg());
    assert_ne!(e1.geometric(&e2), Cl2::zero());
}

#[test]
fn negative_square_basis_vector() {
    // Cl(1, 3): vector 0 is timelike-positive, vectors 1..=3 square to -1.
    let e2 = Cl13::basis(1);
    assert_eq!(e2.geometric(&e2), Cl13::scalar(-1.0));

    let e1 = Cl13::basis(0);
    assert_eq!(e1.geometric(&e1), Cl13::scalar(1.0));
}

#[test]
fn geometric_product_is_associative() {
    let a = Cl2::scalar(2.0)
        .add(&Cl2::basis(0))
        .sub(&Cl2::blade(3).scale(0.5));
    let b = Cl2::basis(1).add(&Cl2::blade(3));
    let c = Cl2::scalar(-1.5)
        .add(&Cl2::basis(0).scale(3.0))
        .add(&Cl2::basis(1))
        .add(&Cl2::blade(3));

    let left = a.geometric(&b).geometric(&c);
    let right = a.geometric(&b.geometric(&c));
    assert_close(&left, &right, 1e-12);

    // Same law in a mixed signature.
    let a = Cl13::basis(0).add(&Cl13::basis(1).scale(2.0));
    let b = Cl13::basis(2).sub(&Cl13::blade(0b1011));
    let c = Cl13::scalar(0.5).add(&Cl13::blade(0b0110));
    assert_close(
        &a.geometric(&b).geometric(&c),
        &a.geometric(&b.geometric(&c)),
        1e-12,
    );
}

#[test]
fn vector_times_reverse_is_norm_squared() {
    let v = Cl2::vector(&[3.0, 4.0]);
    let product = v.geometric(&v.reverse());

    assert_eq!(product, Cl2::scalar(25.0));
    assert_eq!(product.coefficient(0), v.norm() * v.norm());

    // Mixed signature: (1, 2) gives 1 - 4 = -3, and the norm uses |.|.
    let w = Cl13::vector(&[1.0, 2.0, 0.0, 0.0]);
    let product = w.geometric(&w.reverse());
    assert_eq!(product, Cl13::scalar(-3.0));
    assert!((w.norm() - 3.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn rotor_by_half_turn_of_plane_maps_e1_to_e2() {
    let rotor = Cl2::rotor(0, 1, FRAC_PI_2);
    let rotated = rotor.sandwich(&Cl2::basis(0));

    assert_close(&rotated, &Cl2::basis(1), 1e-9);
    assert!(rotor.coefficient(0).abs() > 0.0);
    assert_close(&rotor.geometric(&rotor.reverse()), &Cl2::scalar(1.0), 1e-12);
}

#[test]
fn grade_projection_keeps_only_requested_grade() {
    let mixed = Cl2::scalar(1.0)
        .add(&Cl2::basis(0))
        .add(&Cl2::basis(1).scale(2.0))
        .add(&Cl2::blade(3).scale(3.0));

    let bivector = mixed.grade(2);
    assert_eq!(bivector, Cl2::blade(3).scale(3.0));
    assert_eq!(bivector.grade(2), bivector);
    assert_eq!(bivector.grade(0), Cl2::zero());
    assert_eq!(mixed.grade(1), Cl2::vector(&[1.0, 2.0]));
}

#[test]
fn unit_rotor_sandwich_preserves_norm() {
    let rotor = Cl2::rotor(0, 1, 0.37);
    let unit = rotor.normalized().expect("rotor has non-zero norm");
    let v = Cl2::vector(&[3.0, 4.0]);

    let rotated = unit.sandwich(&v);
    assert!((rotated.norm() - v.norm()).abs() < 1e-12);
    assert!((rotated.norm() - 5.0).abs() < 1e-12);

    let angle = 0.37_f64;
    let expected = Cl2::vector(&[
        3.0 * angle.cos() - 4.0 * angle.sin(),
        3.0 * angle.sin() + 4.0 * angle.cos(),
    ]);
    assert_close(&rotated, &expected, 1e-12);
}

#[test]
fn reflector_sandwich_moves_the_axis_onto_the_other() {
    // Pin action: a unit vector reflects across the plane normal to it.
    let e1 = Cl2::basis(0);
    let e2 = Cl2::basis(1);

    assert_close(&e1.sandwich(&e2), &e2.neg(), 1e-12);
}

#[test]
fn normalized_rejects_null_and_degenerate_input() {
    assert_eq!(Cl2::zero().normalized(), None);
    assert_eq!(Cl2::scalar(f64::NAN).normalized(), None);
    assert_eq!(Cl2::scalar(f64::INFINITY).normalized(), None);
}

#[test]
#[should_panic]
fn basis_index_out_of_range_panics() {
    let _ = Cl2::basis(4);
}
