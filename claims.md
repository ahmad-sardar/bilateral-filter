# Claim Register — Bilateral Filter Carrier

Each claim has: type, object, property, oracle, evidence, scope,
and error classification (what kind of discrepancy a failure would be).

---

## C0-01: Local convex hull

Type:     verification (math property)
Object:   B on arbitrary float64 input within contract
Property: for all p, min_{q in N(p)} I(q) <= B(I)_p <= max_{q in N(p)} I(q)
Oracle:   lemma — weights non-negative, sum positive, therefore convex combination
Evidence: property test on random inputs (TestAnalyticOracles.test_output_in_local_range)
Scope:    float64, skip-border, R >= 1
Failure classification: implementation error

---

## C0-02: Constant preservation (interior)

Type:     verification (math property)
Object:   B applied to constant image, interior pixels
Property: B(c)_p = c for all p with N(p) entirely within the image
Oracle:   all neighbors identical => range weights all 1 => weighted average of c = c
Evidence: analytic oracle test at values 0.0, 0.25, 0.5, 0.75, 1.0
Scope:    float64, skip-border, interior pixels only
Failure classification: rounding error if <= machine epsilon; implementation error if larger
Observed: exact zero for 0.0, 0.25, 0.5, 1.0; half-epsilon (1.11e-16) for 0.75.
          Classified as rounding error in the multiply-divide chain.

---

## C0-03: Gaussian limit

Type:     verification (math property)
Object:   B with sigma_r -> infinity
Property: B(I)_p -> G(I)_p where G is the normalized spatial Gaussian average
Oracle:   range weight -> 1 for all finite intensity differences
Evidence: comparison to independent Gaussian reference at sigma_r = 1e6
Scope:    float64, skip-border
Failure classification: implementation error
Oracle independence: reference must be computed through a different code
                     path than the bilateral filter (see note in tests)

---

## C0-04: Determinism

Type:     verification (implementation contract)
Object:   bilateral_filter function
Property: identical inputs produce bit-identical outputs
Oracle:   hash comparison across two runs
Evidence: SHA-256 match on output.npy
Scope:    same platform, same numpy version, same dtype
Failure classification: implementation error (non-deterministic operation)

---

## C0-05: Input validation

Type:     verification (implementation contract)
Object:   bilateral_filter function
Property: rejects 3D input, non-positive sigma, R < 1
Oracle:   carrier contract (parameter constraints section)
Evidence: ValueError raised on each invalid case
Scope:    all inputs outside contract
Failure classification: specification error (code does not match contract)

---

## C0-06: Truncation convergence

Type:     truncation convergence
Object:   B at finite R vs B at R = 3*sigma_s
Refinement variable: R (kernel radius)
Error norm: max absolute pixel difference on interior
Expected behavior: monotone decrease, bounded by spatial weight at
                   distance R, which is exp(-R^2 / (2*sigma_s^2))
Evidence: convergence table at R = 2, 3, 5, 7
Scope:    float64, sigma_s = 2.0, sigma_r = 0.1
Failure classification: if non-monotone, implementation error or
                        rounding interaction

---

## C0-07: Precision sensitivity

Type:     precision sensitivity (not correctness verification)
Object:   B at float64 input vs float32-roundtripped input
Refinement variable: input dtype
Error norm: max absolute pixel difference
Property: max diff < 1e-6
Oracle:   measured envelope from perturbation drill (actual ~2.5e-8)
Evidence: precision envelope test
Scope:    sigma_s = 3.0, sigma_r = 0.1, R = 3, seed 42
Failure classification: cross-layer contract error (input dtype differs
                        from contract) producing rounding propagation
Tolerance rationale: 1e-6 is 40x above measurement (2.5e-8), well below
                     input resolution (4e-3). Not arbitrary.

---

## C0-08: Identity limit

Type:     verification (parameter limit)
Object:   B with sigma_r -> 0
Property: B(I)_p -> I(p) when I(p) != I(q) for all q != p in N(p)
Oracle:   range weight -> 0 for all q != p; self-weight stays 1
Evidence: max_diff_from_input at sigma_r = 0.1, 0.01, 0.001, 0.0001
Scope:    float64, skip-border, non-constant neighborhoods
Failure classification: implementation error if does not approach identity
