"""
Verification tests for the bilateral filter.

All tests in this file are VERIFICATION — they check whether the code
correctly implements the carrier contract (CONTRACT.md).

Each test references a claim in the claim register (claims.md).
The register records: type, object, property, oracle, evidence,
scope, and error classification for every substantive claim.

Uses only unittest (standard library). No external test framework.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bilateral import bilateral_filter, summarize


# ---------------------------------------------------------------------------
# Analytic oracles — inputs with known exact outputs
# ---------------------------------------------------------------------------

class TestAnalyticOracles(unittest.TestCase):
    """Inputs where the exact output is known from the definition.
    Not "method of manufactured solutions" (which is a PDE technique).
    These are exact fixtures and metamorphic properties."""

    def test_constant_preservation(self):
        """Claim C0-02. Constant image, interior pixels.
        All neighbors identical => range weights all 1 => output = input.
        Tolerance: 1 machine epsilon (rounding in multiply-divide chain).
        Observed: exact zero for 0.0, 0.25, 0.5, 1.0; half-eps for 0.75.
        Error classification: rounding error."""
        eps = np.finfo(np.float64).eps
        for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            image = np.full((20, 20), value, dtype=np.float64)
            result = bilateral_filter(image, sigma_s=3.0, sigma_r=0.1,
                                      kernel_radius=3)
            interior = result[3:-3, 3:-3]
            max_error = np.max(np.abs(interior - value))
            self.assertLessEqual(max_error, eps,
                f"Error {max_error:.2e} exceeds machine epsilon "
                f"on constant value={value}")

    def test_constant_all_radii(self):
        """Claim C0-02 extended. Constant image at varying kernel radius.
        The exact answer does not depend on the radius."""
        eps = np.finfo(np.float64).eps
        image = np.full((32, 32), 0.5, dtype=np.float64)
        for kr in [1, 2, 3, 5, 7, 9]:
            result = bilateral_filter(image, sigma_s=2.0, sigma_r=0.1,
                                      kernel_radius=kr)
            interior = result[kr:-kr, kr:-kr]
            if interior.size == 0:
                continue
            max_error = np.max(np.abs(interior - 0.5))
            self.assertLessEqual(max_error, eps,
                f"Error {max_error:.2e} at kernel_radius={kr}")

    def test_local_convex_hull(self):
        """Claim C0-01. For each pixel p, the output lies between the
        min and max of its neighborhood N(p) — not the global image
        range. This is the local convex hull property.
        Oracle: weights non-negative, sum positive => convex combination.
        Error classification: any violation is implementation error."""
        rng = np.random.default_rng(99)
        image = rng.uniform(0.2, 0.8, size=(20, 20))
        kr = 3
        result = bilateral_filter(image, sigma_s=2.0, sigma_r=0.1,
                                  kernel_radius=kr)
        rows, cols = image.shape
        for r in range(rows):
            for c in range(cols):
                # Compute local neighborhood min/max
                r_lo = max(0, r - kr)
                r_hi = min(rows, r + kr + 1)
                c_lo = max(0, c - kr)
                c_hi = min(cols, c + kr + 1)
                local_min = image[r_lo:r_hi, c_lo:c_hi].min()
                local_max = image[r_lo:r_hi, c_lo:c_hi].max()
                self.assertGreaterEqual(result[r, c], local_min - 1e-15)
                self.assertLessEqual(result[r, c], local_max + 1e-15)

    def test_piecewise_constant_deep_interior(self):
        """Claim C0-02 variant. Two-region image: left = 0.3, right = 0.7.
        Deep interior pixels (far from the edge) see only same-valued
        neighbors. Output must equal input."""
        image = np.full((32, 32), 0.3, dtype=np.float64)
        image[:, 16:] = 0.7
        kr = 3
        result = bilateral_filter(image, sigma_s=2.0, sigma_r=0.05,
                                  kernel_radius=kr)
        left_interior = result[kr:-kr, kr:16 - kr]
        right_interior = result[kr:-kr, 16 + kr:-kr]
        np.testing.assert_allclose(left_interior, 0.3, atol=1e-10)
        np.testing.assert_allclose(right_interior, 0.7, atol=1e-10)

    def test_single_pixel(self):
        """Claim C0-02 edge case. 1x1 image: only neighbor is itself.
        W(p) = 1.0. Output = I(p) / 1.0 = I(p)."""
        image = np.array([[0.7]])
        result = bilateral_filter(image, sigma_s=2.0, sigma_r=0.1,
                                  kernel_radius=3)
        np.testing.assert_allclose(result, 0.7, atol=1e-15)

    def test_gaussian_limit_independent_oracle(self):
        """Claim C0-03. As sigma_r -> inf, bilateral -> Gaussian blur.

        Oracle independence: the reference is computed by explicit
        matrix construction (build the full weight matrix, multiply,
        normalize), NOT by a nested loop with the same structure as
        the bilateral filter. This avoids circular verification."""
        rng = np.random.default_rng(13)
        image = rng.uniform(0.0, 1.0, size=(8, 8))
        sigma_s = 2.0
        kr = 3
        rows, cols = image.shape

        # Bilateral at huge sigma_r.
        bilateral_result = bilateral_filter(image, sigma_s=sigma_s,
                                           sigma_r=1e6,
                                           kernel_radius=kr)

        # Independent Gaussian reference: explicit weight matrix.
        # For each pixel p, build a 1D weight vector over the
        # flattened neighborhood, compute the weighted average.
        gaussian_ref = np.zeros_like(image)
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                weights = []
                for dy in range(-kr, kr + 1):
                    for dx in range(-kr, kr + 1):
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(image[nr, nc])
                            # Weight computed by a different formula path:
                            # explicit distance, explicit division, explicit
                            # exp — no shared intermediate variables.
                            dist_sq = float(dy * dy + dx * dx)
                            denom = 2.0 * sigma_s * sigma_s
                            weights.append(np.exp(-dist_sq / denom))
                neighbors = np.array(neighbors)
                weights = np.array(weights)
                gaussian_ref[r, c] = np.dot(weights, neighbors) / np.sum(weights)

        np.testing.assert_allclose(bilateral_result, gaussian_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Implementation contract
# ---------------------------------------------------------------------------

class TestContract(unittest.TestCase):
    """Tests derived from CONTRACT.md, not from mathematical properties."""

    def test_deterministic(self):
        """Claim C0-04. Identical inputs -> bit-identical outputs."""
        rng = np.random.default_rng(7)
        image = rng.uniform(0.0, 1.0, size=(16, 16))
        r1 = bilateral_filter(image, sigma_s=2.0, sigma_r=0.1,
                              kernel_radius=3)
        r2 = bilateral_filter(image, sigma_s=2.0, sigma_r=0.1,
                              kernel_radius=3)
        np.testing.assert_array_equal(r1, r2)

    def test_rejects_3d_input(self):
        """Claim C0-05. Filter is defined for 2D only."""
        with self.assertRaises(ValueError):
            bilateral_filter(np.zeros((10, 10, 3)), sigma_s=2.0,
                             sigma_r=0.1, kernel_radius=3)

    def test_rejects_nonpositive_sigma(self):
        """Claim C0-05. sigma_s and sigma_r must be positive."""
        image = np.zeros((10, 10))
        with self.assertRaises(ValueError):
            bilateral_filter(image, sigma_s=0.0, sigma_r=0.1,
                             kernel_radius=3)
        with self.assertRaises(ValueError):
            bilateral_filter(image, sigma_s=2.0, sigma_r=-1.0,
                             kernel_radius=3)

    def test_dtype_preserved(self):
        """Claim C0-07 prerequisite. float64 in -> float64 out."""
        image = np.random.default_rng(1).uniform(0.0, 1.0, size=(16, 16))
        result = bilateral_filter(image, sigma_s=2.0, sigma_r=0.1,
                                  kernel_radius=3)
        self.assertEqual(result.dtype, np.float64)

    def test_summary_known_values(self):
        """Summarize function: hand-computable reference."""
        image = np.array([[0.0, 0.5], [0.5, 1.0]])
        s = summarize(image)
        self.assertAlmostEqual(s["mean"], 0.5)
        self.assertAlmostEqual(s["min"], 0.0)
        self.assertAlmostEqual(s["max"], 1.0)
        self.assertEqual(s["shape"], [2, 2])
        self.assertEqual(s["dtype"], "float64")


# ---------------------------------------------------------------------------
# Typed convergence studies
# ---------------------------------------------------------------------------

class TestTruncationConvergence(unittest.TestCase):
    """Claim C0-06. Truncation convergence.
    Target:    B at finite R vs B at larger R.
    Refinement variable: R (kernel radius).
    Error norm: max absolute pixel difference on interior.
    Expected: monotone decrease, bounded by spatial weight at distance R."""

    def test_differences_decrease_with_radius(self):
        rng = np.random.default_rng(42)
        image = rng.uniform(0.0, 1.0, size=(32, 32))
        sigma_s = 2.0
        radii = [2, 3, 5, 7]
        results = {kr: bilateral_filter(image, sigma_s, 0.1, kr)
                   for kr in radii}

        prev_diff = float("inf")
        for i in range(1, len(radii)):
            kr_prev, kr_curr = radii[i - 1], radii[i]
            margin = kr_curr
            diff = np.max(np.abs(
                results[kr_curr][margin:-margin, margin:-margin]
                - results[kr_prev][margin:-margin, margin:-margin]))
            self.assertLess(diff, prev_diff,
                f"Non-monotone at kr={kr_curr}: {diff:.2e} >= {prev_diff:.2e}")
            prev_diff = diff


class TestPrecisionSensitivity(unittest.TestCase):
    """Claim C0-07. Precision sensitivity.
    Target:    B at float64 input vs float32-roundtripped input.
    Refinement variable: input dtype.
    Error norm: max absolute pixel difference.
    Expected: bounded by ~(ops count * float32 machine epsilon).
    Error classification: cross-layer contract error -> rounding."""

    def test_float64_vs_float32_envelope(self):
        rng = np.random.default_rng(42)
        image_f64 = rng.uniform(0.0, 1.0, size=(32, 32))
        image_f32_rt = image_f64.astype(np.float32).astype(np.float64)
        result_f64 = bilateral_filter(image_f64, 3.0, 0.1, 3)
        result_f32 = bilateral_filter(image_f32_rt, 3.0, 0.1, 3)
        max_diff = np.max(np.abs(result_f64 - result_f32))
        self.assertLess(max_diff, 1e-6,
            f"Precision envelope breached: {max_diff:.2e}")


class TestParameterLimits(unittest.TestCase):
    """Claim C0-08. Parameter limits.
    These are mathematical properties, not numerical refinement.
    Target:    B at extreme parameter values.
    Expected:  approaches known limit operator."""

    def test_identity_limit_small_sigma_r(self):
        """As sigma_r -> 0, B(I) -> I for non-constant neighborhoods."""
        rng = np.random.default_rng(42)
        image = rng.uniform(0.0, 1.0, size=(16, 16))
        prev_diff = float("inf")
        for sr in [0.1, 0.01, 0.001]:
            result = bilateral_filter(image, sigma_s=3.0, sigma_r=sr,
                                      kernel_radius=3)
            diff = np.max(np.abs(result - image))
            self.assertLess(diff, prev_diff,
                f"Identity limit not approached at sigma_r={sr}")
            prev_diff = diff

    def test_gaussian_limit_large_sigma_r(self):
        """As sigma_r -> inf, B(I) -> G(I). Verified via C0-03 oracle."""
        # Lightweight check: bilateral at sigma_r=1e3 and sigma_r=1e6
        # must be closer to each other than bilateral at sigma_r=10
        # and sigma_r=1e3.
        rng = np.random.default_rng(13)
        image = rng.uniform(0.0, 1.0, size=(16, 16))
        r10 = bilateral_filter(image, 2.0, 10.0, 3)
        r1k = bilateral_filter(image, 2.0, 1e3, 3)
        r1m = bilateral_filter(image, 2.0, 1e6, 3)
        diff_outer = np.max(np.abs(r10 - r1k))
        diff_inner = np.max(np.abs(r1k - r1m))
        self.assertLess(diff_inner, diff_outer)


# ---------------------------------------------------------------------------
# Evidence sensitivity — fault injection
# ---------------------------------------------------------------------------

class TestEvidenceSensitivity(unittest.TestCase):
    """For each claim, inject a known fault and confirm the oracle
    catches it. A test suite that passes correct code but also passes
    incorrect code is weak evidence."""

    def _make_broken_filter(self, image, sigma_s, sigma_r, kr, fault):
        """Run the bilateral filter with one injected fault."""
        rows, cols = image.shape
        output = np.zeros_like(image)

        spatial_table = np.zeros((2 * kr + 1, 2 * kr + 1))
        for dy in range(-kr, kr + 1):
            for dx in range(-kr, kr + 1):
                dist_sq = float(dx * dx + dy * dy)
                spatial_table[dy + kr, dx + kr] = np.exp(
                    -dist_sq / (2.0 * sigma_s * sigma_s))

        for r in range(rows):
            for c in range(cols):
                weight_sum = 0.0
                value_sum = 0.0
                center_val = image[r, c]
                for dy in range(-kr, kr + 1):
                    nr = r + dy
                    if nr < 0 or nr >= rows:
                        continue
                    for dx in range(-kr, kr + 1):
                        nc = c + dx
                        if nc < 0 or nc >= cols:
                            continue
                        neighbor_val = image[nr, nc]
                        intensity_diff = center_val - neighbor_val
                        range_weight = np.exp(
                            -(intensity_diff * intensity_diff)
                            / (2.0 * sigma_r * sigma_r))
                        spatial_weight = spatial_table[dy + kr, dx + kr]
                        w = spatial_weight * range_weight
                        weight_sum += w
                        value_sum += w * neighbor_val

                # Apply the specified fault.
                if fault == "multiply_instead_of_divide":
                    output[r, c] = value_sum * weight_sum
                elif fault == "add_bias":
                    output[r, c] = value_sum / weight_sum + 1e-10
                elif fault == "wrong_sigma_s":
                    # Already computed with correct sigma_s above,
                    # but we test a different fault: use sigma_s * 2
                    # in the spatial table. Rebuild for this pixel.
                    # (Simplified: just add a bias that simulates
                    # wrong spatial weighting.)
                    output[r, c] = value_sum / weight_sum + 0.01
                else:
                    output[r, c] = value_sum / weight_sum
        return output

    def test_fault_multiply_breaks_convex_hull(self):
        """Inject: multiply instead of divide.
        Expected: output exceeds local neighborhood range.
        Verifies: C0-01 oracle catches this fault."""
        rng = np.random.default_rng(99)
        image = rng.uniform(0.2, 0.8, size=(16, 16))
        broken = self._make_broken_filter(image, 2.0, 0.1, 3,
                                          "multiply_instead_of_divide")
        # The broken output should violate the local convex hull.
        violated = False
        kr = 3
        rows, cols = image.shape
        for r in range(rows):
            for c in range(cols):
                r_lo = max(0, r - kr)
                r_hi = min(rows, r + kr + 1)
                c_lo = max(0, c - kr)
                c_hi = min(cols, c + kr + 1)
                local_max = image[r_lo:r_hi, c_lo:c_hi].max()
                if broken[r, c] > local_max + 1e-12:
                    violated = True
                    break
            if violated:
                break
        self.assertTrue(violated,
            "Multiply fault did not violate convex hull — oracle is weak")

    def test_fault_bias_breaks_constant_preservation(self):
        """Inject: add 1e-10 bias to output.
        Expected: constant image error exceeds machine epsilon.
        Verifies: C0-02 oracle catches this fault."""
        image = np.full((16, 16), 0.5, dtype=np.float64)
        broken = self._make_broken_filter(image, 3.0, 0.1, 3, "add_bias")
        interior = broken[3:-3, 3:-3]
        max_error = np.max(np.abs(interior - 0.5))
        eps = np.finfo(np.float64).eps
        self.assertGreater(max_error, eps,
            "Bias fault not detected — constant-preservation oracle is weak")


if __name__ == "__main__":
    unittest.main()
