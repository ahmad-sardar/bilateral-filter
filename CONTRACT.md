# Carrier Contract — Bilateral Filter

Reference: Tomasi and Manduchi, "Bilateral Filtering for Gray and
Color Images," Proc. IEEE ICCV, 1998.

## Operator definition

For pixel p in a 2D grayscale image I:

    B(I)_p = (1 / W(p)) * sum_{q in N(p)} s(p,q) * r(I(p), I(q)) * I(q)

where:

    s(p,q) = exp( -||p - q||^2 / (2 * sigma_s^2) )    spatial kernel
    r(a,b) = exp( -(a - b)^2 / (2 * sigma_r^2) )       range kernel
    W(p)   = sum_{q in N(p)} s(p,q) * r(I(p), I(q))    normalization
    N(p)   = { q : max(|p_x - q_x|, |p_y - q_y|) <= R } ∩ image domain

## Domain and codomain

- Input:  2D ndarray, dtype float64, shape (H, W), H >= 1, W >= 1.
- Output: 2D ndarray, dtype float64, same shape as input.

## Permitted input range

[0.0, 1.0]. Values outside are accepted but not guaranteed meaningful.

## Parameter constraints

- sigma_s > 0
- sigma_r > 0
- R >= 1 (integer, kernel radius)

## Border policy

Skip. Neighbors outside the image boundary are excluded from N(p).
No padding, no reflection, no extension. W(p) is computed over
the available neighbors only.

Consequence: boundary pixels have smaller neighborhoods. The operator
is still well-defined (W(p) >= 1 for all p, because the center pixel
is always in its own neighborhood).

## Normalization

Explicit division by W(p). The output at each pixel is a convex
combination of neighbor values (weights non-negative, sum to 1
after normalization).

## NaN and infinity

Unspecified. NaN or infinity in the input produces undefined output.
No explicit guard.

## Overflow

Not expected in float64 for the permitted input range and Gaussian
kernels. Exponents are non-positive (spatial and intensity distances
are non-negative). Products of exp values are in (0, 1]. No explicit
overflow guard.

At extreme parameters (sigma_r very small), range weights can
underflow to 0.0 in float64 for neighbors with large intensity
difference. This is numerically correct behavior, not an error:
the neighbor's contribution is negligible.

## Determinism

Same input array (identical dtype, shape, values), same parameters,
same platform, same Python version, same numpy version: identical
output, bit for bit.
