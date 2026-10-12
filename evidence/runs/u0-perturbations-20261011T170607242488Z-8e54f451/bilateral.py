"""
Bilateral filter implementation.

Reference: Tomasi and Manduchi, "Bilateral Filtering for Gray and
Color Images," Proc. IEEE ICCV, 1998.

Reference: Paris, Kornprobst, Tumblin, Durand, "Bilateral Filtering:
Theory and Applications," Foundations and Trends in Computer Graphics
and Vision, 4(1), 2009.

Structure:
    build_spatial_table   — pure, parameters → weight table
    compute_range_weight  — pure, two values + sigma → one weight
    valid_neighbors       — pure, position + bounds → valid (nr, nc, dy, dx)
    filter_one_pixel      — pure, image + position + params → one value
    filter_one_pixel_traced — same, with a log of every intermediate
    bilateral_filter      — traversal, image + params → new image

Mutation is in one place: filling the output array in bilateral_filter.
Every function below it is pure.
"""

import numpy as np


# --------------------------------------------------------------------------
# Pure components
# --------------------------------------------------------------------------

def build_spatial_table(sigma_s: float, kernel_radius: int) -> np.ndarray:
    """Pure. Parameters → one 2D weight table.

    The spatial weight depends only on the offset (dy, dx), not on the
    image. It is computed once and reused for every pixel.
    """
    size = 2 * kernel_radius + 1
    table = np.zeros((size, size), dtype=np.float64)
    for dy in range(-kernel_radius, kernel_radius + 1):
        for dx in range(-kernel_radius, kernel_radius + 1):
            dist_sq = float(dy * dy + dx * dx)
            table[dy + kernel_radius, dx + kernel_radius] = np.exp(
                -dist_sq / (2.0 * sigma_s * sigma_s)
            )
    return table


def compute_range_weight(center_val: float, neighbor_val: float,
                         sigma_r: float) -> float:
    """Pure. Two intensity values + sigma → one weight.

    The range weight depends on the intensity difference between the
    center pixel and one neighbor. A small difference gives a weight
    near 1. A large difference gives a weight near 0.
    """
    diff = center_val - neighbor_val
    return float(np.exp(-(diff * diff) / (2.0 * sigma_r * sigma_r)))


def valid_neighbors(r: int, c: int, kernel_radius: int,
                    rows: int, cols: int):
    """Pure generator. Position + bounds → valid (nr, nc, dy, dx).

    Boundary check happens here, nowhere else.
    Every consumer receives only valid positions.
    """
    for dy in range(-kernel_radius, kernel_radius + 1):
        nr = r + dy
        if nr < 0 or nr >= rows:
            continue
        for dx in range(-kernel_radius, kernel_radius + 1):
            nc = c + dx
            if nc < 0 or nc >= cols:
                continue
            yield (nr, nc, dy, dx)


def filter_one_pixel(image: np.ndarray, r: int, c: int,
                     sigma_r: float, kernel_radius: int,
                     spatial_table: np.ndarray) -> float:
    """Pure. Image + position + parameters → one output value.

    Reads image but does not modify it.
    No boundary logic here — valid_neighbors handles it.
    """
    rows, cols = image.shape
    center_val = image[r, c]

    # Accumulate: (weight_sum, value_sum), one neighbor at a time
    weight_sum = 0.0
    value_sum = 0.0

    for nr, nc, dy, dx in valid_neighbors(r, c, kernel_radius, rows, cols):
        neighbor_val = image[nr, nc]
        rw = compute_range_weight(center_val, neighbor_val, sigma_r)
        sw = spatial_table[dy + kernel_radius, dx + kernel_radius]
        w = sw * rw
        weight_sum += w
        value_sum += w * neighbor_val

    # Normalize the accumulated sums
    return value_sum / weight_sum


# --------------------------------------------------------------------------
# Traced version — logs every intermediate at one pixel
# --------------------------------------------------------------------------

def filter_one_pixel_traced(image: np.ndarray, r: int, c: int,
                            sigma_r: float, kernel_radius: int,
                            spatial_table: np.ndarray) -> tuple:
    """Same computation as filter_one_pixel, with a full trace log.

    Returns (output_value, trace_log).
    trace_log is a list of dicts, one per neighbor, plus a final entry.
    """
    rows, cols = image.shape
    center_val = image[r, c]
    log = []

    weight_sum = 0.0
    value_sum = 0.0

    for nr, nc, dy, dx in valid_neighbors(r, c, kernel_radius, rows, cols):
        neighbor_val = image[nr, nc]
        rw = compute_range_weight(center_val, neighbor_val, sigma_r)
        sw = spatial_table[dy + kernel_radius, dx + kernel_radius]
        w = sw * rw

        weight_sum += w
        value_sum += w * neighbor_val

        log.append({
            "neighbor": (nr, nc),
            "offset": (dy, dx),
            "neighbor_val": neighbor_val,
            "neighbor_dtype": type(neighbor_val).__name__,
            "intensity_diff": center_val - neighbor_val,
            "spatial_weight": sw,
            "range_weight": rw,
            "combined_weight": w,
            "running_weight_sum": weight_sum,
            "running_weight_sum_dtype": type(weight_sum).__name__,
            "running_value_sum": value_sum,
        })

    output = value_sum / weight_sum

    log.append({
        "final_value_sum": value_sum,
        "final_weight_sum": weight_sum,
        "output": output,
        "output_dtype": type(output).__name__,
        "center_val": center_val,
        "center_dtype": type(center_val).__name__,
    })

    return output, log


# --------------------------------------------------------------------------
# Top-level: the only function that mutates
# --------------------------------------------------------------------------

def bilateral_filter(image: np.ndarray, sigma_s: float,
                     sigma_r: float, kernel_radius: int) -> np.ndarray:
    """Image + parameters → new image.

    The only mutation: filling the output array.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D.")
    if sigma_s <= 0 or sigma_r <= 0:
        raise ValueError("sigma_s and sigma_r must be positive.")
    if kernel_radius < 1:
        raise ValueError("kernel_radius must be at least 1.")

    assert image.dtype == np.float64, f"Expected float64, got {image.dtype}"

    rows, cols = image.shape
    spatial_table = build_spatial_table(sigma_s, kernel_radius)
    output = np.zeros_like(image)

    for r in range(rows):
        for c in range(cols):
            output[r, c] = filter_one_pixel(
                image, r, c, sigma_r, kernel_radius, spatial_table
            )

    assert output.dtype == np.float64, f"Output dtype drifted to {output.dtype}"

    return output


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def summarize(image: np.ndarray) -> dict:
    """Compute summary statistics of an image."""
    return {
        "mean": float(np.mean(image)),
        "variance": float(np.var(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }
