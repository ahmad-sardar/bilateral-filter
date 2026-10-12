"""A readable bilateral filter for a 2D grayscale float64 image.

Tomasi and Manduchi, "Bilateral Filtering for Gray and Color Images" (1998).
Paris et al., "Bilateral Filtering: Theory and Applications" (2009).

The input image is read only. Local arrays and accumulators may be updated
while they are built, and the caller receives a new output array.
"""

from collections.abc import Iterator

import numpy as np


def build_spatial_table(sigma_s: float, kernel_radius: int) -> np.ndarray:
    """Build weights for every offset from -radius to +radius on each axis.

    The table depends on offsets, not on image values, so one table can be
    reused for every pixel. An offset of zero is stored at index radius.
    """
    size = 2 * kernel_radius + 1
    table = np.empty((size, size), dtype=np.float64)

    for dy in range(-kernel_radius, kernel_radius + 1):
        for dx in range(-kernel_radius, kernel_radius + 1):
            scaled_dy = dy / sigma_s
            scaled_dx = dx / sigma_s
            table[dy + kernel_radius, dx + kernel_radius] = np.exp(
                -0.5 * (scaled_dy * scaled_dy + scaled_dx * scaled_dx)
            )

    # A caller can use this table for lookup but cannot change its weights.
    table.flags.writeable = False
    return table


def compute_range_weight(
    center_val: float, neighbor_val: float, sigma_r: float
) -> float:
    """Return the weight determined by the two pixel intensities."""
    scaled_difference = (center_val - neighbor_val) / sigma_r
    return float(np.exp(-0.5 * scaled_difference * scaled_difference))


def valid_neighbors(
    r: int, c: int, kernel_radius: int, rows: int, cols: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield valid (neighbor_row, neighbor_col, dy, dx) tuples.

    dy and dx remain signed offsets. For example, a neighbor above the
    center has dy = -1, even though its table index is nonnegative.
    """
    for dy in range(-kernel_radius, kernel_radius + 1):
        neighbor_row = r + dy
        if not (0 <= neighbor_row < rows):
            continue

        for dx in range(-kernel_radius, kernel_radius + 1):
            neighbor_col = c + dx
            if 0 <= neighbor_col < cols:
                yield neighbor_row, neighbor_col, dy, dx


def _filter_pixel(
    image: np.ndarray,
    r: int,
    c: int,
    sigma_r: float,
    kernel_radius: int,
    spatial_table: np.ndarray,
    *,
    trace: bool,
) -> tuple[float, list[dict[str, object]]]:
    """Compute one pixel, optionally recording the same computation.

    Local sums and the optional log are built here. The image and the
    spatial table are only read; no caller-owned object is modified.
    """
    rows, cols = image.shape
    center_val = float(image[r, c])
    weight_sum = 0.0
    value_sum = 0.0
    log: list[dict[str, object]] = []

    for nr, nc, dy, dx in valid_neighbors(r, c, kernel_radius, rows, cols):
        neighbor_val = float(image[nr, nc])
        spatial_weight = float(spatial_table[dy + kernel_radius, dx + kernel_radius])
        range_weight = compute_range_weight(center_val, neighbor_val, sigma_r)
        combined_weight = spatial_weight * range_weight

        weight_sum += combined_weight
        value_sum += combined_weight * neighbor_val

        if trace:
            log.append({
                "neighbor": (nr, nc),
                "offset": (dy, dx),
                "neighbor_val": neighbor_val,
                "neighbor_dtype": image.dtype.name,
                "intensity_diff": center_val - neighbor_val,
                "spatial_weight": spatial_weight,
                "range_weight": range_weight,
                "combined_weight": combined_weight,
                "running_weight_sum": weight_sum,
                "running_weight_sum_dtype": type(weight_sum).__name__,
                "running_value_sum": value_sum,
            })

    # The center pixel is always a valid neighbor with weight 1.
    output = value_sum / weight_sum

    if trace:
        log.append({
            "final_value_sum": value_sum,
            "final_weight_sum": weight_sum,
            "output": output,
            "output_dtype": type(output).__name__,
            "center_val": center_val,
            "center_dtype": image.dtype.name,
        })

    return output, log


def filter_one_pixel(
    image: np.ndarray,
    r: int,
    c: int,
    sigma_r: float,
    kernel_radius: int,
    spatial_table: np.ndarray,
) -> float:
    """Return one filtered value without modifying the input image."""
    output, _ = _filter_pixel(
        image, r, c, sigma_r, kernel_radius, spatial_table, trace=False
    )
    return output


def filter_one_pixel_traced(
    image: np.ndarray,
    r: int,
    c: int,
    sigma_r: float,
    kernel_radius: int,
    spatial_table: np.ndarray,
) -> tuple[float, list[dict[str, object]]]:
    """Return one filtered value and a log of its intermediate values."""
    return _filter_pixel(
        image, r, c, sigma_r, kernel_radius, spatial_table, trace=True
    )


def bilateral_filter(
    image: np.ndarray, sigma_s: float, sigma_r: float, kernel_radius: int
) -> np.ndarray:
    """Return a new 2D grayscale image after bilateral filtering.

    The radius defines a square window of side length 2 * radius + 1.
    Near image boundaries, invalid neighbors are omitted and the remaining
    weights are normalized by their own sum.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("image must be a 2D NumPy array")
    if image.dtype != np.float64:
        raise ValueError(f"image must have dtype float64, got {image.dtype}")
    if image.size == 0:
        raise ValueError("image must have at least one pixel")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values")
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError("sigma_s must be finite and positive")
    if not np.isfinite(sigma_r) or sigma_r <= 0:
        raise ValueError("sigma_r must be finite and positive")
    if isinstance(kernel_radius, (bool, np.bool_)) or not isinstance(
        kernel_radius, (int, np.integer)
    ) or kernel_radius < 1:
        raise ValueError("kernel_radius must be a positive integer")

    rows, cols = image.shape
    spatial_table = build_spatial_table(sigma_s, kernel_radius)
    output = np.empty_like(image)

    # This function owns the output array. It fills every position once.
    for r in range(rows):
        for c in range(cols):
            output[r, c] = filter_one_pixel(
                image, r, c, sigma_r, kernel_radius, spatial_table
            )

    return output


def summarize(image: np.ndarray) -> dict[str, object]:
    """Return summary statistics without modifying the image."""
    return {
        "mean": float(np.mean(image)),
        "variance": float(np.var(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }
