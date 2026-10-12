"""
Reproducible experiment runner for the bilateral filter.

Usage:

    python src/run.py
    python src/run.py config/tomasi.json

With no configuration argument, the runner uses:

    config/synthetic.json

Evidence is written to:

    evidence/runs/<run-name>/
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make src/ importable when this file runs from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bilateral import bilateral_filter, summarize


# --------------------------------------------------------------------------
# Identity and serialization
# --------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def array_identity(array: np.ndarray) -> str:
    """
    Return an identity digest for a NumPy array.

    The digest includes:

    1. shape
    2. dtype
    3. C-order value bytes
    """
    contiguous = np.ascontiguousarray(array)

    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(contiguous.dtype.str.encode("utf-8"))
    h.update(contiguous.tobytes(order="C"))

    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write deterministic, human-readable JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def source_hashes(src_dir: Path) -> dict:
    """Hash every Python source file in src/."""
    return {
        path.name: file_hash(path)
        for path in sorted(src_dir.glob("*.py"))
    }


def installed_packages() -> dict:
    """Capture package versions from the active Python environment."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "list",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    packages = json.loads(result.stdout)

    return {
        package["name"]: package["version"]
        for package in packages
    }


# --------------------------------------------------------------------------
# Synthetic input
# --------------------------------------------------------------------------

def generate_test_image(
    height: int,
    width: int,
    seed: int,
) -> np.ndarray:
    """
    Generate a deterministic synthetic grayscale image.

    The image contains:

    1. a smooth vertical gradient
    2. Gaussian noise
    3. a sharp vertical edge
    """
    if height < 1 or width < 1:
        raise ValueError("Image height and width must be positive.")

    rng = np.random.default_rng(seed)

    gradient = np.linspace(
        0.2,
        0.8,
        height,
        dtype=np.float64,
    ).reshape(-1, 1)

    gradient = np.broadcast_to(
        gradient,
        (height, width),
    ).copy()

    noise = rng.normal(
        0.0,
        0.05,
        size=(height, width),
    )

    edge = np.zeros(
        (height, width),
        dtype=np.float64,
    )
    edge[:, width // 2:] = 0.3

    image = gradient + noise + edge
    image = np.clip(image, 0.0, 1.0)

    if image.dtype != np.float64:
        raise TypeError(
            f"Synthetic input must be float64, got {image.dtype}."
        )

    return image


def load_synthetic_image(
    image_config: dict,
) -> tuple[np.ndarray, dict]:
    """Generate and describe the controlled synthetic fixture."""
    height = int(image_config["height"])
    width = int(image_config["width"])
    seed = int(image_config["seed"])

    image = generate_test_image(
        height=height,
        width=width,
        seed=seed,
    )

    record = {
        "kind": "synthetic",
        "generator": "generate_test_image",
        "generator_parameters": {
            "height": height,
            "width": width,
            "seed": seed,
        },
        "filter_input": {
            "shape_rows_cols": list(image.shape),
            "dtype": str(image.dtype),
            "minimum": float(np.min(image)),
            "maximum": float(np.max(image)),
            "array_identity_sha256": array_identity(image),
        },
    }

    return image, record


# --------------------------------------------------------------------------
# Acquired file input
# --------------------------------------------------------------------------

def load_file_image(
    repo_root: Path,
    input_config: dict,
) -> tuple[np.ndarray, dict]:
    """
    Verify, decode, and normalize an acquired grayscale image.

    Boundaries:

    raw file bytes
        -> decoded uint8 grayscale array
        -> normalized float64 filter input
    """
    relative_path = Path(input_config["path"])
    path = repo_root / relative_path

    if not path.is_file():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    # Boundary 1: exact downloaded file bytes.
    expected_raw_hash = input_config["raw_sha256"]
    actual_raw_hash = file_hash(path)

    if actual_raw_hash != expected_raw_hash:
        raise ValueError(
            "Raw file hash mismatch.\n"
            f"Path:     {path}\n"
            f"Expected: {expected_raw_hash}\n"
            f"Actual:   {actual_raw_hash}"
        )

    # Boundary 2: decoded grayscale values.
    decode_mode = input_config["decode_mode"]

    with Image.open(path) as source:
        source_format = source.format
        source_mode = source.mode
        source_size = list(source.size)
        frame_count = getattr(source, "n_frames", 1)

        decoded = np.array(
            source.convert(decode_mode),
            dtype=np.uint8,
            copy=True,
        )

    expected_shape = tuple(
        int(value)
        for value in input_config["expected_shape"]
    )

    if decoded.shape != expected_shape:
        raise ValueError(
            "Decoded shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {decoded.shape}"
        )

    expected_decoded_identity = (
        input_config["decoded_uint8_sha256"]
    )
    actual_decoded_identity = array_identity(decoded)

    if actual_decoded_identity != expected_decoded_identity:
        raise ValueError(
            "Decoded array identity mismatch.\n"
            f"Expected: {expected_decoded_identity}\n"
            f"Actual:   {actual_decoded_identity}"
        )

    # Boundary 3: model-ready floating-point representation.
    scale_divisor = float(input_config["scale_divisor"])

    if not np.isfinite(scale_divisor):
        raise ValueError("scale_divisor must be finite.")

    if scale_divisor <= 0:
        raise ValueError("scale_divisor must be positive.")

    image = decoded.astype(np.float64) / scale_divisor

    record = {
        "kind": "file",
        "path": str(relative_path),
        "source_url": input_config["source_url"],
        "raw": {
            "file_sha256": actual_raw_hash,
            "format": source_format,
            "native_mode": source_mode,
            "size_width_height": source_size,
            "frame_count": frame_count,
        },
        "decoded": {
            "requested_mode": decode_mode,
            "shape_rows_cols": list(decoded.shape),
            "dtype": str(decoded.dtype),
            "array_identity_sha256": actual_decoded_identity,
        },
        "normalization": {
            "operation": "divide",
            "divisor": scale_divisor,
        },
        "filter_input": {
            "shape_rows_cols": list(image.shape),
            "dtype": str(image.dtype),
            "minimum": float(np.min(image)),
            "maximum": float(np.max(image)),
            "array_identity_sha256": array_identity(image),
        },
    }

    return image, record


def load_input(
    repo_root: Path,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """
    Select the input construction from the configuration.

    The old synthetic schema remains supported:

        "image": {...}

    The acquired-file schema uses:

        "input": {"kind": "file", ...}
    """
    if "input" in config:
        input_config = config["input"]
        input_kind = input_config["kind"]

        if input_kind == "file":
            return load_file_image(
                repo_root,
                input_config,
            )

        if input_kind == "synthetic":
            return load_synthetic_image(input_config)

        raise ValueError(
            f"Unsupported input kind: {input_kind!r}"
        )

    if "image" in config:
        return load_synthetic_image(config["image"])

    raise ValueError(
        "Configuration must contain either 'input' or 'image'."
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible bilateral-filter experiment."
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="config/synthetic.json",
        help="Configuration path relative to the repository root.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    config_argument = Path(args.config)

    if config_argument.is_absolute():
        config_path = config_argument
    else:
        config_path = repo_root / config_argument

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    run_name = config.get(
        "run",
        {},
    ).get(
        "name",
        config_path.stem,
    )

    if (
        not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
    ):
        raise ValueError(
            "run.name must be one nonempty directory name."
        )

    evidence_dir = (
        repo_root
        / "evidence"
        / "runs"
        / run_name
    )
    evidence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filter_config = config["filter"]

    sigma_s = float(filter_config["sigma_s"])
    sigma_r = float(filter_config["sigma_r"])
    kernel_radius = int(filter_config["kernel_radius"])

    image, input_record = load_input(
        repo_root,
        config,
    )

    input_summary = summarize(image)

    # Time only the filter computation.
    start = time.perf_counter()

    filtered = bilateral_filter(
        image,
        sigma_s,
        sigma_r,
        kernel_radius,
    )

    elapsed = time.perf_counter() - start

    output_summary = summarize(filtered)

    input_identity = array_identity(image)
    output_identity = array_identity(filtered)

    # Preserve the exact arrays that entered and left the filter.
    input_npy_path = evidence_dir / "input.npy"
    output_npy_path = evidence_dir / "output.npy"

    np.save(input_npy_path, image)
    np.save(output_npy_path, filtered)

    summary = {
        "input": input_summary,
        "output": output_summary,
    }
    write_json(
        evidence_dir / "summary.json",
        summary,
    )

    timing = {
        "filter_seconds": elapsed,
    }
    write_json(
        evidence_dir / "timing.json",
        timing,
    )

    # PNG files are inspection artifacts, not exact numerical evidence.
    input_uint8 = (
        np.clip(image, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    output_uint8 = (
        np.clip(filtered, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    Image.fromarray(input_uint8).save(
        evidence_dir / "input.png"
    )
    Image.fromarray(output_uint8).save(
        evidence_dir / "output.png"
    )

    lineage = {
        "run_name": run_name,
        "config_path": str(config_argument),
        "config": config,
        "config_file_sha256": file_hash(config_path),
        "array_identity_definition": (
            "SHA256("
            "UTF8(str(shape)) || "
            "UTF8(dtype.str) || "
            "C-order value bytes"
            ")"
        ),
        "input": input_record,
        "output": {
            "shape_rows_cols": list(filtered.shape),
            "dtype": str(filtered.dtype),
            "array_identity_sha256": output_identity,
        },
        "artifacts": {
            "input_npy": {
                "file_sha256": file_hash(input_npy_path),
                "array_identity_sha256": input_identity,
            },
            "output_npy": {
                "file_sha256": file_hash(output_npy_path),
                "array_identity_sha256": output_identity,
            },
        },
        "filter": {
            "sigma_s": sigma_s,
            "sigma_r": sigma_r,
            "kernel_radius": kernel_radius,
        },
        "source_hashes": source_hashes(
            repo_root / "src"
        ),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": installed_packages(),
        },
    }

    write_json(
        evidence_dir / "lineage.json",
        lineage,
    )

    print(f"Run:     {run_name}")
    print(f"Input:   {input_summary}")
    print(f"Output:  {output_summary}")
    print(f"Time:    {elapsed:.4f} s")
    print(f"Input identity:  {input_identity}")
    print(f"Output identity: {output_identity}")
    print(f"Evidence written to {evidence_dir}/")


if __name__ == "__main__":
    main()
