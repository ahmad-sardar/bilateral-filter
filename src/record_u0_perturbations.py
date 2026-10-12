"""Record the two C0-U0 experiments already performed on the Tomasi image.

Run from the bilateral-filter project root:
    python record_u0_perturbations.py

The script uses the existing src/bilateral.py. It creates a new evidence
directory on each run. Existing configuration, source, and baseline files
are read without being changed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import PIL
from PIL import Image


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(array)),
        "variance": float(np.var(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def compare(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = actual - reference
    return {
        "exactly_equal": bool(np.array_equal(actual, reference)),
        "changed_elements": int(np.count_nonzero(difference)),
        "total_elements": int(difference.size),
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
    }


def save_array(directory: Path, name: str, array: np.ndarray) -> dict[str, object]:
    path = directory / name
    np.save(path, array, allow_pickle=False)
    return {
        "file": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "npy_file_sha256": file_hash(path),
    }


def main() -> None:
    root = Path.cwd().resolve()
    config_path = root / "config/tomasi.json"
    if not config_path.is_file():
        raise RuntimeError("Run this script from the bilateral-filter project root.")

    # Resolve the same project module used in the earlier experiments.
    sys.path.insert(0, str(root))
    module = importlib.import_module("src.bilateral")
    source_path = Path(module.__file__).resolve()
    if source_path != (root / "src/bilateral.py").resolve():
        raise RuntimeError("The imported filter is not this project's src/bilateral.py.")
    bilateral_filter = module.bilateral_filter

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["input"]
    raw_path = root / source["path"]
    if file_hash(raw_path) != source["raw_sha256"]:
        raise RuntimeError("Raw-file identity mismatch.")

    with Image.open(raw_path) as opened:
        decoded = np.array(opened.convert(source["decode_mode"]))
    if list(decoded.shape) != list(source["expected_shape"]):
        raise RuntimeError("Decoded shape mismatch.")
    if decoded.dtype != np.uint8 or decoded.shape != (356, 325):
        raise RuntimeError("This recorder expects the 356-by-325 uint8 Tomasi image.")
    if source["scale_divisor"] != 255.0:
        raise RuntimeError("This recorder expects the normalized baseline scale.")

    parameters = {key: config["filter"][key] for key in ("sigma_s", "sigma_r", "kernel_radius")}
    if parameters != {"sigma_s": 3.0, "sigma_r": 0.1, "kernel_radius": 3}:
        raise RuntimeError("The filter parameters differ from those used in these experiments.")

    image = decoded.astype(np.float64) / source["scale_divisor"]
    image.setflags(write=False)
    baseline_path = root / "evidence/runs/tomasi-kitten-baseline/output.npy"
    baseline = np.load(baseline_path, allow_pickle=False)
    if baseline.shape != image.shape or baseline.dtype != np.float64:
        raise RuntimeError("Baseline shape or dtype mismatch.")

    # This ties the newly recorded experiments to the exact saved baseline.
    reproduced = bilateral_filter(image, **parameters)
    if not np.array_equal(reproduced, baseline):
        raise RuntimeError("The full-image rerun differs from the saved baseline. No archive was created.")

    crop = image[50:306, 50:275]
    crop_reference = baseline[50:306, 50:275]
    start = time.perf_counter()
    crop_output = bilateral_filter(crop, **parameters)
    crop_seconds = time.perf_counter() - start
    boundary = np.ones(crop.shape, dtype=bool)
    boundary[3:-3, 3:-3] = False
    crop_record = {
        "claim_type": "observed",
        "comparison": "Filter the crop versus crop the full-image filtered output.",
        "crop_bounds_stop_exclusive": {"rows": [50, 306], "columns": [50, 275]},
        "prediction": "Exact interior agreement; possible differences in the outer three rows or columns.",
        "input": summarize(crop),
        "output": summarize(crop_output),
        "filter_seconds_single_measurement": crop_seconds,
        "interior": compare(crop_output[~boundary], crop_reference[~boundary]),
        "boundary": compare(crop_output[boundary], crop_reference[boundary]),
    }

    image32 = image.astype(np.float32)
    image32.setflags(write=False)
    # Record the actual exception rather than assuming its class.
    try:
        direct_output = bilateral_filter(image32, **parameters)
    except (AssertionError, TypeError, ValueError) as error:
        direct_record = {
            "result": "rejected",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
    else:
        direct_record = {"result": "accepted", "output_dtype": str(direct_output.dtype)}

    rounded = image32.astype(np.float64)
    rounded.setflags(write=False)
    start = time.perf_counter()
    rounded_output = bilateral_filter(rounded, **parameters)
    rounded_seconds = time.perf_counter() - start
    numeric_record = {
        "claim_type": "observed",
        "experiment": "float32-rounded input values passed to the existing float64 filter.",
        "direct_float32_prediction": "Rejection under the existing float64 input contract.",
        "direct_float32_observation": direct_record,
        "float64_input_data_bytes": int(image.nbytes),
        "float32_input_data_bytes": int(image32.nbytes),
        "input_rounding": compare(rounded, image),
        "filter_input": summarize(rounded),
        "output": summarize(rounded_output),
        "output_comparison": compare(rounded_output, baseline),
        "filter_seconds_single_measurement": rounded_seconds,
        "interpretation": "Input precision was reduced; this does not measure a filter doing all arithmetic in float32.",
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / "evidence/runs" / f"u0-perturbations-{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    crop_directory = directory / "crop-50"
    numeric_directory = directory / "float32-rounded-input"
    crop_directory.mkdir()
    numeric_directory.mkdir()

    shutil.copyfile(Path(__file__).resolve(), directory / "record_u0_perturbations.py")
    shutil.copyfile(source_path, directory / "bilateral.py")
    shutil.copyfile(config_path, directory / "baseline-config.json")
    shared_arrays = {
        "baseline_input": save_array(directory, "baseline-input.npy", image),
        "baseline_output": save_array(directory, "baseline-output.npy", baseline),
    }
    crop_record["arrays"] = {
        "input": save_array(crop_directory, "input.npy", crop),
        "output": save_array(crop_directory, "output.npy", crop_output),
        "reference": save_array(crop_directory, "reference.npy", crop_reference),
        "difference": save_array(crop_directory, "difference.npy", crop_output - crop_reference),
        "boundary_mask": save_array(crop_directory, "boundary-mask.npy", boundary),
    }
    numeric_record["reference"] = "../baseline-output.npy"
    numeric_record["arrays"] = {
        "float32_input": save_array(numeric_directory, "input-float32.npy", image32),
        "filter_input": save_array(numeric_directory, "input.npy", rounded),
        "output": save_array(numeric_directory, "output.npy", rounded_output),
        "difference": save_array(numeric_directory, "difference.npy", rounded_output - baseline),
    }
    write_json(crop_directory / "comparison.json", crop_record)
    write_json(numeric_directory / "comparison.json", numeric_record)
    environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    write_json(directory / "environment.json", environment)
    index = {
        "status": "complete",
        "recorded_at_utc": stamp,
        "baseline_reproduced_exactly": True,
        "raw_input": {"path": source["path"], "sha256": file_hash(raw_path)},
        "saved_baseline_path": str(baseline_path.relative_to(root)),
        "saved_baseline_npy_sha256": file_hash(baseline_path),
        "shared_arrays": shared_arrays,
        "experiments": ["crop-50/comparison.json", "float32-rounded-input/comparison.json"],
        "snapshots": {
            name: file_hash(directory / name)
            for name in ("record_u0_perturbations.py", "bilateral.py", "baseline-config.json", "environment.json")
        },
        "scope": "Records the crop and numeric-type experiments. Full C0-U0 closure also requires its audit, thread map, and engineering records.",
    }
    write_json(directory / "index.json", index)
    print("baseline_reproduced:", True)
    print("crop_interior:", crop_record["interior"])
    print("crop_boundary:", crop_record["boundary"])
    print("direct_float32:", direct_record)
    print("rounded_input_output_comparison:", numeric_record["output_comparison"])
    print("Evidence written to", directory)


if __name__ == "__main__":
    main()
