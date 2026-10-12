from hashlib import sha256
from pathlib import Path
import platform
import sys

import numpy as np
import PIL
from PIL import Image


IMAGE_PATH = Path(
    "data/raw/tomasi-manduchi/kitten-orig.gif"
)


def array_sha256(array: np.ndarray) -> str:
    """Return a SHA-256 digest of an array's shape, dtype, and data.

       The array is first normalized to a contiguous C-order (row-major) copy,
       so the digest depends only on the logical values, shape, and dtype —
       never on memory layout, strides, or base object. Two logically equal
       arrays (a view and its copy, a transposed-and-transposed-back slice,
       arrays from different platforms) all produce the same digest.

       Digest inputs, in order:
         1. str(array.shape)   — e.g. "(240, 320)"
         2. array.dtype.str    — e.g. "<f8" (endian tag + type code)
         3. array.tobytes(order="C") — the raw data bytes

       Notes:
         - A no-op on already C-contiguous float64 images; allocates only for
           strided or non-contiguous views.
         - The shape/dtype prefixes guard against a collision where different
           shapes happen to produce identical byte sequences.
         - Deterministic: independent of process, BLAS threads, and numpy
           version, given the same logical array contents.

       Args:
           array: Any NumPy array (2D grayscale in this project's use).

       Returns:
           Hex-encoded SHA-256 digest (64 lowercase hex chars).

       """

    contiguous = np.ascontiguousarray(array)


    digest = sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))

    return digest.hexdigest()


raw_bytes = IMAGE_PATH.read_bytes()

with Image.open(IMAGE_PATH) as image:
    native_array = np.asarray(image.copy())
    grayscale_array = np.asarray(image.convert("L"))

    palette = image.getpalette()

    print("SYSTEM")
    print("os:", platform.system())
    print("os_release:", platform.release())
    print("architecture:", platform.machine())
    print("python:", sys.version.split()[0])
    print("pillow:", PIL.__version__)
    print("numpy:", np.__version__)

    print("\nRAW FILE")
    print("path:", IMAGE_PATH)
    print("byte_count:", len(raw_bytes))
    print("sha256:", sha256(raw_bytes).hexdigest())

    print("\nGIF CONTAINER")
    print("format:", image.format)
    print("size:", image.size)
    print("native_mode:", image.mode)
    print("frame_count:", getattr(image, "n_frames", 1))
    print(
        "palette_entries:",
        len(palette) // 3 if palette is not None else 0,
    )
    print("transparency:", image.info.get("transparency"))
    print("background:", image.info.get("background"))

    print("\nNATIVE DECODED ARRAY")
    print("shape:", native_array.shape)
    print("dtype:", native_array.dtype)
    print("minimum:", native_array.min())
    print("maximum:", native_array.max())
    print("identity_sha256:", array_sha256(native_array))

    print("\nGRAYSCALE DECODED ARRAY")
    print("shape:", grayscale_array.shape)
    print("dtype:", grayscale_array.dtype)
    print("minimum:", grayscale_array.min())
    print("maximum:", grayscale_array.max())
    print("identity_sha256:", array_sha256(grayscale_array))
    python3 - <<'PY'
    from hashlib import sha256
    from pathlib import Path
    import platform
    import sys

    import numpy as np
    import PIL
    from PIL import Image


    IMAGE_PATH = Path(
        "data/raw/tomasi-manduchi/kitten-orig.gif"
    )


    def array_sha256(array: np.ndarray) -> str:
        """Hash array metadata and its contiguous C-order bytes."""
        contiguous = np.ascontiguousarray(array)

        digest = sha256()
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("utf-8"))
        digest.update(contiguous.tobytes(order="C"))

        return digest.hexdigest()


    raw_bytes = IMAGE_PATH.read_bytes()

    with Image.open(IMAGE_PATH) as image:
        native_array = np.asarray(image.copy())
        grayscale_array = np.asarray(image.convert("L"))

        palette = image.getpalette()

        print("SYSTEM")
        print("os:", platform.system())
        print("os_release:", platform.release())
        print("architecture:", platform.machine())
        print("python:", sys.version.split()[0])
        print("pillow:", PIL.__version__)
        print("numpy:", np.__version__)

        print("\nRAW FILE")
        print("path:", IMAGE_PATH)
        print("byte_count:", len(raw_bytes))
        print("sha256:", sha256(raw_bytes).hexdigest())

        print("\nGIF CONTAINER")
        print("format:", image.format)
        print("size:", image.size)
        print("native_mode:", image.mode)
        print("frame_count:", getattr(image, "n_frames", 1))
        print(
            "palette_entries:",
            len(palette) // 3 if palette is not None else 0,
        )
        print("transparency:", image.info.get("transparency"))
        print("background:", image.info.get("background"))

        print("\nNATIVE DECODED ARRAY")
        print("shape:", native_array.shape)
        print("dtype:", native_array.dtype)
        print("minimum:", native_array.min())
        print("maximum:", native_array.max())
        print("identity_sha256:", array_sha256(native_array))

        print("\nGRAYSCALE DECODED ARRAY")
        print("shape:", grayscale_array.shape)
        print("dtype:", grayscale_array.dtype)
        print("minimum:", grayscale_array.min())
        print("maximum:", grayscale_array.max())
        print("identity_sha256:", array_sha256(grayscale_array))
