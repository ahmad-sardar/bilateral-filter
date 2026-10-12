# Bilateral Filter — C0 Carrier Computation

Reference: Tomasi and Manduchi, "Bilateral Filtering for Gray and Color
Images," Proc. IEEE ICCV, 1998.

## What this is

A bilateral filter applied to a synthetic grayscale image, implemented
with no library calls hiding the computation. Every multiply, every exp,
every accumulate is in `src/bilateral.py`.

This repository is the ES0 artifact: a reproducible construction
environment for the C0-U0 carrier computation.

## Reproduce

```
pip install -r requirements.txt
make run
```

## Verify

```
make verify
```

This runs the experiment and prints the SHA-256 of the output array.
If the hash matches the recorded baseline, the result is reproduced.

## Evidence

After `make run`, the `evidence/` directory contains:

- `summary.json` — numerical summary of input and output
- `timing.json` — wall-clock duration
- `output.npy` — exact binary output for bitwise comparison
- `output.png` — human-readable output image
- `lineage.json` — what produced this result (source hashes,
  dependency versions, platform, configuration)
