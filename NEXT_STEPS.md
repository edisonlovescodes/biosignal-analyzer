# Next steps

1. Run the corrected pipeline against local public MIT-BIH records; archive the model, data hashes, subject splits and full per-class evaluation.
2. Benchmark R-peak detection against annotations independently of beat classification.
3. Evaluate class imbalance using training-only methods and a documented baseline; do not infer quality from synthetic tests.
4. Review filtering, lead selection, sampling-rate support and HRV window requirements against the intended research use.
5. Verify container startup and an end-to-end deployed workflow before advertising hosting.

Current setup and evidence boundaries are in [README.md](README.md) and [CLAIMS.md](CLAIMS.md). No accuracy improvement or clinical efficacy is promised by these steps.
