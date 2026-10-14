# Contributor guidance

Read README.md, CLAIMS.md and CHANGELOG.md before changing code or describing this project. Run the documented regression suite and build after relevant edits. Keep generated dependencies, caches, local runs and secrets out of Git.

Separate implemented behavior, test fixtures, measured evaluation and historical claims. Synthetic tests establish software contracts, not clinical performance or biological function. Never add accuracy, impact or deployment claims without a reproducible artifact and protocol. Preserve historical checkpoints/evidence; create versioned outputs for new experiments.

Changes to tokenization, labels, normalization, sampling rate, model objective or artifact format must update training, inference, documentation and regression tests together. Do not silently serve incompatible, random or synthetic checkpoints as validated models.
