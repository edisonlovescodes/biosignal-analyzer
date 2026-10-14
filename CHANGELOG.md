# Changelog

## 2026-10-13

- Fixed notch units, complete odd-length segments, flat/short peak behavior and frequency-coordinate HRV integration.
- Corrected fusion annotations and used a fixed persisted class order, JSON metadata and artifact checksums.
- Added one local-data training entry point with subject-grouped splits, including the 201/202 shared-subject relationship.
- Shared preprocessing/segmentation across training, UI and API; removed default random-weight serving.
- Added CSV validation/limits and explicit classification availability; removed unimplemented upload formats.
- Retired duplicate/implicit-download and fake clinical synthetic-training shortcuts.
- Removed unsupported efficacy/training assertions and consolidated setup instructions.
- Added scientific/runtime regression tests and CI; repaired container healthcheck.

## 2026-10-13 — Distinct interface redesign

White/cobalt waveform-first lab, self-hosted IBM Plex Sans, deterministic synthetic preview, native controls and analysis tabs, filter-change invalidation, responsive layout and neutral input-amplitude units.

Existing research/evaluation limitations remain in CLAIMS.md. Font licenses are included alongside the assets.
