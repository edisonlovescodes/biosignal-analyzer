# Claim ledger

| Claim | Status |
|---|---|
| ECG filtering, peak/HRV exploration, Streamlit and API | Implemented; local synthetic regression/UI checks |
| 187-sample beats and fixed coarse class order | Shared contract with regression tests |
| 95% classification accuracy / prior dataset counts / class balancing | Original evaluation evidence not tracked; excluded |
| Atrial-fibrillation diagnosis | Not this beat-classification task; fusion annotations corrected |
| Clinically reliable HR/HRV/quality assessment | Not established; these are exploratory signal estimates |
| Trained classifier | Requires a compatible new public-data bundle; not fabricated from random or synthetic weights |
| Real-data training/evaluation | Pipeline implemented; full MIT-BIH training not run during this repair |

Zero/insufficient-data HRV outputs are computational defaults, not evidence of physiologic normality. Short synthetic samples do not establish stable frequency-domain HRV.
