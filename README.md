# ICEMAP-code

This is a code repository for radar-based ice sensing. It has currently been validated for radar frequencies of 7.29–60 GHz, ice densities of 0.80–0.917 g/cm³, ice thicknesses of 5–105 cm, and sensing distances of 0.5–2.5 m.

**Code Organization.** The code is grouped into data preparation, signal processing, and model training and evaluation. These parts correspond to `p0_prehandle/`, `p1_design1/`–`p3_design3/`, and `p4_design4/`, respectively, with shared utilities in `common/`. To support broad testing across measurement conditions, multiple algorithm variants and parameter configurations are provided.

**Runtime Environment.** MATLAB R2024a or later with Signal Processing Toolbox is recommended for data processing. The supplied Python environment specifies Python 3.10.x, PyTorch 2.9.0, NumPy 2.1.2, and h5py 3.15.1. SciPy must be >=1.13 and <1.16. The environment and dependency files provide the complete package configuration. GPU acceleration is optional and requires a compatible CUDA runtime.

**Data Preparation.** MATLAB (`.mat`) and HDF5 (`.h5`) files are used at different stages. Set local data paths and measurement-specific configurations before running the code. 

We hope this code will facilitate research on radar-based ice sensing and, more broadly, on the analysis of layered dielectric media.

**Update Log**

Dates provide an approximate overview of development, including earlier code revisions.

| Year   | Month | Update                                   |
| ------ | ----- | ---------------------------------------- |
| [2025] | [Nov] | Initial code framework and data loading  |
| [2025] | [Dec] | Data preprocessing                       |
| [2026] | [Jan] | Algorithm: denoise / alignment / network |
| [2026] | [Feb] | Dataset format organization              |
| [2026] | [Mar] | Model train and evaluation               |
| [2026] | [Apr] | Algorithm variants test                  |
| [2026] | [May] | Result inspection                        |
| [2026] | [Jun] | Experiment organization                  |
