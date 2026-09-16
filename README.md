# SALRM Reproducibility Package

This repository contains the code, derived analysis data, statistical
outputs, figures, and reproducibility records supporting the manuscript:

**Engineering Reliability Assessment of Large Language Models:
A Sensitivity-Aware Latent Framework with Uncertainty Quantification**

## Overview

This study proposes a sensitivity-aware latent reliability assessment
framework centered on the Sensitivity-Aware Latent Reliability Model
(SALRM). SALRM combines item-difficulty-adjusted latent reliability,
model-specific intervention effects, sensitivity adjustment, statistical
uncertainty, and benchmark-specific behavioral diagnostics.

## Experimental Design

The confirmatory experiment evaluated four large language models:

- GPT-4.1-mini
- Qwen2.5-7B-Instruct
- Claude Haiku 4.5
- Llama 3.1-8B-Instruct

Five benchmarks were used:

- BBQ
- StereoSet
- AdvGLUE
- BOLD
- RealToxicityPrompts

The experiment included:

- 7,738 unique primary items
- 4 models
- 2 primary conditions: baseline and safety
- 61,904 primary responses
- 96,000 stochastic responses
- 157,904 total responses
- 32 complete raw response files

## SALRM

Primary favorable outcomes were analyzed using a Rasch-type random-item
logistic model. Let R_m0 and R_m1 denote the baseline and safety-condition
reliabilities for model m.

The signed condition effect is:

D_m = R_m1 - R_m0

Absolute sensitivity is:

S_m = |D_m|

The primary sensitivity-adjusted reliability score is:

A_m(lambda) = R_m0 exp(-lambda S_m)

The primary analysis used lambda = 1.

## Repository Contents

- `src/`: final SALRM analysis pipeline
- `config/`: public configuration file
- `environment/`: software and package version records
- `model_records/`: Ollama model information and model digests
- `data/`: derived analysis data and data documentation
- `results/`: primary statistical results
- `validation/`: robustness and complete-refit validation results
- `figures/`: manuscript and supplementary figures
- `integrity/`: SHA-256 checksums and response-file inventory
- `supplement/`: supplementary reproducibility material

## Execution Environment

The reported analysis used:

- Windows 10
- PowerShell
- Python 3.11.15
- Conda environment: `salrm`
- Ollama 0.33.0
- SALRM pipeline: `v8-final`

Complete package versions are provided in:

- `environment/environment.yml`
- `environment/pip_freeze.txt`
- `environment/core_package_versions.txt`

## Data Availability

Derived analysis data, statistical outputs, sampling records, response-file
inventories, and cryptographic checksums are provided in the `data/`
directory of this repository.

The processed item-level dataset, including the model outputs retained for
scoring and analysis, is distributed as a compressed archive at
`data/scores.csv.zip`. The SHA-256 checksums of both the compressed archive
and the extracted CSV file are provided in `data/scores_checksums.csv` to
support integrity verification.

Aggregate evaluation results are provided in `data/metrics.csv`. Sampling
information and response-file completeness records are provided in
`data/sampling_manifest.json` and `data/response_inventory.csv`,
respectively.

The original provider-specific raw response files are not publicly
distributed. Their completeness and integrity are documented through the
response inventory and associated cryptographic checksum records included
in this repository.

Original benchmark datasets and benchmark-derived content remain subject to
the licenses and terms of their respective owners and are not relicensed by
the author. Users should obtain the original benchmark datasets from their
official sources. The availability and reuse of model outputs are also
subject to the applicable terms of the respective model providers.

## Code Availability

The final analysis pipeline is provided in `src/salrm_pipeline.py`.
The public configuration file does not contain API keys or local credentials.

## Integrity Verification

The `integrity/` directory contains:

- pipeline SHA-256
- artifact checksums
- reproducibility-record checksums
- response inventory
- local-model digests

These records identify the files used to produce the reported results.

## Reproducibility Scope

The archived materials support reproduction of the reported data processing
and statistical analyses. Byte-identical regeneration of hosted-model
responses is not guaranteed because provider-side models, moderation systems,
and API endpoints may change over time.

## Citation

Citation information is provided in `CITATION.cff`.

## License

The source code in this repository, including `salrm_pipeline.py` and
author-created utility scripts, is licensed under the MIT License.

The MIT License does not automatically apply to benchmark materials, prompts,
model outputs, or other third-party content. The original benchmark datasets
remain subject to their respective licenses and terms of use. Model outputs
may also be subject to the applicable model-provider terms.

Author-generated summary tables, figures, manifests, and metadata may be reused
for scholarly purposes with appropriate citation, unless otherwise indicated.

### Score-level data

The item-level score data are distributed as a compressed CSV file:

- `data/scores.csv.zip`

After downloading the repository, extract the archive before running analyses.

#### PowerShell
```powershell
Expand-Archive `
    -Path "data/scores.csv.zip" `
    -DestinationPath "data" `
    -Force