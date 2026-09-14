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
inventories, and cryptographic checksums are provided in this repository.

Original benchmark datasets are governed by their respective licenses and
are not relicensed by the authors. Users should obtain the original datasets
from their official sources.

The raw model responses are [NOT PUBLICLY DISTRIBUTED / AVAILABLE IN RELEASE
v1.0.0]. Select the applicable statement before publication.

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