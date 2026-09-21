# FedSift

**Differentially private federated learning for diabetes and diabetic retinopathy prediction.**

FedSift is a research implementation and reproducibility package for federated binary classification on two tabular medical datasets. It combines record-level differentially private local training, FedAdam server updates, a two-phase noise schedule, and a public-control rule that accepts a partial server step when its improvement over the full step meets a specified margin.

The package includes the implementation, fixed experiment plans, input data, an offline Python runtime, trained models, and reference results. It supports checking the recorded experiment, rebuilding result tables, and rerunning training through a single command-line interface.

[Quick start](#quick-start) · [Experiments](#experiments) · [Reproduction](#reproduction) · [Results](#results) · [Repository structure](#repository-structure) · [Troubleshooting](#troubleshooting)

## Quick start

### Prerequisites

- **Linux x86-64**, or **Windows with an installed WSL 2 distribution**. The PowerShell launcher uses `Ubuntu` by default.
- Bash, GNU `tar`, and `sha256sum` available in the Linux environment.
- The complete package, including all four `environment/runtime.part*` files. Python and the experiment dependencies are included; a separate Python or Conda installation is unnecessary.

**Environment matching is strict.** The reference experiment used an Intel Core i9-13900HX, 32 logical CPUs, and WSL 2 Linux kernel `6.18.33.2-microsoft-standard-WSL2`. The runner checks the CPU, platform, process settings, package versions, and thread configuration against [the runtime contract](environment/runtime_contract.json). Other hardware or kernels can be rejected even when the packaged runtime starts successfully.

### Verify the package and saved models

Install [Git LFS](https://git-lfs.com/), then clone the complete repository:

```bash
git lfs install
git clone https://github.com/wohahiha/fedSift.git
cd fedSift
git lfs pull
```

Run the following commands from the repository root. An existing complete copy of the repository can also be used offline.

**Linux / WSL**

```bash
bash run.sh verify
```

**Windows PowerShell**

```powershell
.\run.ps1 -Mode verify
```

On first use, the launcher verifies the runtime parts and their combined checksum, then extracts the runtime into `~/.cache/FedSift/runtime` by default. Subsequent commands reuse that installation. This setup works offline when all package files are present.

A successful verification prints a JSON report with `"status": "PASS"`. It checks the distribution manifest and source integrity, validates the execution environment, and recomputes **13,433 saved predictions from 70 models** with an absolute tolerance of `1e-12`. Reports are saved under `output/validation/`.

To select another installed WSL distribution:

```powershell
.\run.ps1 -Mode verify -Distribution Ubuntu-22.04
```

To choose a different runtime location from Linux / WSL, use an empty directory:

```bash
FEDSIFT_RUNTIME="$HOME/.local/share/FedSift/runtime" bash run.sh verify
```

## Experiments

### Datasets

| Dataset | Task | Records | Features | Local input |
| --- | --- | ---: | ---: | --- |
| Pima Indians Diabetes | Diabetes classification | 768 | 8 | [`data/diabetes.csv`](data/diabetes.csv) |
| Diabetic Retinopathy Debrecen | Classification of retinopathy signs from extracted image features | 1,151 | 19 | [`messidor_features.arff`](data/raw/diabetic_retinopathy_debrecen/messidor_features.arff) |

Both inputs are included. The [dataset registry](src/fedsift/dataset_registry.py) fixes their checksums, feature order, target encoding, and row counts. Data sources and attribution are listed [below](#data-attribution-and-licensing).

### Evaluation protocol

The [main experiment plan](config/experiment_plan.json) contains **70 training units: 2 datasets × 5 outer folds × 7 methods**, with one outer repeat. The saved main models are logistic classifiers with `float64` parameters.

| Method | Identifier | Role |
| --- | --- | --- |
| FedAvg | `fedavg_nonprivate` | Non-private baseline |
| DP-FedAvg | `dp_fedavg` | Differentially private averaging baseline |
| DP-FedAdam | `dp_fedadam` | Differentially private adaptive server baseline |
| FedSift | `fedsift` | Two-phase noise schedule and supported partial-step control |
| FedSift with uniform scheduling | `fedsift_uniform_schedule` | Replaces the two-phase noise schedule with a uniform schedule |
| FedSift without Sift | `fedsift_without_sift` | Disables public-control queries and uses the full server step |
| FedSift with public argmin | `fedsift_public_argmin_rule` | Chooses the step with the lowest public-control log loss |

Hyperparameter selection, public control, threshold selection, and outer evaluation have separate data-access roles. Selection decisions and preprocessing records are bound to each training unit; outer-test access is recorded explicitly. The [study design](config/study.json) specifies **8,640 inner-training units**: 2 datasets × 4 main methods × 5 outer folds × 24 candidates × 3 inner folds × 3 random streams. Ablations inherit the selected FedSift configuration. The complete inner predictions supporting the 40 selection decisions are included in [the selection evidence archive](output/reference/selection_evidence.zip).

Within each outer split, initialization is paired and fixed across methods and random streams. Seeds `101`, `211`, and `307` control the private sampling and noise streams during selection; non-private full-batch FedAvg is deterministic. Final refits use evaluation seed `939188524` with the same initialization convention. Candidate coordinates are paired within FedAvg/DP-FedAvg and within DP-FedAdam/FedSift; the latter pair shares both local and server optimizer settings.

The [evaluation plan](config/evaluation_plan.json) specifies privacy accounting, resource and communication measurements, membership inference, attribute reconstruction, and poisoning experiments. Attack evaluations use the six designated models from outer fold 0; their scope is narrower than the 70-unit main comparison. FedSift's control margin is a heuristic decision rule, as documented in [the implementation](src/fedsift/control_rule.py).

## Reproduction

Start with `verify`, then choose the level of reproduction needed. On Windows, use `.\run.ps1 -Mode <command>` for the corresponding command.

| Command | Action | Training scope |
| --- | --- | --- |
| `bash run.sh verify` | Check package integrity, environment, and saved-model predictions | No retraining |
| `bash run.sh rebuild` | Rebuild and compare main and follow-up result tables | Uses saved models and records |
| `bash run.sh train-smoke` | Retrain and compare one complete FedSift outer-fold unit per dataset | 2 units; outer-test access remains closed |
| `bash run.sh replay` | Retrain the main and poisoning experiments using the saved selections, then compare result tables | 70 main units and 12 poisoning units |
| `bash run.sh benchmark` | Repeat the resource measurement protocol with warm-up and balanced execution order | Dedicated resource runs |
| `bash run.sh analyze` | Reproduce paired record-group bootstrap intervals and editable bootstrap and poisoning plots | Saved outer predictions; 10,000 paired draws |
| `bash run.sh audit-search` | Verify archived inner predictions and recompute all 40 selection decisions | No retraining |
| `bash run.sh search` | Rerun the manuscript's hyperparameter search | 8,640 units |
| `bash run.sh tests` | Run algorithm and protocol regression tests in isolated processes | Reports passed, failed, and skipped cases separately |

`replay` uses the selections included in this package. `search` writes a separate search run and does not replace those selections. A complete search is substantially more expensive than checking saved results.

New experiments are written to timestamped directories under `output/runs/`; verification reports are written to `output/validation/`. The launchers preserve `output/reference/`. Access logs and failure records belong to their respective runs and must be retained to preserve the recorded evaluation history.

### Validation

The [acceptance record](provenance/acceptance.json) binds the delivered sources to their verification results. Tests cover per-record gradients against a direct autograd reference, local clipping and noise, fixed reference counts under record removal, control-set fallback, metric consistency, model export, selection boundaries, and replay in the packaged runtime. The test runner records actual passed, failed, and skipped counts.

`verify` independently reconstructs saved-model predictions. `audit-search` recomputes candidate metrics from concatenated inner-fold predictions before applying the declared lexicographic selection rule. `analyze` reproduces the paired group bootstrap using common resampling weights for every method comparison. Resource timings and memory measurements reflect the machine load during the measurement.

### Private model training

The command-line experiments use public datasets and reproducible random streams. For private records, [`train_private_model`](src/fedsift/private_training.py) uses fresh operating-system randomness for Poisson sampling and Gaussian noise, then returns only the final model, its architecture, and the single-run privacy accounting result. Internal record memberships, gradient commitments, seeds, and audit receipts are excluded from that output.

Fix the public auxiliary configuration before calling this API: preprocessing, client reference slots and counts, optimizer settings, and the training budget. [`seal_training_role_tables`](src/fedsift/train_unit.py) accepts the records present in those fixed private slots; removing a record leaves the normalization and aggregation weights unchanged. Public control and threshold-selection records remain fixed. The reported budget covers one model-training run; a workflow that selects settings or releases multiple models using private records needs the corresponding accounting.

## Results

Reference results can be inspected directly without running training:

| Artifact | Contents |
| --- | --- |
| [Method summary](output/reference/main_summary/method_summary.csv) | Metrics by dataset and method, with fold means, standard deviations, and confidence intervals |
| [Paired comparisons](output/reference/main_summary/paired_fedsift_differences.csv) | Paired differences between FedSift and the comparison methods |
| [Paired bootstrap](output/reference/main_summary/paired_group_bootstrap.csv) | Record-group bootstrap estimates and 95% intervals for log loss and Brier differences |
| [Out-of-fold predictions](output/reference/main_summary/oof_predictions.csv) | Saved predictions for the outer evaluation |
| [Privacy accounting](output/reference/followup/privacy_resource/privacy_accounting.csv) | Accounting results for the designated private training units |
| [Resource and communication summary](output/reference/followup/privacy_resource/resource_and_communication.csv) | Recorded computation and communication quantities |
| [Follow-up evaluations](output/reference/followup/) | Membership inference, reconstruction, and poisoning results |
| [Resource benchmark](output/reference/resource_benchmark/) | Dedicated timing and memory measurements |

Unit-level models and their evaluation records are retained in [`output/reference/main/`](output/reference/main/). Result tables can be regenerated with `rebuild` and compared against these fixed references.

## Repository structure

```text
.
├── src/fedsift/       # Federated training, control rules, privacy accounting, and evaluation
├── scripts/          # Search, training, summarization, attack, and benchmark entry points
├── tests/            # Algorithm and experiment-protocol regression tests
├── tools/            # Unified runner, result comparisons, and isolated test execution
├── config/           # Fixed experiment plans and execution authorization
├── data/             # Registered input datasets
├── environment/      # Offline runtime parts, dependency lock, and runtime contract
├── output/reference/ # Saved selections, trained models, results, and evidence archives
├── provenance/       # Study identities, source hashes, path mappings, and acceptance record
├── MANIFEST.json     # File sizes and SHA-256 hashes; excludes the manifest itself
├── run.sh            # Linux / WSL launcher
├── run.ps1           # Windows-to-WSL launcher
└── run.py            # Python command dispatcher
```

The packaged runtime uses **CPython 3.10.19**, **PyTorch 2.5.1**, **NumPy 2.2.6**, **SciPy 1.15.2**, **scikit-learn 1.7.2**, **Opacus 1.5.4**, and **pandas 2.3.3**. Exact builds are listed in [`environment/conda-lock.txt`](environment/conda-lock.txt) and [`environment/packages.json`](environment/packages.json).

The [source manifest](provenance/release_sources.json) identifies the executable files in this distribution. [Study identities](provenance/study_identity.json) identify the recorded experiment, while the [path map](provenance/path_map.json) resolves paths embedded in its records to locations in this package. Immutable record identifiers and random-seed namespaces are retained in `provenance/` so the saved experiment can be reconstructed exactly. The command-line execution scope is defined by `config/study.json`; broader registered inventories are used only to reconstruct those original identities.

## Git and large files

Use this directory as the repository root. Runtime parts, evidence archives, and large receipt files are tracked through [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage), as configured in [`.gitattributes`](.gitattributes).

For a Git checkout, install Git LFS, then retrieve and check the large-file contents from the repository root:

```bash
git lfs install
git lfs pull
git lfs fsck
```

Offline use requires the actual LFS objects; pointer files alone are insufficient. Finish retrieving all large-file contents before copying the repository to an offline machine. Preserve the repository's byte-handling rules because the manifests bind source code, data, and evidence to exact checksums.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| WSL cannot find `Ubuntu` | Run `wsl --list --quiet` in PowerShell and pass an installed distribution name with `-Distribution` |
| Runtime-part checksum fails | Confirm all four parts were extracted completely; for a Git checkout, run `git lfs pull` |
| Runtime directory is not empty | Choose a new, empty directory with `FEDSIFT_RUNTIME` and rerun the Linux launcher |
| Environment check rejects the host | Compare the reported mismatch with [`runtime_contract.json`](environment/runtime_contract.json); this package enforces the reference execution conditions |
| `Distribution file changed` | Compare the named file with [`MANIFEST.json`](MANIFEST.json) and restore it from the matching package |

## Data attribution and licensing

- **Pima Indians Diabetes:** original data from the National Institute of Diabetes and Digestive and Kidney Diseases. The bundled CSV is registered against [OpenML dataset 37](https://www.openml.org/api/v1/json/data/37), whose metadata labels the license as `Public`.
- **Diabetic Retinopathy Debrecen:** Balint Antal and Andras Hajdu (2014), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/329/diabetic%2Bretinopathy%2Bdebrecen), DOI [10.24432/C5XP4P](https://doi.org/10.24432/C5XP4P). The dataset is distributed under CC BY 4.0.

The project source code is released under the [MIT License](LICENSE), copyright (c) 2026 WOHAHIHA. Bundled datasets and third-party packages retain their respective license terms.
