"""Exact local-source registry for the two real FedSift datasets.

Synthetic rows remain useful for unit tests, but they are not accepted by this
registry.  Formal study construction must start here so a renamed, reformatted,
or silently replaced source file cannot enter an experiment under an old
dataset identity.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from .group_manifest import DatasetRows, load_arff_rows, load_csv_rows


class DatasetRegistryError(RuntimeError):
    """Raised when a formal real-data source differs from its registry."""


PIMA_PREDICTORS: tuple[str, ...] = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
DEBRECEN_PREDICTORS: tuple[str, ...] = tuple((str(index) for index in range(19)))


@dataclass(frozen=True, slots=True)
class RegisteredDataset:
    dataset_id: str
    relative_path: str
    file_format: str
    file_sha256: str
    target_name: str
    predictor_names: tuple[str, ...]
    row_count: int
    positive_count: int
    authoritative_metadata_url: str
    authoritative_download_url: str
    source_artifact_sha256: str
    provenance_status: str
    license_statement: str


PIMA = RegisteredDataset(
    dataset_id="pima",
    relative_path="data/diabetes.csv",
    file_format="csv",
    file_sha256="b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac",
    target_name="Outcome",
    predictor_names=PIMA_PREDICTORS,
    row_count=768,
    positive_count=268,
    authoritative_metadata_url="https://www.openml.org/api/v1/json/data/37",
    authoritative_download_url="https://openml.org/data/v1/download/37/diabetes.arff",
    source_artifact_sha256="4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40",
    provenance_status="local_csv_numeric_rows_and_binary_labels_verified_equal_in_order_to_openml_dataset_37_on_2026_08_31",
    license_statement="OpenML dataset 37 metadata declares licence Public",
)
DEBRECEN = RegisteredDataset(
    dataset_id="retinopathy",
    relative_path="data/raw/diabetic_retinopathy_debrecen/messidor_features.arff",
    file_format="arff",
    file_sha256="b83485dd519127ac1ba06a95da5f1d12d4bde2a362eb32d7d353bdcabfa08a93",
    target_name="Class",
    predictor_names=DEBRECEN_PREDICTORS,
    row_count=1151,
    positive_count=611,
    authoritative_metadata_url="https://archive.ics.uci.edu/dataset/329/diabetic%2Bretinopathy%2Bdebrecen",
    authoritative_download_url="https://archive.ics.uci.edu/static/public/329/diabetic%2Bretinopathy%2Bdebrecen.zip",
    source_artifact_sha256="64ee2dbaffc69dab77cc0fb7458fa93cb21a20f15938cafacdc195a5e2226b2c",
    provenance_status="local_arff_sha256_verified_equal_to_the_sole_uci_329_zip_entry_on_2026_08_31",
    license_statement="UCI dataset 329 is licensed CC BY 4.0",
)
REGISTERED_DATASETS: dict[str, RegisteredDataset] = {
    PIMA.dataset_id: PIMA,
    DEBRECEN.dataset_id: DEBRECEN,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_dataset(dataset_id: str) -> RegisteredDataset:
    try:
        return REGISTERED_DATASETS[dataset_id]
    except (KeyError, TypeError) as exc:
        raise DatasetRegistryError("dataset is outside the frozen real-data registry") from exc


def load_registered_dataset(workspace_root: Path | str, dataset_id: str) -> DatasetRows:
    """Load one exact real source and recheck its content-level invariants."""
    spec = registered_dataset(dataset_id)
    root = Path(workspace_root).resolve()
    source = (root / Path(spec.relative_path)).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise DatasetRegistryError("registered dataset path escapes the workspace") from exc
    if not source.is_file():
        raise DatasetRegistryError("registered dataset file is missing")
    observed_hash = _sha256_file(source)
    if observed_hash != spec.file_sha256:
        raise DatasetRegistryError("registered dataset file SHA-256 differs")
    if spec.file_format == "csv":
        rows = load_csv_rows(
            source,
            dataset=spec.dataset_id,
            target=spec.target_name,
            predictors=spec.predictor_names,
        )
    elif spec.file_format == "arff":
        rows = load_arff_rows(
            source,
            dataset=spec.dataset_id,
            target=spec.target_name,
            predictors=spec.predictor_names,
        )
    else:
        raise DatasetRegistryError("registered dataset format is unsupported")
    if (
        rows.source_sha256 != spec.file_sha256
        or rows.feature_names != spec.predictor_names
        or rows.target_name != spec.target_name
        or (len(rows.row_ids) != spec.row_count)
        or (sum(rows.labels) != spec.positive_count)
        or (rows.predictor_allowlist_explicit is not True)
    ):
        raise DatasetRegistryError("registered dataset semantic invariants differ")
    return DatasetRows(
        dataset=rows.dataset,
        source_path=f"registered://{spec.dataset_id}/{spec.file_sha256}",
        source_sha256=rows.source_sha256,
        feature_names=rows.feature_names,
        features=rows.features,
        labels=rows.labels,
        row_ids=rows.row_ids,
        target_name=rows.target_name,
        predictor_allowlist_explicit=rows.predictor_allowlist_explicit,
    )


__all__ = [
    "DEBRECEN",
    "DEBRECEN_PREDICTORS",
    "DatasetRegistryError",
    "PIMA",
    "PIMA_PREDICTORS",
    "REGISTERED_DATASETS",
    "RegisteredDataset",
    "load_registered_dataset",
    "registered_dataset",
]
