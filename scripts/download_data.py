
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


DATASETS = {
    "module2": {
        "slug": "arafatsahinafridi/multi-class-driver-behavior-image-dataset",
        "zip_name": "multi-class-driver-behavior-image-dataset.zip",
        "extract_dir": "module2_distraction",
    },
    "module3": {
        "slug": "amanmehra23/travel-recommendation-dataset",
        "zip_name": "travel-recommendation-dataset.zip",
        "extract_dir": "module3_recommender",
    },
}


def download_dataset(output_dir: str, module: str = "module2") -> Path:
    dataset = DATASETS[module]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset["slug"],
        "-p",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No esta instalada la CLI de Kaggle. Ejecuta: pip install kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle no pudo descargar el dataset. Verifica que kaggle.json exista en "
            "%USERPROFILE%\\.kaggle\\kaggle.json y que aceptaste las condiciones del dataset."
        ) from exc

    zip_path = output_path / dataset["zip_name"]
    if not zip_path.exists():
        zip_files = sorted(output_path.glob("*.zip"), key=lambda path: path.stat().st_mtime)
        if not zip_files:
            raise FileNotFoundError("Kaggle no genero un archivo .zip en el directorio destino.")
        matching_zips = [
            path for path in zip_files
            if dataset["slug"].split("/")[-1] in path.stem
        ]
        if not matching_zips:
            available = ", ".join(path.name for path in zip_files)
            raise FileNotFoundError(
                f"No encontre el ZIP esperado para {module}. ZIPs disponibles: {available}"
            )
        zip_path = matching_zips[-1]
    extract_dir = output_path / dataset["extract_dir"]
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)

    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga datasets del proyecto desde Kaggle.")
    parser.add_argument("--output-dir", default="data/raw", help="Directorio para guardar el dataset.")
    parser.add_argument("--module", default="module2", choices=sorted(DATASETS), help="Dataset a descargar.")
    args = parser.parse_args()
    extract_dir = download_dataset(args.output_dir, module=args.module)
    print(f"Dataset extraido en: {extract_dir}")


if __name__ == "__main__":
    main()
