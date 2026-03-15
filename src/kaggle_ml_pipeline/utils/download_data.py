from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


COMPETITION = "playground-series-s5e7"
REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def _has_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    return bool((__import__("os").environ.get("KAGGLE_USERNAME")) and (__import__("os").environ.get("KAGGLE_KEY")))


def _ensure_kaggle_cli() -> None:
    if shutil.which("kaggle"):
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    if not shutil.which("kaggle"):
        raise RuntimeError("Kaggle CLI is not available after installation attempt.")


def download_competition_data(data_dir: Path | str = "data", competition: str = COMPETITION) -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    _ensure_kaggle_cli()
    if not _has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials are missing. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or create ~/.kaggle/kaggle.json before downloading."
        )

    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(data_path), "--force"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}".strip()
        if "401" in combined or "Unauthorized" in combined:
            raise RuntimeError(
                "Kaggle competition access denied (401). Open the competition page, "
                "join/accept rules with this account, and ensure account verification is complete. "
                f"Then rerun: {' '.join(cmd)}"
            )
        raise RuntimeError(f"Kaggle download failed: {combined}")

    zip_path = data_path / f"{competition}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    missing = [name for name in REQUIRED_FILES if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Downloaded archive is missing required files: {', '.join(missing)}")

    return data_path


def main() -> None:
    path = download_competition_data()
    print(f"Data downloaded and extracted to: {path}")


if __name__ == "__main__":
    main()
