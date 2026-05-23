import shutil
from pathlib import Path

ARTIFACT_DIRS = ("outputs", "masks")
ARTIFACT_FILES = ("metrics.csv",)


def main() -> None:
    data_root = Path("assets/data")
    if not data_root.exists():
        return

    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        for filename in ARTIFACT_FILES:
            artifact_path = dataset_dir / filename
            if artifact_path.exists():
                artifact_path.unlink()
                print(f"removed {artifact_path}")

        for dirname in ARTIFACT_DIRS:
            artifact_dir = dataset_dir / dirname
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
                print(f"removed {artifact_dir}")


if __name__ == "__main__":
    main()
