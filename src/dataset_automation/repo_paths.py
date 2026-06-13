from pathlib import Path


def find_repo_root(start: Path = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "lib").is_dir():
            return parent
    raise RuntimeError(
        f"Could not locate repo root above {here}: no ancestor contains a 'lib/' dir."
    )


REPO_ROOT = find_repo_root()

MAIN_CONFIG = REPO_ROOT / "config.yaml"
LIB_DIR = REPO_ROOT / "lib"
TRACK_GENERATOR_DIR = LIB_DIR / "track-generator"  
TRACK_GENERATOR_CONFIG = TRACK_GENERATOR_DIR / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
BUILD_DIR = REPO_ROOT / "build"
SCRIPTS_DIR = REPO_ROOT / "scripts"
RECONSTRUCTOR_BIN = BUILD_DIR / "track_to_centerline"