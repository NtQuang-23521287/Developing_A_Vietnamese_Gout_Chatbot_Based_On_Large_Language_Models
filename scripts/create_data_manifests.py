from pathlib import Path

from gout_llmops.core.io import write_json
from gout_llmops.data.manifest import build_directory_manifest


def main():
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    write_json(
        "data/manifests/kb_raw_manifest.json",
        build_directory_manifest("data/raw/knowledge_base"),
    )
    write_json(
        "data/manifests/testset_raw_manifest.json",
        build_directory_manifest("data/raw/testset"),
    )
    print("Created data manifests in data/manifests/")


if __name__ == "__main__":
    main()
