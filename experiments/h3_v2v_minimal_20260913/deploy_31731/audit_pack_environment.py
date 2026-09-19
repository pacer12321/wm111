"""Read-only gate for known conda/pip metadata differences in the source env."""
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

PREFIX = Path("/cache/yunfeng/envs/minimax-h3-npu")
SITE = "lib/python3.12/site-packages/"
ALLOWED_MISSING = {
    *(SITE + "setuptools-83.0.0-py3.12.egg-info/" + name for name in (
        "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt")),
    SITE + "setuptools/_distutils/compilers/C/py.typed",
    SITE + "setuptools/_distutils/py.typed",
    SITE + "setuptools/launcher manifest.xml",
    *(SITE + "wheel-0.47.0.dist-info/" + name for name in (
        "METADATA", "RECORD", "WHEEL", "entry_points.txt", "licenses/LICENSE.txt")),
}


def main():
    if Path(sys.prefix).resolve() != PREFIX:
        raise RuntimeError("Audit must run with the original H3 interpreter")
    missing = set()
    for meta_path in (PREFIX / "conda-meta").glob("*.json"):
        metadata = json.loads(meta_path.read_text())
        for name in metadata.get("files", []):
            path = PREFIX / name
            if not path.exists() and not path.is_symlink():
                missing.add(name)
    if missing != ALLOWED_MISSING:
        raise RuntimeError("Conda missing-file set changed: " + repr(sorted(missing ^ ALLOWED_MISSING)))
    records = {}
    for name, version in (("setuptools", "80.10.2"), ("wheel", "0.48.0")):
        dist = importlib.metadata.distribution(name)
        if dist.version != version:
            raise RuntimeError(f"Actual pip distribution changed: {name} {dist.version}")
        count = 0
        for item in dist.files or []:
            path = Path(dist.locate_file(item)).resolve()
            if not path.is_relative_to(PREFIX) or not path.is_file():
                raise RuntimeError(f"Missing/escaping current distribution file: {name}/{item}")
            if item.hash:
                digest = hashlib.new(item.hash.mode, path.read_bytes()).digest()
                value = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                if value != item.hash.value:
                    raise RuntimeError(f"RECORD hash mismatch: {name}/{item}")
                count += 1
        records[name] = {"version": version, "record_files": len(dist.files or []), "verified_hashes": count}
    print(json.dumps({"status": "passed", "allowed_legacy_missing_count": len(missing),
                      "allowed_legacy_missing": sorted(missing), "current_distribution_records": records}))


if __name__ == "__main__":
    main()
