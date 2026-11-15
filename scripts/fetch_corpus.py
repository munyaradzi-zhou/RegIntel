"""Download everything listed in data/manifest.json into data/raw/.

python scripts/fetch_corpus.py                 # guidance + CDISC + FAERS sample
python scripts/fetch_corpus.py --only guidance
"""

from __future__ import annotations

import argparse
from pathlib import Path

from regintel.corpus.fetch import fetch_cdisc, fetch_faers, fetch_guidance, make_client
from regintel.corpus.manifest import load_manifest

try:  # use the OS certificate store when available (corporate proxies, Windows)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

STEPS = {"guidance": fetch_guidance, "cdisc": fetch_cdisc, "faers": fetch_faers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--only", choices=sorted(STEPS), action="append")
    args = parser.parse_args()

    manifest = load_manifest(args.data_dir / "manifest.json")
    with make_client() as client:
        for name in args.only or list(STEPS):
            result = STEPS[name](manifest, args.data_dir / "raw" / name, client)
            print(f"{name}: {result.downloaded} downloaded, {result.skipped} already present")


if __name__ == "__main__":
    main()
