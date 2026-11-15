"""Create (or update) the Blob knowledge source and report ingestion progress.

python scripts/azure/build_blob_knowledge_source.py            # create/update, then show status
python scripts/azure/build_blob_knowledge_source.py --status   # status only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from azure.identity import AzureCliCredential

from regintel.azure_search.knowledge import AzureNames, blob_knowledge_source, index_client, storage_resource_id
from regintel.config import get_settings

RESOURCE_GROUP = "rg-legalmind-dev"


def subscription_id() -> str:
    out = subprocess.run(
        ["az", "account", "show", "--query", "id", "-o", "tsv"],  # noqa: S607
        capture_output=True, text=True, check=True, shell=sys.platform == "win32",
    )  # fmt: skip
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    names = AzureNames(settings.azure_name_prefix)
    client = index_client(settings, AzureCliCredential())

    if not args.status:
        if not settings.azure_storage_account:
            raise SystemExit("AZURE_STORAGE_ACCOUNT is not set")
        connection = storage_resource_id(subscription_id(), RESOURCE_GROUP, settings.azure_storage_account)
        created = client.create_or_update_knowledge_source(blob_knowledge_source(settings, names, connection))
        print("created/updated:", created.name)
        print(json.dumps(created.as_dict().get("azureBlobParameters", {}).get("createdResources", {}), indent=2))

    status = client.get_knowledge_source_status(names.blob_ks)
    print(json.dumps(status.as_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
