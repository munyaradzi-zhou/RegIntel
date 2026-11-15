#!/usr/bin/env bash
# Set up RegIntel's dev environment on top of LegalMind's existing Azure resources.
#
# Why reuse: the subscription is a free trial with a fixed credit, and LegalMind's Basic search
# service already bills around the clock. RegIntel adds only what it needs, prefixed `regintel-`:
#
#   1. a gpt-5-mini deployment on LegalMind's Azure OpenAI resource (pay per token)
#   2. a system-assigned managed identity on the search service, so the knowledge base can call
#      Azure OpenAI and read Blob Storage without keys
#   3. role assignments for that identity (Cognitive Services User, Storage Blob Data Reader)
#   4. a blob container with the 23 guidance PDFs, each tagged with its manifest metadata
#
# Idempotent: every step checks or uses create-or-update semantics, so it is safe to re-run.
# RegIntel's own Bicep (phase 8) defines dedicated resources for a clean, standalone deployment.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would rewrite "/subscriptions/..." scopes into file paths

RG="${RG:-rg-legalmind-dev}"
SEARCH="${SEARCH:-srch-legalmind-dev-5bbwe}"
OPENAI="${OPENAI:-oai-legalmind-dev-5bbwe}"
STORAGE="${STORAGE:-stlegalminddev5bbwe}"
CONTAINER="${CONTAINER:-regintel-guidance}"
CHAT_MODEL="${CHAT_MODEL:-gpt-5-mini}"
CHAT_VERSION="${CHAT_VERSION:-2025-08-07}"
CHAT_TPM_K="${CHAT_TPM_K:-100}"   # capacity in thousands of tokens per minute (quota: 500)
# The reviewer runs on its own AI Services resource: gpt-oss-120b is not offered on the Azure OpenAI
# resource, and o4-mini (the first choice) no longer accepts new deployments.
JUDGE="${JUDGE:-ais-regintel-dev-5bbwe}"
JUDGE_LOCATION="${JUDGE_LOCATION:-eastus2}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"
JUDGE_VERSION="${JUDGE_VERSION:-1}"
JUDGE_TPM_K="${JUDGE_TPM_K:-50}"
PDF_DIR="${PDF_DIR:-data/raw/guidance}"
MANIFEST="${MANIFEST:-data/manifest.json}"

say() { printf '\n== %s\n' "$*"; }

say "1. ${CHAT_MODEL} deployment on ${OPENAI}"
if az cognitiveservices account deployment show -g "$RG" -n "$OPENAI" --deployment-name "$CHAT_MODEL" >/dev/null 2>&1; then
  echo "exists"
else
  az cognitiveservices account deployment create -g "$RG" -n "$OPENAI" \
    --deployment-name "$CHAT_MODEL" --model-name "$CHAT_MODEL" --model-version "$CHAT_VERSION" \
    --model-format OpenAI --sku-name GlobalStandard --sku-capacity "$CHAT_TPM_K" -o none
  echo "created (${CHAT_TPM_K}K TPM)"
fi

say "1b. ${JUDGE_MODEL} on ${JUDGE} (answer reviewer: a different model from the writer)"
if ! az cognitiveservices account show -g "$RG" -n "$JUDGE" -o none 2>/dev/null; then
  az cognitiveservices account create -g "$RG" -n "$JUDGE" -l "$JUDGE_LOCATION" --kind AIServices \
    --sku S0 --custom-domain "$JUDGE" --yes -o none
fi
if az cognitiveservices account deployment show -g "$RG" -n "$JUDGE" --deployment-name "$JUDGE_MODEL" >/dev/null 2>&1; then
  echo "exists"
else
  az cognitiveservices account deployment create -g "$RG" -n "$JUDGE" \
    --deployment-name "$JUDGE_MODEL" --model-name "$JUDGE_MODEL" --model-version "$JUDGE_VERSION" \
    --model-format OpenAI-OSS --sku-name GlobalStandard --sku-capacity "$JUDGE_TPM_K" -o none
  echo "created (${JUDGE_TPM_K}K TPM)"
fi
echo "AZURE_JUDGE_ENDPOINT=https://${JUDGE}.cognitiveservices.azure.com/"

say "2. System-assigned identity on ${SEARCH}"
PRINCIPAL=$(az search service show -g "$RG" -n "$SEARCH" --query identity.principalId -o tsv)
if [[ -z "$PRINCIPAL" ]]; then
  az search service update -g "$RG" -n "$SEARCH" --identity-type SystemAssigned -o none
  PRINCIPAL=$(az search service show -g "$RG" -n "$SEARCH" --query identity.principalId -o tsv)
fi
echo "principal: $PRINCIPAL"

say "3. Roles for the search identity"
OPENAI_ID=$(az cognitiveservices account show -g "$RG" -n "$OPENAI" --query id -o tsv)
STORAGE_ID=$(az storage account show -g "$RG" -n "$STORAGE" --query id -o tsv)
assign() {  # role scope
  if [[ -n "$(az role assignment list --assignee "$PRINCIPAL" --role "$1" --scope "$2" --query '[0].id' -o tsv)" ]]; then
    echo "has '$1'"
  else
    az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
      --role "$1" --scope "$2" -o none
    echo "granted '$1'"
  fi
}
assign "Cognitive Services User" "$OPENAI_ID"      # knowledge base -> gpt-5-mini, integrated vectorization
assign "Storage Blob Data Reader" "$STORAGE_ID"    # blob knowledge source -> read the PDFs

say "4. Container ${CONTAINER} and the guidance PDFs"
az storage container create --account-name "$STORAGE" --name "$CONTAINER" --auth-mode login -o none
python - "$MANIFEST" "$PDF_DIR" "$STORAGE" "$CONTAINER" <<'PY'
"""Upload each PDF with its manifest metadata, so blob-indexed chunks know their issuer and date."""
import json, subprocess, sys
from pathlib import Path

manifest, pdf_dir, account, container = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for doc in json.loads(Path(manifest).read_text(encoding="utf-8"))["guidance"]:
    metadata = [f"doc_id={doc['id']}", f"issuer={doc['issuer']}", f"jurisdiction={doc['jurisdiction']}",
                f"doc_date={doc['date']}", f"source_url={doc['url']}"]
    subprocess.run(
        ["az", "storage", "blob", "upload", "--account-name", account, "--container-name", container,
         "--name", f"{doc['id']}.pdf", "--file", str(pdf_dir / f"{doc['id']}.pdf"), "--auth-mode", "login",
         "--overwrite", "--content-type", "application/pdf", "--metadata", *metadata, "-o", "none"],
        check=True, shell=sys.platform == "win32",
    )
    print("uploaded", doc["id"])
PY

say "Done. Endpoints for .env:"
echo "AZURE_SEARCH_ENDPOINT=https://${SEARCH}.search.windows.net"
echo "AZURE_OPENAI_ENDPOINT=$(az cognitiveservices account show -g "$RG" -n "$OPENAI" --query properties.endpoint -o tsv)"
echo "AZURE_STORAGE_ACCOUNT=${STORAGE}  AZURE_STORAGE_CONTAINER=${CONTAINER}"
