#!/usr/bin/env bash
# =============================================================================
# RegIntel — deploy to Azure (idempotent: re-run to roll out a new version)
#
#   scripts/azure/deploy.sh                 # dev, image tag = current git commit
#   TAG=v1 RG=rg-regintel-dev scripts/azure/deploy.sh
#   CA_BUNDLE=roots.pem scripts/azure/deploy.sh   # HTTPS-scanning antivirus/proxy on the build machine
#   CD_REPOSITORY=owner/repo scripts/azure/deploy.sh   # also create the GitHub Actions OIDC identity
#
# 1. resource providers and the resource group
# 2. infrastructure (main.bicep, deployApps=false): monitoring, identity, registry, Key Vault,
#    Cosmos DB, role grants on the shared search and model resources
# 3. API keys generated here and written straight to Key Vault (never printed or committed)
# 4. images: built locally and pushed (default), or built in the registry with IMAGE_BUILD=acr
#    (ACR Tasks are blocked on free-trial subscriptions); the runtime data is also pushed as its
#    own image, regintel-data:<content hash>, which CD builds use (the data is not in git)
# 5. the container apps (main.bicep, deployApps=true)
# 6. smoke test: health, readiness, one answered question
#
# Needs: az login, Docker (for the default local build), the data in data/ (`make corpus ingest`), Bash.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash on Windows: keep /subscriptions/... paths intact

RG="${RG:-rg-regintel-dev}"
LOCATION="${LOCATION:-eastus2}"
PARAMS="${PARAMS:-infra/parameters/dev.bicepparam}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
IMAGE_BUILD="${IMAGE_BUILD:-local}"  # local: docker build + push (needs Docker); acr: az acr build
CD_REPOSITORY="${CD_REPOSITORY:-}"  # owner/repo: create the identity GitHub Actions deploys as
CD_SUBJECT=""  # the repository as GitHub's OIDC subject names it (owner@<id>/name@<id> on new repositories)
if [ -n "$CD_REPOSITORY" ]; then
  CD_SUBJECT=$(gh api "repos/${CD_REPOSITORY}" --jq '"\(.owner.login)@\(.owner.id)/\(.name)@\(.id)"')
fi

say() { printf '\n== %s\n' "$*"; }
output() { az deployment group show -g "$RG" -n "$1" --query "properties.outputs.$2.value" -o tsv; }

say "1. Providers and resource group ${RG} (${LOCATION})"
for ns in Microsoft.App Microsoft.ContainerRegistry Microsoft.DocumentDB Microsoft.KeyVault \
          Microsoft.OperationalInsights Microsoft.Insights Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait -o none
done
az group create -n "$RG" -l "$LOCATION" --tags project=regintel managedBy=bicep -o none

say "2. Infrastructure"
ME=$(az ad signed-in-user show --query id -o tsv)
az deployment group create -g "$RG" -n regintel-infra -f infra/main.bicep -p "$PARAMS" \
  -p deployerPrincipalId="$ME" deployApps=false githubRepository="$CD_REPOSITORY" githubSubjectRepository="$CD_SUBJECT" -o none
REGISTRY=$(output regintel-infra registryName)
SERVER=$(output regintel-infra registryServer)
VAULT=$(output regintel-infra keyVaultName)
echo "registry ${SERVER}, key vault ${VAULT}"

say "3. API keys in Key Vault"
if az keyvault secret show --vault-name "$VAULT" -n ui-api-key -o none 2>/dev/null; then
  echo "keys exist (rotate by deleting the secrets and re-running)"
else
  UI_KEY=$(openssl rand -hex 32)
  OWNER_KEY=$(openssl rand -hex 32)
  for attempt in 1 2 3 4 5 6; do  # the Secrets Officer role can take a minute to apply
    if az keyvault secret set --vault-name "$VAULT" -n ui-api-key --value "$UI_KEY" -o none 2>/dev/null; then break; fi
    echo "waiting for Key Vault permissions (${attempt})"; sleep 20
  done
  az keyvault secret set --vault-name "$VAULT" -n owner-api-key --value "$OWNER_KEY" -o none
  az keyvault secret set --vault-name "$VAULT" -n api-keys --value "${UI_KEY},${OWNER_KEY}" -o none
  echo "created: ui-api-key (the UI), owner-api-key (you: curl, MCP), api-keys (what the API accepts)"
fi

say "4. Images ${TAG} (${IMAGE_BUILD} build)"
if [ "$IMAGE_BUILD" = "acr" ]; then
  # Builds inside the registry; not allowed on free-trial subscriptions (TasksOperationsNotAllowed).
  az acr build -r "$REGISTRY" -t "regintel-api:${TAG}" -f Dockerfile . -o none
  az acr build -r "$REGISTRY" -t "regintel-ui:${TAG}" -f Dockerfile.frontend . -o none
else
  az acr login -n "$REGISTRY"
  DATA_TAG=$(find data/manifest.json data/processed data/raw/cdisc -type f -print0 | sort -z \
    | xargs -0 sha256sum | sha256sum | cut -c1-12)
  # CA_BUNDLE: PEM file of trusted roots, for machines whose antivirus or proxy re-signs HTTPS
  secret=()
  if [ -n "${CA_BUNDLE:-}" ]; then secret=(--secret "id=ca_cert,src=${CA_BUNDLE}"); fi
  docker build "${secret[@]}" -t "${SERVER}/regintel-api:${TAG}" -f Dockerfile .
  docker build "${secret[@]}" -t "${SERVER}/regintel-ui:${TAG}" -f Dockerfile.frontend .
  docker push "${SERVER}/regintel-api:${TAG}"
  docker push "${SERVER}/regintel-ui:${TAG}"
  if ! az acr repository show -n "$REGISTRY" --image "regintel-data:${DATA_TAG}" -o none 2>/dev/null; then
    docker build -t "${SERVER}/regintel-data:${DATA_TAG}" --target data-local -f Dockerfile .
    docker push "${SERVER}/regintel-data:${DATA_TAG}"
  fi
  echo "data image: regintel-data:${DATA_TAG} (the CD workflow's REGINTEL_DATA_TAG)"
fi

say "5. Container apps"
az deployment group create -g "$RG" -n regintel-apps -f infra/main.bicep -p "$PARAMS" \
  -p deployerPrincipalId="$ME" deployApps=true githubRepository="$CD_REPOSITORY" githubSubjectRepository="$CD_SUBJECT" \
  apiImage="${SERVER}/regintel-api:${TAG}" uiImage="${SERVER}/regintel-ui:${TAG}" -o none
API_URL=$(output regintel-apps apiUrl)
UI_URL=$(output regintel-apps uiUrl)

say "6. Smoke test"
scripts/azure/smoke_test.sh "$API_URL" "$VAULT"
printf '\nAPI: %s/docs\nUI:  %s\n' "$API_URL" "$UI_URL"
if [ -n "$CD_REPOSITORY" ]; then
  printf 'CD:  repository variables AZURE_CLIENT_ID=%s AZURE_TENANT_ID=%s AZURE_SUBSCRIPTION_ID=%s\n' \
    "$(output regintel-apps cdClientId)" "$(az account show --query tenantId -o tsv)" \
    "$(az account show --query id -o tsv)"
fi
