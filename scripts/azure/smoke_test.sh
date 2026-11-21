#!/usr/bin/env bash
# Smoke test a deployed API: health, readiness (waits through a cold start), auth, one answer
# (which must have an audit record), and the audit trail's hash chain.
#   scripts/azure/smoke_test.sh https://ca-regintel-dev-api.<...>.azurecontainerapps.io kv-regintel-dev-xxxxx
set -euo pipefail
API="${1:?API URL}"
VAULT="${2:?Key Vault name}"
KEY=$(az keyvault secret show --vault-name "$VAULT" -n owner-api-key --query value -o tsv)

echo "health:  $(curl -s -o /dev/null -w '%{http_code}' "${API}/healthz")"
for _ in $(seq 1 36); do  # scale-to-zero: the first request starts a replica
  code=$(curl -s -o /dev/null -w '%{http_code}' "${API}/readyz")
  [ "$code" = "200" ] && break
  sleep 5
done
echo "ready:   ${code} $(curl -s "${API}/readyz")"
[ "$code" = "200" ] || exit 1

echo "no key:  $(curl -s -o /dev/null -w '%{http_code}' -X POST "${API}/v1/ask" -H 'content-type: application/json' -d '{"question":"q"}') (expect 401)"
curl -s -X POST "${API}/v1/ask" -H "X-API-Key: ${KEY}" -H 'content-type: application/json' \
  -d '{"question":"What is the minimum number of evaluable subjects in an EU bioequivalence study?"}' \
  | python -c "import json,sys; b=json.load(sys.stdin); print('answer: ', b['label'], b['path'], '|', b['answer'][:160]); print('source: ', b['sources'][0]['label'] if b['sources'] else '-'); print('audit:  ', b['audit_id'])"
curl -s "${API}/v1/audit/verify" -H "X-API-Key: ${KEY}" \
  | python -c "import json,sys; r=json.load(sys.stdin); print('chain:  ', 'ok' if r['ok'] else 'BROKEN', r['records_checked'], 'records', r['first_break'] or ''); sys.exit(0 if r['ok'] else 1)"
