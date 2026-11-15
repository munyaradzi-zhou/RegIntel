using '../main.bicep'

// Development environment on the free-trial subscription. The search service and the model
// resources are shared with LegalMind (same subscription, its resource group) to stay in budget.
param location = 'eastus2'
// The free trial allows ONE Container Apps environment per subscription and LegalMind holds it,
// so the RegIntel apps run in LegalMind's environment (eastus2). Everything else is RegIntel's own.
param appsLocation = 'eastus2'
param existingEnvironmentId = '/subscriptions/5b70b0a3-8af1-4746-81a7-018bf0a00be3/resourceGroups/rg-legalmind-dev/providers/Microsoft.App/managedEnvironments/cae-legalmind-dev'
param environmentName = 'dev'
param sharedResourceGroup = 'rg-legalmind-dev'
param searchServiceName = 'srch-legalmind-dev-5bbwe'
param openAiName = 'oai-legalmind-dev-5bbwe'
param judgeName = 'ais-regintel-dev-5bbwe'
param chatDeployment = 'gpt-5-mini'
param judgeDeployment = 'gpt-oss-120b'
