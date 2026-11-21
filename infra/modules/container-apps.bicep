// Container Apps environment (or an existing one) + the API and the UI. Both run as the shared user-assigned identity.
// Scale to zero when idle (dev, trial credit): the first request after a pause waits for a cold
// start (the API loads its data and builds its clients, roughly 20-40 s).

param location string
param name string
param tags object
param workspaceCustomerId string
param workspaceId string
param appInsightsConnectionString string
param identityId string
param identityClientId string
param registryServer string
param keyVaultUri string
param apiImage string
param uiImage string
param searchEndpoint string
param openAiEndpoint string
param chatDeployment string
param judgeEndpoint string
param judgeDeployment string
param cosmosEndpoint string

@description('Reference the `openfda-api-key` secret. Only when it exists: a missing secret stops the revision from starting.')
param openFdaKeyInVault bool = false

@description('Use an existing Container Apps environment instead of creating one (free-trial subscriptions allow only one per subscription). Must be in `location`.')
param existingEnvironmentId string = ''

var createEnvironment = empty(existingEnvironmentId)

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: last(split(workspaceId, '/'))
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = if (createEnvironment) {
  name: 'cae-${name}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspaceCustomerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
  }
}

var environmentId = createEnvironment ? environment.id : existingEnvironmentId

// Key Vault references, read with the identity. `api-keys` is the API's list of client keys;
// `ui-api-key` is the one key the UI presents (it is also in that list).
var apiKeySecret = { name: 'api-keys', keyVaultUrl: '${keyVaultUri}secrets/api-keys', identity: identityId }
var uiKeySecret = { name: 'ui-api-key', keyVaultUrl: '${keyVaultUri}secrets/ui-api-key', identity: identityId }
var openFdaSecret = { name: 'openfda-api-key', keyVaultUrl: '${keyVaultUri}secrets/openfda-api-key', identity: identityId }

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${name}-api'
  location: location
  tags: union(tags, { 'azd-service-name': 'api' })
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identityId}': {} } }
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 8000, transport: 'auto', allowInsecure: false }
      registries: [ { server: registryServer, identity: identityId } ]
      secrets: openFdaKeyInVault ? [ apiKeySecret, openFdaSecret ] : [ apiKeySecret ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat([
            { name: 'AZURE_CLIENT_ID', value: identityClientId } // which identity DefaultAzureCredential uses
            { name: 'SEARCH_BACKEND', value: 'azure' }
            { name: 'AZURE_SEARCH_ENDPOINT', value: searchEndpoint }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openAiEndpoint }
            { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatDeployment }
            { name: 'AZURE_JUDGE_ENDPOINT', value: judgeEndpoint }
            { name: 'AZURE_JUDGE_DEPLOYMENT', value: judgeDeployment }
            { name: 'COSMOS_ENDPOINT', value: cosmosEndpoint }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsightsConnectionString }
            { name: 'API_KEYS', secretRef: 'api-keys' }
            { name: 'DATA_DIR', value: '/app/data' }
          ], openFdaKeyInVault ? [ { name: 'OPENFDA_API_KEY', secretRef: 'openfda-api-key' } ] : [])
          probes: [
            // timeoutSeconds 5: the default 1 s restarted the container under load (probes queue behind requests)
            { type: 'Startup', httpGet: { path: '/healthz', port: 8000 }, initialDelaySeconds: 3, periodSeconds: 5, timeoutSeconds: 5, failureThreshold: 30 }
            { type: 'Liveness', httpGet: { path: '/healthz', port: 8000 }, periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 3 }
            { type: 'Readiness', httpGet: { path: '/readyz', port: 8000 }, periodSeconds: 10, timeoutSeconds: 5, failureThreshold: 3 }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 2, rules: [ { name: 'http', http: { metadata: { concurrentRequests: '10' } } } ] }
    }
  }
}

resource ui 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${name}-ui'
  location: location
  tags: union(tags, { 'azd-service-name': 'ui' })
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identityId}': {} } }
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 8501, transport: 'auto', allowInsecure: false }
      registries: [ { server: registryServer, identity: identityId } ]
      secrets: [ uiKeySecret ]
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: uiImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'REGINTEL_API_URL', value: 'https://${api.properties.configuration.ingress.fqdn}' }
            { name: 'REGINTEL_API_KEY', secretRef: 'ui-api-key' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/_stcore/health', port: 8501 }, periodSeconds: 30, failureThreshold: 3 }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output uiUrl string = 'https://${ui.properties.configuration.ingress.fqdn}'
