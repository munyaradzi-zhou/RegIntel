// =============================================================================
// RegIntel — dedicated resources for one environment (resource-group scope)
//
//   monitoring   Log Analytics + Application Insights
//   identity     user-assigned identity both apps run as
//   registry     Azure Container Registry (images built with `az acr build`)
//   keyvault     API client keys (written by the deploy script, never by a template)
//   cosmos       serverless Cosmos DB for conversation sessions
//   shared       role grants on the shared Azure AI Search, Azure OpenAI and AI Services
//                resources (reused from LegalMind's resource group to stay within budget)
//   apps         Container Apps environment + API + UI (only when deployApps = true, i.e.
//                after the images and the Key Vault secrets exist)
//   cd           identity GitHub Actions deploys as, through OIDC (only when githubRepository
//                is set): push images, update the two apps, read the smoke-test key, and
//                `join` on the environment when it belongs to another project
//
// Deployed by scripts/azure/deploy.sh. Every service is reached with Entra ID; no keys.
// =============================================================================
targetScope = 'resourceGroup'

@description('Region for the RegIntel resources.')
param location string = resourceGroup().location

@description('Region for the Container Apps environment and apps. Separate because a free-trial subscription allows one environment per region, and another project may already hold one in `location`.')
param appsLocation string = location

@allowed([ 'dev', 'prod' ])
param environmentName string = 'dev'

param projectName string = 'regintel'

@description('Existing Container Apps environment to run the apps in (empty: create one).')
param existingEnvironmentId string = ''

@description('Second step of the deployment: create the container apps (images and secrets must exist).')
param deployApps bool = false

@description('Full image references, e.g. <registry>/regintel-api:<git sha>.')
param apiImage string = ''
param uiImage string = ''

@description('Object id of the person deploying (may write Key Vault secrets).')
param deployerPrincipalId string = ''

@description('Shared resources RegIntel uses (reused from the LegalMind resource group; see scripts/azure/setup_dev_on_legalmind.sh).')
param sharedResourceGroup string
param searchServiceName string
param openAiName string
param judgeName string
param chatDeployment string = 'gpt-5-mini'
param judgeDeployment string = 'gpt-oss-120b'

@description('GitHub repository (owner/name) allowed to deploy through OIDC. Empty: no CD identity.')
param githubRepository string = ''

@description('The OIDC subject GitHub sends for that repository. Repositories created since 2026 use immutable ids (repo:owner@<id>/name@<id>:ref:...); deploy.sh looks them up. Empty: the classic repo:owner/name form.')
param githubSubjectRepository string = ''

var name = '${projectName}-${environmentName}'
var tags = { project: projectName, environment: environmentName, managedBy: 'bicep' }

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: { location: location, name: name, tags: tags }
}

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: { location: location, name: name, tags: tags }
}

module registry 'modules/registry.bicep' = {
  name: 'registry'
  params: { location: location, name: name, tags: tags, pullPrincipalId: identity.outputs.principalId }
}

module keyvault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    location: location
    name: name
    tags: tags
    readerPrincipalId: identity.outputs.principalId
    deployerPrincipalId: deployerPrincipalId
  }
}

module cosmos 'modules/cosmos.bicep' = {
  name: 'cosmos'
  params: { location: location, name: name, tags: tags, writerPrincipalId: identity.outputs.principalId }
}

module shared 'modules/shared-access.bicep' = {
  name: 'shared-access'
  scope: resourceGroup(sharedResourceGroup)
  params: {
    principalId: identity.outputs.principalId
    searchServiceName: searchServiceName
    openAiName: openAiName
    judgeName: judgeName
  }
}

module apps 'modules/container-apps.bicep' = if (deployApps) {
  name: 'container-apps'
  params: {
    location: appsLocation
    name: name
    tags: tags
    workspaceCustomerId: monitoring.outputs.workspaceCustomerId
    workspaceId: monitoring.outputs.workspaceId
    appInsightsConnectionString: monitoring.outputs.connectionString
    identityId: identity.outputs.id
    identityClientId: identity.outputs.clientId
    registryServer: registry.outputs.loginServer
    keyVaultUri: keyvault.outputs.uri
    apiImage: apiImage
    uiImage: uiImage
    searchEndpoint: shared.outputs.searchEndpoint
    openAiEndpoint: shared.outputs.openAiEndpoint
    chatDeployment: chatDeployment
    judgeEndpoint: shared.outputs.judgeEndpoint
    judgeDeployment: judgeDeployment
    cosmosEndpoint: cosmos.outputs.endpoint
    existingEnvironmentId: existingEnvironmentId
  }
}

module cd 'modules/cd-identity.bicep' = if (!empty(githubRepository)) {
  name: 'cd-identity'
  dependsOn: [ apps ]
  params: {
    location: location
    name: name
    tags: tags
    githubRepository: empty(githubSubjectRepository) ? githubRepository : githubSubjectRepository
    registryName: registry.outputs.name
    keyVaultName: keyvault.outputs.name
    appNames: deployApps ? [ 'ca-${name}-api', 'ca-${name}-ui' ] : []
  }
}

// Apps in another project's environment: CD also needs `join` on that environment to update them.
module cdJoin 'modules/cd-environment-join.bicep' = if (!empty(githubRepository) && !empty(existingEnvironmentId) && deployApps) {
  name: 'cd-environment-join'
  scope: resourceGroup(split(existingEnvironmentId, '/')[4])
  params: {
    environmentName: last(split(existingEnvironmentId, '/'))
    principalId: cd.?outputs.principalId ?? ''
    roleName: 'Container Apps environment join (${name})'
  }
}

output registryName string = registry.outputs.name
output registryServer string = registry.outputs.loginServer
output keyVaultName string = keyvault.outputs.name
output cosmosEndpoint string = cosmos.outputs.endpoint
output identityClientId string = identity.outputs.clientId
output apiUrl string = apps.?outputs.apiUrl ?? ''
output uiUrl string = apps.?outputs.uiUrl ?? ''
output cdClientId string = cd.?outputs.clientId ?? ''
