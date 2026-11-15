// Role assignments on the shared resources RegIntel reuses (deployed into their resource group):
// the Azure AI Search service, the Azure OpenAI resource (writer, router, agent) and the AI
// Services resource (gpt-oss-120b reviewer). Data-plane roles only; no keys.

param principalId string
param searchServiceName string
param openAiName string
param judgeName string

var openAiUser = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd' // Cognitive Services OpenAI User
var searchReader = '1407120a-92aa-4202-b7e9-c0e197c71c8f' // Search Index Data Reader (also allows KB retrieve)

resource search 'Microsoft.Search/searchServices@2024-06-01-preview' existing = {
  name: searchServiceName
}

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: openAiName
}

resource judge 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: judgeName
}

resource searchRead 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, principalId, searchReader)
  scope: search
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchReader)
  }
}

resource openAiUse 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, principalId, openAiUser)
  scope: openAi
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAiUser)
  }
}

resource judgeUse 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(judge.id, principalId, openAiUser)
  scope: judge
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAiUser)
  }
}

output searchEndpoint string = 'https://${search.name}.search.windows.net'
output openAiEndpoint string = openAi.properties.endpoint
output judgeEndpoint string = judge.properties.endpoint
