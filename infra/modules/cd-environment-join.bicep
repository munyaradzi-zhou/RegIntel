// The one permission CD needs on a Container Apps environment it does not own: updating an app
// re-checks that the caller may place apps in the app's environment (managedEnvironments/join).
// Used only when the apps run in another project's environment (existingEnvironmentId); a custom
// role with that single action, assignable in that resource group only, granted on the
// environment only. It gives no access to the environment's other apps.

param environmentName string
param principalId string
param roleName string

var joinAction = 'Microsoft.App/managedEnvironments/join/action'

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}

resource join 'Microsoft.Authorization/roleDefinitions@2022-04-01' = {
  name: guid(resourceGroup().id, roleName)
  properties: {
    roleName: roleName
    description: 'Place container apps in a Container Apps environment (join only).'
    type: 'CustomRole'
    permissions: [ { actions: [ joinAction ], notActions: [] } ]
    assignableScopes: [ resourceGroup().id ]
  }
}

resource grant 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(environment.id, principalId, join.id)
  scope: environment
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: join.id
  }
}
