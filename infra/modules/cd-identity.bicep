// Continuous deployment from GitHub Actions without a stored secret: a user-assigned identity that
// trusts GitHub's OIDC tokens for one repository and branch (a federated credential). The workflow
// signs in as it, pushes images and points the apps at them; it cannot change the infrastructure.
//
//   AcrPush            on the registry      push the new images
//   Reader             on the registry      look up its login server (`az acr login`, `az acr show`)
//   Contributor        on the two apps      roll out a new revision (not on the resource group)
//   Secrets User       on the Key Vault     read the owner key for the smoke test

param location string
param name string
param tags object
@description('GitHub repository allowed to deploy, as it appears in the OIDC subject: owner/name, or owner@<id>/name@<id>.')
param githubRepository string
param githubBranch string = 'main'
param registryName string
param keyVaultName string
@description('Names of the container apps the workflow may update (empty before the apps exist).')
param appNames array = []

var acrPush = '8311e382-0749-4cb8-b61a-304f252e45ec'
var reader = 'acdd72a7-3385-48ef-bd42-f606fba81ae7'
var contributor = 'b24988ac-6180-42a0-ab88-20f7382dd24c'
var secretsUser = '4633458b-17de-408a-b874-0445c86b69e6'

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${name}-cd'
  location: location
  tags: tags
}

resource github 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: identity
  name: 'github-${githubBranch}'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: 'repo:${githubRepository}:ref:refs/heads/${githubBranch}'
    audiences: [ 'api://AzureADTokenExchange' ]
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource push 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, acrPush)
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPush)
  }
}

resource lookup 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, reader)
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', reader)
  }
}

resource secrets 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, identity.id, secretsUser)
  scope: vault
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', secretsUser)
  }
}

resource apps 'Microsoft.App/containerApps@2024-03-01' existing = [for app in appNames: {
  name: app
}]

resource update 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (app, i) in appNames: {
  name: guid(apps[i].id, identity.id, contributor)
  scope: apps[i]
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributor)
  }
}]

output clientId string = identity.properties.clientId
output principalId string = identity.properties.principalId
