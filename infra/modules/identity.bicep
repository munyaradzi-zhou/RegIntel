// One user-assigned identity for both container apps: it pulls images, reads the API key from
// Key Vault, writes sessions to Cosmos DB and calls Azure AI Search and the model resources.
// Nothing in the apps holds a key to any Azure service.

param location string
param name string
param tags object

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${name}'
  location: location
  tags: tags
}

output id string = identity.id
output principalId string = identity.properties.principalId
output clientId string = identity.properties.clientId
