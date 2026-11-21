// Cosmos DB (NoSQL, serverless) for conversation sessions and the audit trail: pay per request, nothing while idle.
// Local (key) auth is disabled; the apps' identity gets the built-in data contributor role.

param location string
param name string
param tags object
param writerPrincipalId string

var dataContributor = '00000000-0000-0000-0000-000000000002' // Cosmos DB Built-in Data Contributor

resource account 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: 'cosmos-${name}-${take(uniqueString(resourceGroup().id), 5)}'
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    locations: [ { locationName: location, failoverPriority: 0, isZoneRedundant: false } ]
    capabilities: [ { name: 'EnableServerless' } ]
    disableLocalAuth: true
    minimalTlsVersion: 'Tls12'
    publicNetworkAccess: 'Enabled'
  }
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  parent: account
  name: 'regintel'
  properties: { resource: { id: 'regintel' } }
}

resource sessions 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: database
  name: 'sessions'
  properties: {
    resource: {
      id: 'sessions'
      partitionKey: { paths: [ '/id' ], kind: 'Hash' }
      defaultTtl: -1 // expiry is set per document (7 days after the last turn)
    }
  }
}

// The audit trail: one record per answer, hash-chained, one partition per day. No TTL: records are
// kept. The API creates and reads; nothing in the application updates or deletes a record.
resource audit 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: database
  name: 'audit'
  properties: {
    resource: {
      id: 'audit'
      partitionKey: { paths: [ '/day' ], kind: 'Hash' }
    }
  }
}

resource write 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = {
  parent: account
  name: guid(account.id, writerPrincipalId, dataContributor)
  properties: {
    principalId: writerPrincipalId
    roleDefinitionId: '${account.id}/sqlRoleDefinitions/${dataContributor}'
    scope: account.id
  }
}

output endpoint string = account.properties.documentEndpoint
