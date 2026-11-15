// Log Analytics (container logs) + Application Insights (OpenTelemetry traces from the API).

param location string
param name string
param tags object

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${name}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: 1 } // a runaway log loop cannot eat the trial credit
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${name}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspace.id
    DisableLocalAuth: false
  }
}

output workspaceId string = workspace.id
output workspaceCustomerId string = workspace.properties.customerId
output connectionString string = insights.properties.ConnectionString
