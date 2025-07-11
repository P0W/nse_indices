# ETF Momentum Strategy - Azure Resource Cleanup Script (PowerShell)
# This script deletes all Azure resources created for the ETF momentum strategy

param(
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-momentum-etf-dev",
    
    [Parameter(Mandatory=$false)]
    [switch]$Force = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🗑️ ETF Momentum Strategy - Azure Resource Cleanup" -ForegroundColor Red
Write-Host "=============================================" -ForegroundColor Red

# Check if Azure CLI is installed
try {
    az version | Out-Null
    Write-Host "✅ Azure CLI is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI is not installed. Please install it from https://aka.ms/GetTheAzureCLI" -ForegroundColor Red
    exit 1
}

# Login to Azure
Write-Host "🔐 Checking Azure login status..." -ForegroundColor Yellow
$currentAccount = az account show --output json 2>$null | ConvertFrom-Json
if ($currentAccount -and $currentAccount.id -eq $SubscriptionId) {
    Write-Host "✅ Already logged in to Azure with subscription $SubscriptionId" -ForegroundColor Green
} else {
    Write-Host "🔐 Logging into Azure..." -ForegroundColor Yellow
    az login
    
    # Set subscription after login
    Write-Host "📋 Setting subscription to $SubscriptionId..." -ForegroundColor Yellow
    az account set --subscription $SubscriptionId
}

# Check if resource group exists
Write-Host "🔍 Checking if resource group exists..." -ForegroundColor Yellow
$resourceGroup = az group show --name $ResourceGroupName --output json 2>$null | ConvertFrom-Json

if (-not $resourceGroup) {
    Write-Host "ℹ️ Resource group '$ResourceGroupName' does not exist. Nothing to clean up." -ForegroundColor Cyan
    exit 0
}

# List resources in the group
Write-Host "📋 Resources in '$ResourceGroupName':" -ForegroundColor Cyan
$resources = az resource list --resource-group $ResourceGroupName --output table
Write-Host $resources

Write-Host ""
Write-Host "⚠️ WARNING: This will DELETE ALL resources in the resource group!" -ForegroundColor Red
Write-Host "📦 Resources to be deleted:" -ForegroundColor Yellow
Write-Host "   • Container App (momentum-etf-dev-app)" -ForegroundColor Gray
Write-Host "   • Container Registry (momentumetfdevacrXXXXX)" -ForegroundColor Gray
Write-Host "   • Log Analytics Workspace (momentum-etf-dev-logs)" -ForegroundColor Gray
Write-Host "   • Container Apps Environment (momentum-etf-dev-env)" -ForegroundColor Gray
Write-Host "   • All Docker images and logs" -ForegroundColor Gray
Write-Host ""

if (-not $Force) {
    $confirmation = Read-Host "Are you sure you want to delete all resources? Type 'DELETE' to confirm"
    if ($confirmation -ne "DELETE") {
        Write-Host "❌ Cleanup cancelled by user" -ForegroundColor Yellow
        exit 0
    }
}

# Delete the entire resource group
Write-Host "🗑️ Deleting resource group '$ResourceGroupName'..." -ForegroundColor Red
Write-Host "⏳ This may take several minutes..." -ForegroundColor Yellow

try {
    az group delete --name $ResourceGroupName --yes --no-wait
    
    Write-Host "✅ Deletion initiated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Cleanup Status:" -ForegroundColor White
    Write-Host "   • Deletion started: Resource group and all contents" -ForegroundColor Green
    Write-Host "   • Process: Running in background (--no-wait)" -ForegroundColor Cyan
    Write-Host "   • Duration: 5-15 minutes typically" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔍 Check deletion progress:" -ForegroundColor Yellow
    Write-Host "   az group show --name $ResourceGroupName" -ForegroundColor Gray
    Write-Host ""
    Write-Host "💰 Cost Impact:" -ForegroundColor Green
    Write-Host "   • Container Apps: No longer billing" -ForegroundColor Gray
    Write-Host "   • Container Registry: Storage freed" -ForegroundColor Gray
    Write-Host "   • Log Analytics: Data retention only" -ForegroundColor Gray
    Write-Host ""
    Write-Host "✨ Cleanup complete! All resources are being deleted." -ForegroundColor Green
    
} catch {
    Write-Host "❌ Failed to delete resource group: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Manual cleanup options:" -ForegroundColor Yellow
    Write-Host "   1. Azure Portal: portal.azure.com → Resource Groups → Delete" -ForegroundColor Gray
    Write-Host "   2. Retry command: az group delete --name $ResourceGroupName --yes" -ForegroundColor Gray
    exit 1
}
