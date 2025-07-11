# ETF Momentum Strategy - Azure Container Apps Deployment Script (PowerShell)
# This script deploys the ETF momentum strategy to Azure Container Apps

param(
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-momentum-etf-dev",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "South India",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Initialize variables
$UseAcrBuild = $false

Write-Host "🚀 ETF Momentum Strategy - Azure Container Apps Deployment" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check if Azure CLI is installed
try {
    az version | Out-Null
    Write-Host "✅ Azure CLI is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI is not installed. Please install it from https://aka.ms/GetTheAzureCLI" -ForegroundColor Red
    exit 1
}

# Check if Docker is installed and working
try {
    # Add Docker to PATH if needed
    $dockerPath = "C:\Program Files\Docker\Docker\resources\bin"
    if ((Test-Path $dockerPath) -and ($env:PATH -notlike "*$dockerPath*")) {
        $env:PATH += ";$dockerPath"
    }
    
    docker --version | Out-Null
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker is installed and running" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Docker installed but daemon not accessible, using ACR build" -ForegroundColor Yellow
        $UseAcrBuild = $true
    }
} catch {
    Write-Host "⚠️ Docker not accessible, using ACR build" -ForegroundColor Yellow
    $UseAcrBuild = $true
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

# Create resource group
Write-Host "📂 Creating resource group $ResourceGroupName..." -ForegroundColor Yellow
az group create --name $ResourceGroupName --location $Location

# Deploy infrastructure
Write-Host "🏗️ Deploying Azure infrastructure..." -ForegroundColor Yellow
$deployment = az deployment group create `
    --resource-group $ResourceGroupName `
    --template-file "./infra/main.bicep" `
    --parameters "@./infra/parameters.json" `
    --verbose `
    --output json | ConvertFrom-Json

if ($deployment.properties.provisioningState -ne "Succeeded") {
    Write-Host "❌ Infrastructure deployment failed" -ForegroundColor Red
    exit 1
}

# Get deployment outputs
$outputs = $deployment.properties.outputs
$acrLoginServer = $outputs.containerRegistryLoginServer.value
$acrName = $outputs.containerRegistryName.value
$containerAppUrl = $outputs.containerAppUrl.value

Write-Host "✅ Infrastructure deployed successfully!" -ForegroundColor Green
Write-Host "📊 Container Registry: $acrLoginServer" -ForegroundColor Cyan
Write-Host "🌐 Container App URL: $containerAppUrl" -ForegroundColor Cyan

# Login to ACR
Write-Host "🔐 Logging into Azure Container Registry..." -ForegroundColor Yellow
az acr login --name $acrName

# Build and push Docker image
Write-Host "🐳 Building and pushing Docker image..." -ForegroundColor Yellow
$imageName = "$acrLoginServer/momentum-etf:latest"

# Change to project root for Docker build
$currentDir = Get-Location
Set-Location ".."

if ($UseAcrBuild) {
    Write-Host "☁️ Using Azure Container Registry build..." -ForegroundColor Cyan
    az acr build --registry $acrName --image momentum-etf:latest .
} else {
    Write-Host "🐳 Using local Docker build..." -ForegroundColor Cyan
    # Enable BuildKit for faster builds
    $env:DOCKER_BUILDKIT = "1"
    
    # Build with optimizations
    docker build --platform linux/amd64 --tag $imageName .
    
    if ($LASTEXITCODE -ne 0) {
        Set-Location $currentDir
        Write-Host "❌ Docker build failed" -ForegroundColor Red
        exit 1
    }
    
    docker push $imageName
}

# Return to deploy directory
Set-Location $currentDir

# Clean up old Docker images locally
Write-Host "🧹 Cleaning up old Docker images..." -ForegroundColor Yellow
docker image prune -f
docker system prune -f

# Clean up old images in Azure Container Registry (keep only latest 3)
Write-Host "🧹 Cleaning up old images in Azure Container Registry..." -ForegroundColor Yellow
try {
    $oldImages = az acr repository show-tags --name $acrName --repository momentum-etf --output json | ConvertFrom-Json | Sort-Object -Descending | Select-Object -Skip 3
    if ($oldImages.Count -gt 0) {
        foreach ($tag in $oldImages) {
            Write-Host "   Deleting old image: momentum-etf:$tag" -ForegroundColor Gray
            az acr repository delete --name $acrName --image "momentum-etf:$tag" --yes 2>$null
        }
        Write-Host "   Deleted $($oldImages.Count) old image(s)" -ForegroundColor Green
    } else {
        Write-Host "   No old images to clean up" -ForegroundColor Green
    }
} catch {
    Write-Host "   Note: Unable to clean up ACR images (may not exist yet)" -ForegroundColor Yellow
}

# Update container app with new image
Write-Host "🔄 Updating container app..." -ForegroundColor Yellow
$containerAppName = $outputs.containerAppName.value
az containerapp update `
    --name $containerAppName `
    --resource-group $ResourceGroupName `
    --image $imageName

Write-Host "" -ForegroundColor White
Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host "📱 Your ETF Momentum Strategy API is now running at:" -ForegroundColor White
Write-Host "🌐 $containerAppUrl" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "📋 Available endpoints:" -ForegroundColor White
Write-Host "   • Portfolio:    $containerAppUrl/portfolio" -ForegroundColor Gray
Write-Host "   • Rebalance:    $containerAppUrl/rebalance" -ForegroundColor Gray
Write-Host "   • Historical:   $containerAppUrl/historical" -ForegroundColor Gray
Write-Host "   • API Docs:     $containerAppUrl/docs" -ForegroundColor Gray
Write-Host "   • Health Check: $containerAppUrl/health" -ForegroundColor Gray
Write-Host "" -ForegroundColor White
Write-Host "💡 Test your API:" -ForegroundColor Yellow
Write-Host "   curl -X GET $containerAppUrl/health" -ForegroundColor Gray
Write-Host "   curl -X GET $containerAppUrl/portfolio" -ForegroundColor Gray
Write-Host "" -ForegroundColor White
Write-Host "🛠️ To update your application:" -ForegroundColor Yellow
Write-Host "   1. Make changes to your code" -ForegroundColor Gray
Write-Host "   2. Run this script again" -ForegroundColor Gray
