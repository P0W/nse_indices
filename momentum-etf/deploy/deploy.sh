#!/bin/bash

# ETF Momentum Strategy - Azure Container Apps Deployment Script (Bash)
# This script deploys the ETF momentum strategy to Azure Container Apps

set -e  # Exit on any error

# Default values
RESOURCE_GROUP_NAME="rg-momentum-etf-dev"
LOCATION="East US"
ENVIRONMENT="dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

print_color $GREEN "🚀 ETF Momentum Strategy - Azure Container Apps Deployment"
print_color $GREEN "================================================="

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--subscription)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        -g|--resource-group)
            RESOURCE_GROUP_NAME="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -s SUBSCRIPTION_ID [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  -s, --subscription      Azure subscription ID"
            echo ""
            echo "Optional:"
            echo "  -g, --resource-group    Resource group name (default: rg-momentum-etf-dev)"
            echo "  -l, --location          Azure region (default: East US)"
            echo "  -e, --environment       Environment name (default: dev)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -s 12345678-1234-1234-1234-123456789012"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if subscription ID is provided
if [ -z "$SUBSCRIPTION_ID" ]; then
    print_color $RED "❌ Subscription ID is required. Use -s or --subscription flag."
    echo "Example: $0 -s 12345678-1234-1234-1234-123456789012"
    exit 1
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_color $RED "❌ Azure CLI is not installed. Please install it from https://aka.ms/GetTheAzureCLI"
    exit 1
fi
print_color $GREEN "✅ Azure CLI is installed"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_color $RED "❌ Docker is not installed. Please install Docker"
    exit 1
fi
print_color $GREEN "✅ Docker is installed"

# Check if jq is installed (for JSON parsing)
if ! command -v jq &> /dev/null; then
    print_color $YELLOW "⚠️ jq is not installed. Installing jq for JSON parsing..."
    
    # Install jq based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install jq
        else
            print_color $RED "❌ Homebrew not found. Please install jq manually: https://stedolan.github.io/jq/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y jq
        elif command -v yum &> /dev/null; then
            sudo yum install -y jq
        else
            print_color $RED "❌ Unable to install jq automatically. Please install jq manually: https://stedolan.github.io/jq/"
            exit 1
        fi
    else
        print_color $RED "❌ Unsupported OS. Please install jq manually: https://stedolan.github.io/jq/"
        exit 1
    fi
fi

# Login to Azure
print_color $YELLOW "🔐 Checking Azure login status..."
CURRENT_ACCOUNT=$(az account show --output json 2>/dev/null)
if [ $? -eq 0 ]; then
    CURRENT_SUB_ID=$(echo "$CURRENT_ACCOUNT" | jq -r '.id')
    if [ "$CURRENT_SUB_ID" = "$SUBSCRIPTION_ID" ]; then
        print_color $GREEN "✅ Already logged in to Azure with subscription $SUBSCRIPTION_ID"
    else
        print_color $YELLOW "🔐 Logging into Azure..."
        az login
        
        # Set subscription after login
        print_color $YELLOW "📋 Setting subscription to $SUBSCRIPTION_ID..."
        az account set --subscription "$SUBSCRIPTION_ID"
    fi
else
    print_color $YELLOW "🔐 Logging into Azure..."
    az login
    
    # Set subscription after login
    print_color $YELLOW "📋 Setting subscription to $SUBSCRIPTION_ID..."
    az account set --subscription "$SUBSCRIPTION_ID"
fi

# Create resource group
print_color $YELLOW "📂 Creating resource group $RESOURCE_GROUP_NAME..."
az group create --name "$RESOURCE_GROUP_NAME" --location "$LOCATION"

# Deploy infrastructure
print_color $YELLOW "🏗️ Deploying Azure infrastructure..."
DEPLOYMENT_OUTPUT=$(az deployment group create \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --template-file "./infra/main.bicep" \
    --parameters "@./infra/parameters.json" \
    --output json)

# Check deployment status
PROVISIONING_STATE=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.provisioningState')
if [ "$PROVISIONING_STATE" != "Succeeded" ]; then
    print_color $RED "❌ Infrastructure deployment failed"
    exit 1
fi

# Get deployment outputs
ACR_LOGIN_SERVER=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerRegistryLoginServer.value')
ACR_NAME=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerRegistryName.value')
CONTAINER_APP_URL=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerAppUrl.value')
CONTAINER_APP_NAME=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerAppName.value')

print_color $GREEN "✅ Infrastructure deployed successfully!"
print_color $CYAN "📊 Container Registry: $ACR_LOGIN_SERVER"
print_color $CYAN "🌐 Container App URL: $CONTAINER_APP_URL"

# Login to ACR
print_color $YELLOW "🔐 Logging into Azure Container Registry..."
az acr login --name "$ACR_NAME"

# Build and push Docker image
print_color $YELLOW "🐳 Building and pushing Docker image..."
IMAGE_NAME="$ACR_LOGIN_SERVER/momentum-etf:latest"

# Build and push Docker image
print_color $YELLOW "🐳 Building and pushing Docker image..."
IMAGE_NAME="$ACR_LOGIN_SERVER/momentum-etf:latest"

# Change to project root for Docker build
CURRENT_DIR=$(pwd)
cd ..

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with optimizations
docker build \
    --platform linux/amd64 \
    --progress=plain \
    --tag "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    cd "$CURRENT_DIR"
    print_color $RED "❌ Docker build failed"
    exit 1
fi

docker push "$IMAGE_NAME"
if [ $? -ne 0 ]; then
    cd "$CURRENT_DIR"
    print_color $RED "❌ Docker push failed"
    exit 1
fi

# Return to deploy directory
cd "$CURRENT_DIR"

# Clean up old Docker images locally
print_color $YELLOW "🧹 Cleaning up old Docker images..."
docker image prune -f
docker system prune -f

# Clean up old images in Azure Container Registry (keep only latest 3)
print_color $YELLOW "🧹 Cleaning up old images in Azure Container Registry..."
OLD_IMAGES=$(az acr repository show-tags --name "$ACR_NAME" --repository momentum-etf --output json 2>/dev/null | jq -r '.[] | select(. != "latest")' | head -n -2)
if [ ! -z "$OLD_IMAGES" ]; then
    echo "$OLD_IMAGES" | while read -r tag; do
        if [ ! -z "$tag" ]; then
            print_color $GRAY "   Deleting old image: momentum-etf:$tag"
            az acr repository delete --name "$ACR_NAME" --image "momentum-etf:$tag" --yes 2>/dev/null || true
        fi
    done
    print_color $GREEN "   Cleaned up old images"
else
    print_color $GREEN "   No old images to clean up"
fi

# Update container app with new image
print_color $YELLOW "🔄 Updating container app..."
az containerapp update \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --image "$IMAGE_NAME"

print_color $GREEN ""
print_color $GREEN "🎉 Deployment completed successfully!"
print_color $GREEN "================================================="
print_color $NC "📱 Your ETF Momentum Strategy API is now running at:"
print_color $CYAN "🌐 $CONTAINER_APP_URL"
print_color $NC ""
print_color $NC "📋 Available endpoints:"
print_color $GRAY "   • Portfolio:    $CONTAINER_APP_URL/portfolio"
print_color $GRAY "   • Rebalance:    $CONTAINER_APP_URL/rebalance"
print_color $GRAY "   • Historical:   $CONTAINER_APP_URL/historical"
print_color $GRAY "   • API Docs:     $CONTAINER_APP_URL/docs"
print_color $GRAY "   • Health Check: $CONTAINER_APP_URL/health"
print_color $NC ""
print_color $YELLOW "💡 Test your API:"
print_color $GRAY "   curl -X GET \"$CONTAINER_APP_URL/portfolio?amount=1000000&size=5\""
print_color $NC ""
print_color $YELLOW "🛠️ To update your application:"
print_color $GRAY "   1. Make changes to your code"
print_color $GRAY "   2. Run this script again"
