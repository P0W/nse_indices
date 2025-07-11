#!/bin/bash
# ETF Momentum Strategy - Azure Resource Cleanup Script (Bash)
# This script deletes all Azure resources created for the ETF momentum strategy

set -e

# Default values
RESOURCE_GROUP_NAME="rg-momentum-etf-dev"
FORCE=false

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

print_color $RED "🗑️ ETF Momentum Strategy - Azure Resource Cleanup"
print_color $RED "============================================="

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
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 -s SUBSCRIPTION_ID [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  -s, --subscription      Azure subscription ID"
            echo ""
            echo "Optional:"
            echo "  -g, --resource-group    Resource group name (default: rg-momentum-etf-dev)"
            echo "  -f, --force             Skip confirmation prompt"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -s 12345678-1234-1234-1234-123456789012"
            echo "  $0 -s 12345678-1234-1234-1234-123456789012 --force"
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

# Check if resource group exists
print_color $YELLOW "🔍 Checking if resource group exists..."
if ! az group show --name "$RESOURCE_GROUP_NAME" &>/dev/null; then
    print_color $CYAN "ℹ️ Resource group '$RESOURCE_GROUP_NAME' does not exist. Nothing to clean up."
    exit 0
fi

# List resources in the group
print_color $CYAN "📋 Resources in '$RESOURCE_GROUP_NAME':"
az resource list --resource-group "$RESOURCE_GROUP_NAME" --output table

print_color $RED ""
print_color $RED "⚠️ WARNING: This will DELETE ALL resources in the resource group!"
print_color $YELLOW "📦 Resources to be deleted:"
print_color $GRAY "   • Container App (momentum-etf-dev-app)"
print_color $GRAY "   • Container Registry (momentumetfdevacrXXXXX)"
print_color $GRAY "   • Log Analytics Workspace (momentum-etf-dev-logs)"
print_color $GRAY "   • Container Apps Environment (momentum-etf-dev-env)"
print_color $GRAY "   • All Docker images and logs"
echo ""

if [ "$FORCE" != "true" ]; then
    read -p "Are you sure you want to delete all resources? Type 'DELETE' to confirm: " confirmation
    if [ "$confirmation" != "DELETE" ]; then
        print_color $YELLOW "❌ Cleanup cancelled by user"
        exit 0
    fi
fi

# Delete the entire resource group
print_color $RED "🗑️ Deleting resource group '$RESOURCE_GROUP_NAME'..."
print_color $YELLOW "⏳ This may take several minutes..."

if az group delete --name "$RESOURCE_GROUP_NAME" --yes --no-wait; then
    print_color $GREEN "✅ Deletion initiated successfully!"
    echo ""
    print_color $NC "📋 Cleanup Status:"
    print_color $GREEN "   • Deletion started: Resource group and all contents"
    print_color $CYAN "   • Process: Running in background (--no-wait)"
    print_color $GRAY "   • Duration: 5-15 minutes typically"
    echo ""
    print_color $YELLOW "🔍 Check deletion progress:"
    print_color $GRAY "   az group show --name $RESOURCE_GROUP_NAME"
    echo ""
    print_color $GREEN "💰 Cost Impact:"
    print_color $GRAY "   • Container Apps: No longer billing"
    print_color $GRAY "   • Container Registry: Storage freed"
    print_color $GRAY "   • Log Analytics: Data retention only"
    echo ""
    print_color $GREEN "✨ Cleanup complete! All resources are being deleted."
else
    print_color $RED "❌ Failed to delete resource group"
    echo ""
    print_color $YELLOW "🔧 Manual cleanup options:"
    print_color $GRAY "   1. Azure Portal: portal.azure.com → Resource Groups → Delete"
    print_color $GRAY "   2. Retry command: az group delete --name $RESOURCE_GROUP_NAME --yes"
    exit 1
fi
