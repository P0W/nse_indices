# ETF Momentum Strategy - Azure Container Apps Deployment

This guide will help you deploy your ETF Momentum Strategy Web Application to Azure Container Apps in a cost-effective way.

## 🏗️ Architecture

- **Azure Container Registry**: Stores your Docker images
- **Azure Container Apps**: Hosts your containerized web application with beautiful UI and HTTP APIs
- **Azure Log Analytics**: Monitoring and logging
- **Single Resource Group**: All resources in one place for easy management

## 📁 Project Structure

All deployment-related files are organized in the `deploy/` folder:
```
deploy/
├── deploy.ps1              # Windows deployment script
├── deploy.sh               # Linux/macOS deployment script
├── docker-compose.yml      # Local container testing
├── status.ps1              # Check deployment status
├── cleanup.ps1/.sh         # Resource cleanup scripts
├── optimize-build-cache.ps1/.sh  # Build optimization
└── infra/
    ├── main.bicep          # Azure infrastructure as code
    └── parameters.json     # Infrastructure parameters
```

## 💰 Cost Optimization

- Uses **Basic** Container Registry tier
- Container Apps with minimal CPU/memory allocation (0.5 CPU, 1 GB RAM)
- Auto-scaling from 1-3 replicas based on demand
- 30-day log retention
- **Optimized logging**: File logging disabled in production to reduce storage costs and improve performance
- **Clean temporary files**: Automatic cleanup prevents disk space buildup

## 📋 Prerequisites

### Required Tools
- **Azure CLI**: [Install here](https://aka.ms/GetTheAzureCLI)
- **Docker**: [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **uv**: Already installed (modern Python package manager)

### For Linux/macOS only:
- **jq**: JSON processor (auto-installed by script)

## 🚀 Quick Start

### 1. Get Your Azure Subscription ID

```bash
# List all your subscriptions
az account list --output table

# Or get just the subscription ID of your current subscription
az account show --query id --output tsv
```

### 2. Deploy to Azure

#### Windows (PowerShell):
```powershell
# Navigate to your project
cd c:\Users\prashants\Projects\nse_indices\momentum-etf

# Run deployment script from project root
.\deploy\deploy.ps1 -SubscriptionId "YOUR_SUBSCRIPTION_ID_HERE"

# Optional: Customize resource group name and location
.\deploy\deploy.ps1 -SubscriptionId "YOUR_SUBSCRIPTION_ID_HERE" -ResourceGroupName "my-etf-rg" -Location "West US 2"
```

#### macOS/Linux (Bash):
```bash
# Navigate to your project
cd /path/to/momentum-etf

# Make script executable (Linux/macOS only)
chmod +x deploy/deploy.sh

# Run deployment script from project root
./deploy/deploy.sh -s "YOUR_SUBSCRIPTION_ID_HERE"

# Optional: Customize resource group name and location
./deploy/deploy.sh -s "YOUR_SUBSCRIPTION_ID_HERE" -g "my-etf-rg" -l "West US 2"
```

### 3. What the Script Does

1. ✅ Verifies prerequisites (Azure CLI, Docker)
2. 🔐 Logs you into Azure
3. 📂 Creates resource group
4. 🏗️ Deploys infrastructure using Bicep
5. 🐳 Builds and pushes Docker image
6. 🚀 Deploys container app
7. 📱 Provides your API URL

## 🌐 Using Your Deployed API

After deployment, you'll get a URL like: `https://momentum-etf-dev-app--xxx.eastus.azurecontainerapps.io`

### 🎨 Beautiful Web Interface

Your application includes a stunning web interface! Simply visit your deployment URL to access:
- **Portfolio Analysis**: Interactive forms for optimal portfolio creation
- **Rebalancing Tools**: Upload files or enter data manually for portfolio rebalancing
- **Historical Analysis**: Visualize historical portfolio performance
- **Real-time Results**: All analysis displayed in beautiful, responsive UI

**Web App URL**: `https://your-app-url.com/app`

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/app` | GET | **Beautiful Web Interface** 🎨 |
| `/portfolio` | GET/POST | Current optimal portfolio |
| `/rebalance` | POST | Portfolio rebalancing analysis |
| `/rebalance/upload` | POST | Upload holdings file for rebalancing |
| `/historical` | GET/POST | Historical portfolio analysis |

### Example API Calls

#### 1. Get Current Portfolio
```bash
# GET request
curl -X GET "https://your-app-url.com/portfolio?amount=1000000&size=5"

# POST request
curl -X POST "https://your-app-url.com/portfolio" \
  -H "Content-Type: application/json" \
  -d '{"amount": 1000000, "size": 5}'
```

#### 2. Portfolio Rebalancing
```bash
curl -X POST "https://your-app-url.com/rebalance" \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "NIFTYBEES.NS", "units": 350, "price": 120.50},
      {"symbol": "GOLDBEES.NS", "units": 200, "price": -1}
    ],
    "from_date": "2024-01-01",
    "size": 5
  }'
```

#### 3. Upload Holdings File
```bash
curl -X POST "https://your-app-url.com/rebalance/upload" \
  -F "file=@holdings.json" \
  -F "from_date=2024-01-01" \
  -F "size=5"
```

#### 4. Historical Analysis
```bash
curl -X POST "https://your-app-url.com/historical" \
  -H "Content-Type: application/json" \
  -d '{
    "from_date": "2024-01-01",
    "to_date": "2024-12-31",
    "amount": 1000000,
    "size": 5
  }'
```

## 📱 Cross-Platform Usage

### From Windows:
```powershell
# PowerShell
Invoke-RestMethod -Uri "https://your-app-url.com/portfolio?amount=1000000&size=5" -Method GET

# Command Prompt
curl -X GET "https://your-app-url.com/portfolio?amount=1000000&size=5"
```

### From macOS/Linux:
```bash
# Using curl
curl -X GET "https://your-app-url.com/portfolio?amount=1000000&size=5"

# Using wget
wget -qO- "https://your-app-url.com/portfolio?amount=1000000&size=5"

# Using httpie (if installed)
http GET "https://your-app-url.com/portfolio" amount==1000000 size==5
```

### From Python:
```python
import requests

# Your deployed API URL
API_URL = "https://your-app-url.com"

# Get portfolio
response = requests.get(f"{API_URL}/portfolio", params={
    "amount": 1000000,
    "size": 5
})
print(response.json())

# Rebalance analysis
holdings = [
    {"symbol": "NIFTYBEES.NS", "units": 350, "price": 120.50},
    {"symbol": "GOLDBEES.NS", "units": 200, "price": -1}
]

response = requests.post(f"{API_URL}/rebalance", json={
    "holdings": holdings,
    "from_date": "2024-01-01",
    "size": 5
})
print(response.json())
```

## 🔧 Local Development

### Local Development with uv
```bash
# Install dependencies
uv sync

# Run web server locally
uv run web_server.py

# Your local app will be available at:
# http://localhost:8000/app (Web Interface)
# http://localhost:8000/docs (API Documentation)
```

### Test Local Container
```bash
# Build and run locally (from project root)
docker build -t momentum-etf .
docker run -p 8000:8000 momentum-etf

# Or use docker-compose from deploy folder
cd deploy
docker-compose up
cd ..

# Test local endpoints
curl http://localhost:8000/health
curl http://localhost:8000/portfolio?amount=1000000&size=5

# Access Web Interface
# http://localhost:8000/app
```

## 📊 Monitoring and Management

### View Logs
```bash
# Stream live logs
az containerapp logs show \
  --name momentum-etf-dev-app \
  --resource-group rg-momentum-etf-dev \
  --follow

# Get recent logs
az containerapp logs show \
  --name momentum-etf-dev-app \
  --resource-group rg-momentum-etf-dev \
  --tail 100
```

### Scale Your App
```bash
# Scale to minimum 2 replicas
az containerapp revision set-mode \
  --name momentum-etf-dev-app \
  --resource-group rg-momentum-etf-dev \
  --mode Multiple

# Update scaling rules
az containerapp update \
  --name momentum-etf-dev-app \
  --resource-group rg-momentum-etf-dev \
  --min-replicas 2 \
  --max-replicas 10
```

## 🔄 Updates and Redeployment

To update your application:

1. Make changes to your code
2. Run the deployment script again:
   ```bash
   # Windows
   .\deploy\deploy.ps1 -SubscriptionId "YOUR_SUBSCRIPTION_ID"
   
   # macOS/Linux  
   ./deploy/deploy.sh -s "YOUR_SUBSCRIPTION_ID"
   ```

The script will rebuild and redeploy your container automatically.

## 🗑️ Cleanup

To remove all resources:

```bash
# Delete the entire resource group
az group delete --name rg-momentum-etf-dev --yes --no-wait
```

## 💡 Cost Estimation

With the default configuration:
- **Container Registry (Basic)**: ~$5/month
- **Container Apps**: ~$10-30/month (depending on usage)
- **Log Analytics**: ~$2-5/month
- **Total**: ~$17-40/month

The exact cost depends on:
- Number of API requests
- CPU/memory usage
- Data transfer
- Log retention

## 🔒 Security Considerations

For production use, consider:

1. **Authentication**: Add API keys or OAuth
2. **HTTPS**: Enabled by default in Container Apps
3. **Network Security**: Configure ingress rules
4. **Secrets Management**: Use Azure Key Vault
5. **Container Security**: Regular image updates

## 🆘 Troubleshooting

### Common Issues

1. **Docker build fails**: Ensure Docker Desktop is running
2. **Azure login issues**: Run `az login` manually
3. **Permission errors**: Ensure you have Contributor access to subscription
4. **Container fails to start**: Check logs with `az containerapp logs show`

### Getting Help

1. Check container logs: `az containerapp logs show`
2. Verify health endpoint: `curl https://your-app-url.com/health`
3. Review Azure portal for detailed error messages
4. Check Azure Container Apps documentation

## 📚 Additional Resources

- [Azure Container Apps Documentation](https://docs.microsoft.com/en-us/azure/container-apps/)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
