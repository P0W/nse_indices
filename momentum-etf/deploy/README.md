# Deployment Scripts and Infrastructure

This folder contains all deployment-related files for the ETF Momentum Strategy web application.

## 🚀 Quick Deploy

**Important**: Run these commands from the project root directory, not from inside the deploy folder.

### Windows
```powershell
# From project root
.\deploy\deploy.ps1 -SubscriptionId "YOUR_SUBSCRIPTION_ID"
```

### Linux/macOS
```bash
# From project root
./deploy/deploy.sh -s "YOUR_SUBSCRIPTION_ID"
```

## 📁 Files

- **deploy.ps1** / **deploy.sh**: Main deployment scripts
- **docker-compose.yml**: Local container testing
- **status.ps1**: Check deployment status
- **cleanup.ps1** / **cleanup.sh**: Resource cleanup
- **optimize-build-cache.ps1** / **optimize-build-cache.sh**: Build optimization
- **infra/**: Azure infrastructure as code (Bicep templates)

## 🧪 Local Testing

```bash
# Test with docker-compose (from project root)
cd deploy
docker-compose up
cd ..

# Access at http://localhost:8000/app
```

For detailed instructions, see the main [DEPLOYMENT.md](../DEPLOYMENT.md) file.
