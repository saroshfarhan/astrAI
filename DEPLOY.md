# 🚀 Deploy AstraAI to Google Cloud

## Simple 5-Step Deployment (using Cloud Shell)

### Prerequisites
- ✅ GitHub account (push your code there)
- ✅ Google Cloud account with $300 credits
- ✅ 2 Docker files already created (Dockerfile.streamlit & Spectral Service/Dockerfile)

---

## Step 1: Push Code to GitHub

```bash
# In your local terminal
cd "c:\Users\Ujwal Mojidra\Desktop\AAI\project\astraAI"

# Initialize git (if not already)
git init
git add .
git commit -m "Ready for GCP deployment"

# Create repo on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/astraAI.git
git branch -M main
git push -u origin main
```

---

## Step 2: Open Google Cloud Console

1. Go to: https://console.cloud.google.com/
2. Click **"Activate Cloud Shell"** button (top right, terminal icon)
3. Wait for Cloud Shell to load (takes ~30 seconds)

**You now have a full Linux terminal in your browser!**

---

## Step 3: Clone Your Repo in Cloud Shell

```bash
# In Cloud Shell terminal
git clone https://github.com/YOUR_USERNAME/astraAI.git
cd astraAI
```

---

## Step 4: Deploy Spectral Service

```bash
# Build and deploy in one command
gcloud run deploy spectral-service \
  --source ./Spectral\ Service \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 8001
```

**Wait ~5 minutes for build and deployment**

You'll get output like:
```
Service [spectral-service] deployed.
Service URL: https://spectral-service-abc123-uc.a.run.app
```

**Copy this URL!** You'll need it for the next step.

---

## Step 5: Deploy Streamlit Frontend

```bash
# Replace with YOUR Spectral Service URL from Step 4
SPECTRAL_URL="https://spectral-service-abc123-uc.a.run.app"
GEMINI_KEY="your-google-gemini-api-key"

# Deploy frontend
gcloud run deploy astra-ai-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --port 8501 \
  --set-env-vars="SPECTRAL_SERVICE_URL=$SPECTRAL_URL,GOOGLE_API_KEY=$GEMINI_KEY"
```

**Wait ~5 minutes for build and deployment**

You'll get:
```
Service [astra-ai-frontend] deployed.
Service URL: https://astra-ai-frontend-xyz789-uc.a.run.app
```

---

## 🎉 Done! Your App is Live

**Open the Frontend URL in your browser:**
```
https://astra-ai-frontend-xyz789-uc.a.run.app
```

---

## 💰 Cost Estimate

| Service | Cost/Month |
|---------|------------|
| Spectral Service (2GB RAM) | $10-20 |
| Frontend (1GB RAM) | $5-15 |
| **Total** | **$15-35** |

**Your $300 credits will last 8-20 months!**

---

## 🔧 Common Issues

### Issue 1: "Build failed - requirements.txt not found"

**Solution:** Make sure both services have their own `requirements.txt`

```bash
# Check if files exist
ls requirements.txt                    # For frontend
ls "Spectral Service/requirements.txt" # For backend
```

### Issue 2: "Port 8501 is not exposed"

**Solution:** Update your Dockerfile.streamlit to include:
```dockerfile
EXPOSE 8501
ENV PORT=8501
```

### Issue 3: "Services can't communicate"

**Solution:** Check Spectral Service URL is correct
```bash
# Get the URL again
gcloud run services describe spectral-service \
  --region us-central1 \
  --format="value(status.url)"

# Update frontend with correct URL
gcloud run services update astra-ai-frontend \
  --region us-central1 \
  --set-env-vars="SPECTRAL_SERVICE_URL=https://spectral-service-abc.run.app"
```

---

## 🔄 Update Your Deployment

When you make code changes:

```bash
# 1. Push to GitHub
git add .
git commit -m "Updated code"
git push

# 2. In Cloud Shell, pull and redeploy
cd astraAI
git pull

# Redeploy (same commands as Step 4 & 5)
gcloud run deploy spectral-service --source ./Spectral\ Service --region us-central1
gcloud run deploy astra-ai-frontend --source . --region us-central1
```

---

## 📊 Monitor Your App

### View Logs
```bash
# Frontend logs
gcloud run services logs read astra-ai-frontend --region us-central1 --limit 50

# Backend logs
gcloud run services logs read spectral-service --region us-central1 --limit 50
```

### Check Costs
```bash
# View current spending
gcloud billing accounts list
```

Or visit: https://console.cloud.google.com/billing

---

## 🛑 Stop Services (Save Money)

### Temporary Pause (scales to 0 when idle)
```bash
# Already enabled by default! Cloud Run scales to 0 automatically.
# You only pay when someone uses the app.
```

### Permanent Delete
```bash
# Delete both services
gcloud run services delete astra-ai-frontend --region us-central1
gcloud run services delete spectral-service --region us-central1
```

---

## 💡 Pro Tips

### 1. Set Budget Alerts
Go to: https://console.cloud.google.com/billing/budgets
- Set alert at $50 (warning)
- Set alert at $100 (critical)

### 2. Use Cloud Shell Editor
```bash
# Open built-in VS Code editor
cloudshell edit astraAI
```

### 3. Check Service Status
```bash
# List all services
gcloud run services list --region us-central1
```

---

## ✅ Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Opened Cloud Shell in GCP
- [ ] Cloned repo in Cloud Shell
- [ ] Deployed Spectral Service (got URL)
- [ ] Deployed Frontend with correct env vars
- [ ] Tested live URL in browser
- [ ] Set budget alerts
- [ ] Shared URL with team! 🎉

---

## 📝 Key Files You Have

```
astraAI/
├── Dockerfile.streamlit          # Frontend container
├── Spectral Service/
│   └── Dockerfile                # Backend container
├── .dockerignore                 # Optimize image size
├── docker-compose.yml            # Local testing (optional)
└── DEPLOY.md                     # This guide
```

That's it! No complex scripts needed - Cloud Run handles everything else.
