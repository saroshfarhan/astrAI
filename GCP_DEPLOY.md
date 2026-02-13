# 🚀 Deploy AstraAI to Google Cloud (No Docker)

## Option: Single VM with All Services

**Cost:** ~$50/month → **$300 credits last 6 months**

---

## 📋 Step-by-Step Deployment

### **1. Create VM in Google Cloud Console**

```bash
# Open Google Cloud Shell (browser-based terminal)
# https://console.cloud.google.com/

# Create VM
gcloud compute instances create astra-ai-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-2 \
  --boot-disk-size=30GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=http-server,https-server
```

---

### **2. SSH into VM**

```bash
gcloud compute ssh astra-ai-vm --zone=us-central1-a
```

---

### **3. Setup Environment**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install tmux for managing multiple terminals
sudo apt install tmux -y

# Clone your repo
git clone https://github.com/YOUR_USERNAME/astraAI.git
cd astraAI

# Install dependencies
uv sync
```

---

### **4. Configure Environment**

```bash
# Create .env file
nano .env
```

**Add the following:**
```env
GOOGLE_API_KEY=your-gemini-api-key-here
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=ad-astrAI
SPECTRAL_SERVICE_URL=http://localhost:8001
PYTHONIOENCODING=utf-8
```

**Save:** `Ctrl+X` → `Y` → `Enter`

---

### **5. Open Firewall Ports**

```bash
# Exit SSH (Ctrl+D) and run in Cloud Shell:
gcloud compute firewall-rules create allow-mlflow \
  --allow tcp:5000 \
  --target-tags http-server

gcloud compute firewall-rules create allow-spectral \
  --allow tcp:8001 \
  --target-tags http-server

gcloud compute firewall-rules create allow-streamlit \
  --allow tcp:8501 \
  --target-tags http-server

# SSH back in
gcloud compute ssh astra-ai-vm --zone=us-central1-a
```

---

### **6. Start All Services (Using tmux)**

```bash
cd astraAI

# Start tmux session
tmux new -s astra

# Window 1: MLflow
uv run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db

# Press: Ctrl+B then C (creates new window)

# Window 2: Spectral Service
cd "Spectral Service"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001

# Press: Ctrl+B then C (creates new window)

# Window 3: Streamlit
cd ~/astraAI
uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

**Tmux Navigation:**
- `Ctrl+B` then `0` = Window 0 (MLflow)
- `Ctrl+B` then `1` = Window 1 (Spectral Service)
- `Ctrl+B` then `2` = Window 2 (Streamlit)
- `Ctrl+B` then `D` = Detach (services keep running)

---

### **7. Get Your Public IP**

```bash
# Exit tmux: Ctrl+B then D
# In Cloud Shell:
gcloud compute instances describe astra-ai-vm \
  --zone=us-central1-a \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

---

### **8. Access Your App**

**Your Live URLs:**
- **Frontend:** `http://YOUR_VM_IP:8501`
- **Spectral Service API:** `http://YOUR_VM_IP:8001/docs`
- **MLflow UI:** `http://YOUR_VM_IP:5000`

---

## 🔄 Managing Services

### **Reconnect to tmux session:**
```bash
gcloud compute ssh astra-ai-vm --zone=us-central1-a
tmux attach -t astra
```

### **Stop all services:**
```bash
# In each tmux window: Ctrl+C
# Or kill entire session: tmux kill-session -t astra
```

### **Restart a service:**
```bash
# Navigate to the window (Ctrl+B then 0/1/2)
# Press Ctrl+C to stop
# Run the command again
```

---

## 🛑 Stop VM (Save Costs)

```bash
# Stop VM (doesn't lose data)
gcloud compute instances stop astra-ai-vm --zone=us-central1-a

# Start VM again
gcloud compute instances start astra-ai-vm --zone=us-central1-a
```

**When stopped:** $0/month
**When running:** ~$50/month

---

## 💰 Cost Optimization

### **Option 1: E2-medium (Cheaper)**
```bash
# $25/month → Credits last 12 months
--machine-type=e2-medium
```

### **Option 2: Preemptible VM**
```bash
# $12/month → Credits last 25 months (can be terminated)
--preemptible
```

### **Option 3: Spot VM**
```bash
# $10/month → Credits last 30 months (cheapest, can be terminated)
--provisioning-model=SPOT
```

---

## 📊 Monitor Usage

```bash
# SSH into VM
gcloud compute ssh astra-ai-vm --zone=us-central1-a

# Check CPU/Memory
htop

# Check disk usage
df -h

# View service logs in tmux
tmux attach -t astra
# Navigate with Ctrl+B then 0/1/2
```

---

## 🔒 Security (Optional)

### **Add Authentication:**

**For Streamlit:**
```bash
# Edit app.py to add password protection
# Or use nginx reverse proxy with basic auth
```

**For MLflow:**
```bash
# Only accessible from your IP
gcloud compute firewall-rules update allow-mlflow \
  --source-ranges=YOUR_IP_ADDRESS/32
```

---

## ✅ Deployment Checklist

- [ ] Created VM on GCP
- [ ] SSH'd into VM
- [ ] Installed UV and dependencies
- [ ] Configured `.env` file
- [ ] Opened firewall ports
- [ ] Started all 3 services in tmux
- [ ] Got public IP address
- [ ] Tested frontend at `http://IP:8501`
- [ ] Tested Spectral Analysis (FITS upload)
- [ ] Tested Image Analysis (PKL upload)
- [ ] Set up billing alerts ($50, $100, $150)

---

## 🆘 Troubleshooting

### **Service won't start:**
```bash
# Check logs in tmux window
# Or run manually to see error:
cd ~/astraAI
uv run streamlit run app.py
```

### **Can't access from browser:**
```bash
# Check firewall rules
gcloud compute firewall-rules list

# Check VM is running
gcloud compute instances list
```

### **Out of memory:**
```bash
# Upgrade to larger machine
gcloud compute instances stop astra-ai-vm --zone=us-central1-a
gcloud compute instances set-machine-type astra-ai-vm \
  --machine-type=n1-standard-4 \
  --zone=us-central1-a
gcloud compute instances start astra-ai-vm --zone=us-central1-a
```

---

## 🎯 Next Steps

1. **Test locally first** (ensure everything works)
2. **Push code to GitHub**
3. **Follow this guide to deploy**
4. **Share the `http://IP:8501` URL** with your team!

**Your app will be live on the internet!** 🌐
