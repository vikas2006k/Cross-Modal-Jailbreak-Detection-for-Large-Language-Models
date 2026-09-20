# Docker & Containerized Deployment Guide

This guide details how to build, run, and scale the Cross-Modal Jailbreak Detection (CMJD) system using Docker and Docker Compose.

---

## 1. Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) (v24.0+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.20+)
- *(Optional for GPU acceleration)*: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## 2. 1-Command Startup with Docker Compose

From the root of the repository:

```bash
# Build and start all services in the background
docker compose -f docker/docker-compose.yml up --build -d
```

### Inspect Running Containers
```bash
docker compose -f docker/docker-compose.yml ps
```
*Expected output:*
```text
NAME            IMAGE                  COMMAND                  SERVICE         STATUS                    PORTS
cmjd-backend    docker-cmjd-backend    "python -m uvicorn b…"   cmjd-backend    Up 45s (healthy)          0.0.0.0:8001->8001/tcp
cmjd-frontend   docker-cmjd-frontend   "/docker-entrypoint.…"   cmjd-frontend   Up 20s                    0.0.0.0:5173->80/tcp
```

### Access Applications
- **Web Dashboard**: `http://localhost:5173`
- **FastAPI Documentation**: `http://localhost:8001/docs`
- **Healthcheck**: `http://localhost:8001/health`

### Stop All Containers
```bash
docker compose -f docker/docker-compose.yml down
```

---

## 3. Individual Container Builds

### Building the Backend Container
```bash
docker build -t cmjd-backend:v1.0 -f docker/Dockerfile.backend .
```

Run standalone:
```bash
docker run -d \
  --name cmjd-backend \
  -p 8001:8001 \
  -e DEVICE=cpu \
  -v $(pwd)/models:/app/models:ro \
  cmjd-backend:v1.0
```

### Building the Frontend Container (Multi-stage NGINX)
```bash
docker build -t cmjd-frontend:v1.0 -f docker/Dockerfile.frontend .
```

Run standalone:
```bash
docker run -d \
  --name cmjd-frontend \
  -p 5173:80 \
  cmjd-frontend:v1.0
```

---

## 4. Enabling NVIDIA GPU Acceleration

To run EasyOCR, DistilBERT, and CLIP with GPU acceleration inside Docker:

1. Verify NVIDIA Container Toolkit is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
2. Update `docker/docker-compose.yml` under `cmjd-backend`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   environment:
     - DEVICE=cuda
   ```
3. Restart:
   ```bash
   docker compose -f docker/docker-compose.yml up --build -d
   ```

---

## 5. Kubernetes Production Deployment (Snippet)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cmjd-backend
  namespace: ai-security
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cmjd-backend
  template:
    metadata:
      labels:
        app: cmjd-backend
    spec:
      containers:
      - name: backend
        image: your-registry.com/cmjd-backend:v1.0
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 20
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
```
