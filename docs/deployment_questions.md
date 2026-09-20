# Deployment, DevOps & Production Architecture Viva Questions: 25 In-Depth Questions & Answers

**Project**: Cross-Modal Jailbreak Detection for Large Language Models (CMJD)  
**Focus**: Microservices, Docker containerization, Kubernetes orchestration, CI/CD, performance tuning, and SAP BTP integration.

---

### 1. What is the production architecture of CMJD when deployed in an enterprise environment?
**Answer**:  
In an enterprise cloud (AWS, Azure, GCP, or SAP BTP):
1. **API Gateway / Ingress**: An NGINX or Traefik reverse proxy handles TLS termination, rate-limiting, and authentication (OAuth2 / JWT).
2. **Frontend Layer**: React 19 single-page application compiled into static HTML/JS/CSS, served via high-performance NGINX or a CDN (Cloudflare / Akamai).
3. **Backend Microservice**: FastAPI running under Uvicorn with multiple asynchronous workers inside a Docker container, scaled horizontally via Kubernetes (HPA).
4. **Hardware Acceleration**: Backend pods are bound to NVIDIA T4 or L4 GPU nodes for sub-100ms EasyOCR and CLIP tensor execution.
5. **Database & Audit Layer**: PostgreSQL or SAP HANA Cloud stores structured JSON telemetry, SHAP token vectors, and timestamps.

---

### 2. How is the Dockerfile for the backend structured to minimize image size and build times?
**Answer**:  
We utilize a multi-stage, layer-cached Docker build (`docker/Dockerfile.backend`):
1. **Base Stage**: `python:3.10-slim` to minimize OS vulnerability footprint.
2. **System Dependencies**: Installs only essential shared libraries for OpenCV and PyTorch (`libgl1-mesa-glx`, `libglib2.0-0`).
3. **Layer Caching**: Copies `requirements.txt` and runs `pip install --no-cache-dir` *before* copying the application code. This ensures heavy dependencies (PyTorch, Transformers, EasyOCR) are cached and not reinstalled on code edits.
4. **Security**: Runs as a non-root user (`appuser`, UID 10001) to adhere to CIS Docker benchmarks.

---

### 3. How is the Dockerfile for the frontend structured?
**Answer**:  
We utilize a two-stage Docker build (`docker/Dockerfile.frontend`):
1. **Build Stage (`node:20-alpine`)**:
   - Copies `package.json` and runs `npm ci` for deterministic dependency resolution.
   - Copies TypeScript source files and runs `npm run build` to emit minified production assets to `/dist`.
2. **Runtime Stage (`nginx:alpine-slim`)**:
   - Copies only the compiled `/dist` directory into `/usr/share/nginx/html`.
   - Discards all Node.js runtimes, devDependencies, and build tools.
   - Result: A lightweight production container under 25 MB with zero Node.js runtime attack surface.

---

### 4. Explain how `docker-compose.yml` orchestrates the system.
**Answer**:  
`docker/docker-compose.yml` defines a multi-container local and staging environment:
- **`cmjd-backend` service**:
  - Builds from `docker/Dockerfile.backend`.
  - Exposes port 8001.
  - Mounts `./models` as a read-only volume (`:ro`) to prevent runtime model tampering.
  - Configures environment variables (`DEVICE=cuda`, `WORKERS=4`).
  - Includes a healthcheck hitting `/health`.
- **`cmjd-frontend` service**:
  - Builds from `docker/Dockerfile.frontend`.
  - Exposes port 5173 (or 80).
  - `depends_on` the backend healthcheck.
- **Internal Network**: Isolated bridge network (`cmjd-net`) facilitating low-latency inter-service routing.

---

### 5. What are the environment variables used across the project?
**Answer**:  
- `VITE_API_URL`: Base URL for the FastAPI backend (e.g. `http://127.0.0.1:8001` or `https://api.guardrail.enterprise.com`).
- `DEVICE`: Execution target for PyTorch tensors (`cuda` or `cpu`).
- `API_PORT`: Port for Uvicorn server (default `8001`).
- `API_HOST`: Host binding (default `0.0.0.0` in Docker, `127.0.0.1` locally).
- `MODEL_PATH_TEXT`: Absolute path to DistilBERT checkpoint directory.
- `MODEL_PATH_VISION`: Absolute path to CLIP MLP checkpoint file.
- `LOG_LEVEL`: Logging verbosity (`INFO`, `DEBUG`, `WARNING`).

---

### 6. How do you implement automated health checks in production?
**Answer**:  
The backend exposes `GET /health`, returning HTTP 200:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "timestamp": "2026-09-20T20:00:00Z"
}
```
In Docker / Kubernetes:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 15
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 5
  periodSeconds: 5
```
If models fail to load or VRAM runs out, the probe fails and Kubernetes automatically restarts the pod.

---

### 7. How does CMJD scale to handle 10,000 requests per second in an enterprise AI gateway?
**Answer**:  
1. **Horizontal Pod Autoscaling (HPA)**: Kubernetes HPA scales backend pods dynamically based on CPU utilization and GPU duty cycle.
2. **Model Serving Optimization**: Models can be exported to **ONNX Runtime** or **NVIDIA Triton Inference Server** with dynamic batching.
3. **Decoupled Asynchronous Processing**: Requests are queued via Kafka or RabbitMQ, evaluated in parallel by worker pools, and results published back to client webhooks.
4. **Caching Layer**: Redis caches hashes of previously scanned images and prompts. If an identical document or image is uploaded, the cached verdict is returned in $< 2$ms.

---

### 8. How do you secure the model checkpoints against tampering or weight extraction?
**Answer**:  
1. **Read-Only Filesystem**: Mount the `models/` directory into the Docker container with read-only flags (`ro`).
2. **SafeTensors Format**: DistilBERT uses HuggingFace `safetensors` rather than legacy Python `pickle`, eliminating arbitrary code execution vulnerabilities.
3. **Cryptographic Checksums**: The backend startup script calculates SHA-256 hashes of weights against an environment manifest, refusing to boot if signatures differ.

---

### 9. What is the difference between running with `uvicorn ... --reload` and production Uvicorn?
**Answer**:  
- **Development (`--reload`)**: Uses a single worker process with a file watcher that monitors source code changes and reloads the server. It has high CPU overhead, memory leaks over time, and poor throughput.
- **Production**: Run without `--reload`, using Gunicorn as a process manager managing multiple Uvicorn worker instances:
  ```bash
  gunicorn backend.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001 --timeout 120
  ```
  This provides automatic worker recovery, shared memory management, and maximal throughput across multi-core systems.

---

### 10. How do you monitor GPU memory fragmentation in PyTorch during continuous production inference?
**Answer**:  
1. **`torch.no_grad()` Context**: All inference methods wrap tensor execution inside `with torch.no_grad():`, which prevents PyTorch from storing activation graphs for backpropagation.
2. **`torch.cuda.empty_cache()`**: Periodically invoked to release cached unallocated memory blocks back to the OS.
3. **Tensor Eviction**: Intermediate image tensors and NumPy arrays are explicitly deleted or allowed to fall out of Python scope.

---

### 11. How do you handle graceful shutdown in FastAPI?
**Answer**:  
FastAPI uses the `lifespan` async context manager:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models into GPU memory
    load_models()
    yield
    # Shutdown: Clean up connections, flush audit logs, free GPU VRAM
    torch.cuda.empty_cache()
```
When Docker sends `SIGTERM`, Uvicorn finishes in-flight requests, executes the shutdown block, and exits cleanly within a 30-second grace period.

---

### 12. How do you set up HTTPS/TLS in production?
**Answer**:  
We follow standard cloud architecture: TLS is terminated at the reverse proxy (NGINX or AWS ALB) using Let's Encrypt or enterprise PKI certificates. The internal communication between NGINX and the FastAPI Docker container runs over an isolated virtual bridge network, offloading crypto operations from Python.

---

### 13. How does the frontend handle routing when deployed to NGINX?
**Answer**:  
React Router is a client-side Single Page Application (SPA). When a user visits `http://domain.com/text-scanner` directly, NGINX must not look for a physical file named `text-scanner`.
In `docker/nginx.conf`:
```nginx
location / {
    root /usr/share/nginx/html;
    index index.html index.htm;
    try_files $uri $uri/ /index.html;
}
```
The `try_files` directive ensures that all route paths are redirected to `index.html`, allowing React Router to mount the correct view.

---

### 14. What logging framework is used, and how are logs ingested into enterprise SIEM systems?
**Answer**:  
We use Python's structured `logging` module emitting JSON-formatted log lines:
```json
{"timestamp": "2026-09-20T20:10:02Z", "level": "WARNING", "event": "JAILBREAK_DETECTED", "risk_score": 99.8, "client_ip": "10.0.1.45", "latency_ms": 13.4}
```
Standard output streams (`stdout`) are collected by Docker logging drivers and forwarded to enterprise SIEM platforms (Splunk, Datadog, or SAP Cloud Logging) for automated alerting.

---

### 15. How do you test the complete pipeline in a CI/CD environment (GitHub Actions / GitLab CI)?
**Answer**:  
Our CI/CD pipeline runs on every push:
1. **Linting & Typing**: `oxlint` and `tsc -b` on the frontend; `flake8` and `mypy` on the backend.
2. **Unit Tests**: `pytest tests/` runs isolated tests on the fusion logic, entropy calculations, and regex rules without loading full GPU weights.
3. **Integration Build**: `npm run build` validates the React production bundle.
4. **Container Scan**: Trivy scans Docker images for CVE vulnerabilities before pushing to the container registry.

---

### 16. How do you handle file upload size limits in FastAPI to prevent Denial of Service (DoS)?
**Answer**:  
In `backend/config.py`, we enforce a `MAX_UPLOAD_SIZE = 10 * 1024 * 1024` (10 MB).
If a client uploads a file exceeding 10 MB, a middleware or endpoint validation immediately raises an HTTP 413 (Payload Too Large) exception before reading the stream into memory, preventing memory exhaustion attacks.

---

### 17. How do you optimize EasyOCR download speeds in Docker?
**Answer**:  
By default, EasyOCR downloads CRAFT and CRNN model weights from GitHub on first invocation.
In our Docker build (`Dockerfile.backend`), we pre-download the models during image creation:
```dockerfile
RUN python -c "import easyocr; easyocr.Reader(['en'])"
```
The weights are stored in `~/.EasyOCR/model/` inside the container image, ensuring the container boots instantly in air-gapped or restricted production networks.

---

### 18. What is the difference between `@tailwindcss/postcss` and traditional Tailwind CLI?
**Answer**:  
Tailwind CSS v4 introduced a high-performance Rust-based core engine packaged as `@tailwindcss/postcss`. Instead of requiring a heavy `tailwind.config.js` file with complex scanning regexes, Tailwind v4 uses `@import "tailwindcss";` in `index.css` and automatically detects CSS utility classes across all source files, reducing Vite build times from 3.5s down to 0.7s.

---

### 19. How do you troubleshoot a `504 Gateway Timeout` error on `/cross-modal-predict`?
**Answer**:  
1. **Check NGINX / Proxy Timeout**: Increase proxy read timeout (`proxy_read_timeout 120s;`) to accommodate high-resolution image processing.
2. **Check Device Utilization**: Verify via `nvidia-smi` whether the GPU is throttled or running out of memory, falling back to slow CPU threads.
3. **Image Resolution Scaling**: If the client uploaded an 8K image (30MB), downsample the image resolution to max dimension 1024px before passing to EasyOCR.

---

### 20. How do you troubleshoot `CORS policy: No 'Access-Control-Allow-Origin' header`?
**Answer**:  
1. Check that the backend has `CORSMiddleware` active in `backend/app.py`.
2. Verify that the client's origin (e.g. `http://127.0.0.1:5173`) is included in `allow_origins`.
3. If an unhandled exception occurred in a FastAPI endpoint before the response returned, the default 500 error handler might bypass CORS. Wrap the route in a try-catch block returning a standard JSON error response.

---

### 21. How do you configure Vite dev server proxy to avoid CORS entirely during development?
**Answer**:  
In `frontend/vite.config.ts`:
```typescript
server: {
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8001',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```
This allows the frontend to query `/api/predict` as a relative path, and Vite proxies the request to port 8001 server-side.

---

### 22. What is the impact of running the system on an air-gapped network?
**Answer**:  
CMJD is **100% self-contained**:
- All model checkpoints (`models/text_classifier`, `vision_detector/models`) are stored locally on disk.
- EasyOCR weights are cached locally.
- Zero outbound calls are made to external APIs (OpenAI, HuggingFace, Google).
- The system runs flawlessly in restricted, air-gapped enterprise environments (defense, banking, healthcare).

---

### 23. How do you implement automated rollbacks in Kubernetes if a new model version degrades accuracy?
**Answer**:  
Using Kubernetes Deployment rollouts:
```bash
kubectl rollout undo deployment/cmjd-backend
```
During deployment, Canary releases (e.g. via Istio or Argo Rollouts) route 5% of traffic to the new model container while evaluating shadow benchmark metrics before switching 100% of production traffic.

---

### 24. What are the minimum system specifications required to run the full stack locally?
**Answer**:  
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS (Apple Silicon / Intel).
- **CPU**: 4 cores (Intel Core i5 / AMD Ryzen 5 or higher).
- **RAM**: 8 GB minimum (16 GB recommended).
- **Disk**: 5 GB free disk space.
- **GPU (Optional)**: NVIDIA GPU with 4GB+ VRAM and CUDA 11.8+ for real-time acceleration.

---

### 25. How do you package the final project for distribution to judges and clients?
**Answer**:  
We structure the distribution via `project_release/`:
- Bundles complete documentation, IEEE paper, and SAP slide deck.
- Provides 1-command startup scripts (`start_all.bat` for Windows and `start_all.sh` for Linux).
- Includes pre-built Docker compose configurations for immediate deployment on any workstation.
