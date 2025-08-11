              ENVIROMENT MANAGEMENT - DEPENDENCIES


### **Environment Management**
1. **Isolate Dependencies**:
   - Use virtual environments (`venv`) or Docker.
   - Code:
     ```bash
     python3 -m venv myenv
     source myenv/bin/activate
     pip install -r requirements.txt
     ```

2. **Pin Dependency Versions**:
   - Lock versions in `requirements.txt` or `environment.yml`.
   -  `requirements.txt`:
     ```plaintext
     numpy==1.21.6
     pandas==1.3.5
     ```

---

## **4. Resources**

1. **Docker for ML**:
   - [Docker Best Practices](https://docs.docker.com/develop/)
   - [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

2. **Great Expectations**:
   - [Great Expectations Docs](https://docs.greatexpectations.io/)

3. **MLflow**:
   - [MLflow Docs](https://mlflow.org/)

4. **Weights & Biases**:
   - [Weights & Biases Docs](https://docs.wandb.ai/)

---
