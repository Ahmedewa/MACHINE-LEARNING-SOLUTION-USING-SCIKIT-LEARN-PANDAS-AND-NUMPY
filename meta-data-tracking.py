               META-DATA-TRACKING




## **1. Track All Metadata**

### **1.1 Why Track Metadata?**
Tracking metadata ensures:
- **Reproducibility**: You can recreate the exact experiment or prediction.
- **Accountability**: Know what dataset, preprocessing steps, and code version were used.
- **Debugging**: Quickly identify issues caused by changes in data or code.

---

### **1.2 Tools for Metadata Tracking**
1. **MLflow**: Logs parameters, metrics, and artifacts.
2. **DVC (Data Version Control)**: Tracks dataset versions.
3. **Git**: Tracks code versions.

---

### **1.3 Code  for Metadata Tracking**

#### **Logging Metadata with MLflow**

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow Experiment
mlflow.start_run()

# Train Model
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Log Metadata
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 5)
mlflow.log_metric("accuracy", model.score(X_test, y_test))
mlflow.sklearn.log_model(model, "random_forest_model")

mlflow.end_run()
```

---

#### **Versioning Datasets with DVC**

1. **Initialize DVC**:
   ```bash
   dvc init
   ```

2. **Add Dataset to DVC**:
   ```bash
   dvc add data/training_data.csv
   ```

3. **Commit Changes**:
   ```bash
   git add data/.gitignore data/training_data.csv.dvc
   git commit -m "Add dataset to DVC"
   ```

4. **Push to Remote Storage**:
   ```bash
   dvc remote add -d myremote s3://your-bucket-name
   dvc push
   ```

---

#### **Track Code Versions Using Git**
1. Initialize a Git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Use Git tags to mark specific versions:
   ```bash
   git tag -a v1.0 -m "Version 1.0 - Initial Model Training"
   git push origin --tags
   ```

---

### **1.4 Best Practices for Metadata Tracking**
1. **Centralize Metadata**:
   - Use tools like MLflow or DVC to consolidate dataset versions, preprocessing steps, and model artifacts.
2. **Automate Logging**:
   - Integrate logging into your training script to ensure all experiments are tracked.
3. **Version Critical Artifacts**:
   - Always version datasets, models, and code.

---

