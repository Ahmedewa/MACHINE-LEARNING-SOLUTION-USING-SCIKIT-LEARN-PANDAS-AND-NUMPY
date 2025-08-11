              HANDLING DATA VERSIONING





## **1. Handling Data Versioning with CI/CD**

### **1.1 Why Data Versioning in CI/CD?**
Data versioning ensures:
1. **Reproducibility**: Track changes to datasets over time.
2. **Collaboration**: Share consistent data versions across teams.
3. **Debugging**: Identify dataset changes that may affect model performance.

### **1.2 Tools for Data Versioning**
1. **DVC (Data Version Control)**: Integrates well with Git to version datasets.
2. **Delta Lake**: For versioned data storage on Apache Spark.
3. **MLflow Artifacts**: Can store dataset versions as artifacts.

---

### **1.3  Using DVC for Data Versioning in CI/CD**

#### **Step 1: Install DVC**
```bash
pip install dvc
```

---

#### **Step 2: Initialize DVC in Your Project**
```bash
dvc init
```

---

#### **Step 3: Version Your Dataset**
1. Add the dataset to DVC:
   ```bash
   dvc add data/training_data.csv
   ```
   This creates:
   - `training_data.csv.dvc`: A metadata file to track the dataset.
   - Updates `.gitignore` to exclude the actual dataset file.

2. Commit the changes to Git:
   ```bash
   git add data/.gitignore data/training_data.csv.dvc
   git commit -m "Version dataset with DVC"
   ```

3. Push the dataset to remote storage:
   ```bash
   dvc remote add -d myremote s3://your-bucket-name
   dvc push
   ```

---

#### **Step 4: Automate Data Versioning in CI/CD**

##### **GitHub Actions Workflow**
```yaml
name: Data Versioning with DVC

on:
  push:
    branches:
      - main

jobs:
  data-versioning:
    runs-on: ubuntu-latest

    steps:
      # Checkout the repository
      - name: Checkout Code
        uses: actions/checkout@v3

      # Install Python and DVC
      - name: Set Up Python and DVC
        uses: actions/setup-python@v4
        with:
          python-version: 3.9
      - run: pip install dvc

      # Download dataset from remote storage
      - name: Pull Dataset
        run: dvc pull

      # Verify Dataset Integrity
      - name: Verify Dataset Integrity
        run: dvc status
```

---

#### **Step 5: Validate Datasets in CI/CD**
After pulling the dataset, use tools like **Great Expectations** to validate its quality:
```yaml
- name: Validate Dataset with Great Expectations
  run: |
    great_expectations checkpoint run my_checkpoint
```

---

### **1.4 Best Practices for Data Versioning in CI/CD**
1. **Use Remote Storage**:
   - Save large datasets in cloud storage (e.g., AWS S3, GCS) instead of Git.
2. **Validate Data**:
   - Use validation tools like **Great Expectations** in your CI/CD pipeline.
3. **Automate Versioning**:
   - Automate dataset updates with CI/CD triggers (e.g., `cron` jobs).

---

