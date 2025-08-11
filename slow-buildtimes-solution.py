              SLOW BUILD TIMES


#### **Slow Build Times**
- **Problem**: Repeated installation of dependencies during each build.
- **Solution**:
  - Cache dependencies by copying only `requirements.txt` first.
  - code:
    ```dockerfile
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    ```

---

