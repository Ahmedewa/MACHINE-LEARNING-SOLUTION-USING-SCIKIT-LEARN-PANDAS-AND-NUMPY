   AUTOMATE TESTING


 2. **Automate Tests**:
   - GitHub Actions Workflow:
     ```yaml
     name: CI Pipeline

     on: push
     jobs:
       test:
         runs-on: ubuntu-latest
         steps:
           - name: Checkout Code
             uses: actions/checkout@v3

           - name: Install Dependencies
             run: pip install -r requirements.txt

           - name: Run Tests
             run: pytest
     ```

