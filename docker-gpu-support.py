            ** DOKER-GPU SUPPORT

#### ** GPU Support**
- **Problem**: Docker containers don't have native access to GPUs.
- **Solution**:
  - Use NVIDIA's Docker runtime.
  - Install the necessary drivers.
  - Example:
    ```bash
    docker run --gpus all -it --rm nvidia/cuda:11.8-base nvidia-smi
    ```
  - Add GPU support in the `Dockerfile`:
    ```dockerfile
    FROM nvidia/cuda:11.8-base
    RUN apt-get update && apt-get install -y python3 python3-pip
    ```

