The best way to install Miniconda on Ubuntu 22.04 is by using the official installer script provided by Anaconda. This method ensures you get the latest version and that it's correctly configured for your user's shell.

Here is a step-by-step guide.

-----

### 1\. Update and Install Prerequisites

First, update your system's package list and ensure you have `wget` (or `curl`) to download the installer.

```bash
sudo apt update
sudo apt install wget
```

-----

### 2\. Download the Miniconda Installer

Next, download the latest Linux 64-bit installer script. We'll save it in the `/tmp` directory, as it's only needed for the installation.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
```
For ARM-based systems, use this command
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
```
-----

### 3\. (Optional but Recommended) Verify the Installer

For security, you can verify the integrity of the downloaded script using its SHA256 hash.

1.  Generate the hash of your downloaded file:
    ```bash
    sha256sum /tmp/miniconda.sh
    ```
2.  Compare the output to the official hash. You can find the list of hashes on the [Anaconda repository page](https://www.google.com/search?q=https://repo.anaconda.com/miniconda/). If the hashes match, the file is authentic and uncorrupted.

-----

### 4\. Run the Installer Script

Now, execute the installer script using `bash`:

```bash
bash /tmp/miniconda.sh
```

You will be guided through a few prompts:

  * **Review License:** Press **Enter** to review the license agreement. You can press the **Spacebar** to scroll through it.
  * **Accept License:** At the end, type `yes` and press **Enter** to accept the terms.
  * **Confirm Location:** You'll be asked to confirm the installation location. The default (`/home/YOUR-USER/miniconda3`) is strongly recommended. Press **Enter** to accept it.
  * **Initialize Miniconda:** The installer will ask, "Do you wish the installer to initialize Miniconda3 by running conda init?" This is the most important step. Type `yes` and press **Enter**. This will modify your `.bashrc` file so `conda` is available in your terminal.

-----

### 5\. Activate and Verify the Installation

The installer made changes to your `.bashrc` file, but they aren't active in your current terminal session. You must reload your shell.

Run this command to apply the changes:

```bash
source ~/.bashrc
```

> **Note:** Alternatively, you can simply close your current terminal and open a new one.

You'll know it worked because your command prompt will now be prepended with `(base)`.

To verify the installation, check the `conda` version:

```bash
conda --version
```

You can also see a list of default installed packages:

```bash
conda list
```

### A Quick Note on the `(base)` Environment

The `(base)` text in your prompt indicates you are in the default conda environment. It's generally good practice to create separate environments for your projects rather than installing packages directly into `base`.

If you find the `(base)` prompt annoying, you can disable it from activating automatically with this command:

```bash
conda config --set auto_activate_base false
```

After running that, the `(base)` prompt will disappear when you open a new terminal. You can then activate it manually anytime with `conda activate base`.

## Installation
You got it. You are completely right, and my previous guide was incorrect for your new hardware. Running this project on a Blackwell GPU requires a *full modernization* of every single component.

Here is the correct, revised guide. We will be building an environment based on the latest software that is confirmed to be compatible with the Blackwell architecture.

-----

### 🗺️ Project Modernization Plan: `triton_amex` for Blackwell

The original project uses CUDA 11.8, RAPIDS 23.04, and an old Triton container. **None of these will work** on a Blackwell GPU.

We will replace them with:

  * **GPU Driver:** The latest NVIDIA driver for Blackwell.
  * **CUDA:** CUDA 13.0 (or newer) via Conda.
  * **RAPIDS:** RAPIDS 25.10 (or newer), which supports modern CUDA.
  * **PyTorch:** PyTorch 2.7 (or newer), the first to support Blackwell.
  * **Triton Server:** A 2025 (or newer) container, which is certified for Blackwell.

-----

### Step 1: Host System Prerequisites

Before you touch the project code, your host machine must be ready.

1.  **Install Blackwell Driver:** This is the most important step. Go to the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page and get the latest **Game Ready** or **Studio Driver** for your specific Blackwell GPU.
2.  **Install Conda:** Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or the full Anaconda.
3.  **Install Docker:** The project's inference step relies on Docker.
      * Install [Docker Engine](https://docs.docker.com/engine/install/).
      * After Docker is working, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). This is what allows Docker containers to access your Blackwell GPU.

-----

### Step 2: Clone the Project

In your terminal, clone the repository:

```bash
git clone https://github.com/davrollins/triton_amex_blackwell.git
cd triton_amex
```

-----

### Step 3: Create a Modern Conda Environment

We will completely ignore the `conda create` command in the old `README.md`. It is obsolete.

1.  **Create the Environment:** Run this command to create a new environment named `rapids-modern`. This installs the latest RAPIDS, Python 3.13, and the CUDA 12.8 toolkit, all of which are compatible with Blackwell.
    ```bash
    conda create -n rapids-modern -c rapidsai -c conda-forge -c nvidia \
        rapids=25.10 python=3.13 cuda-version=12.8
    ```
2.  **Activate the Environment:**
    ```bash
    conda activate rapids-modern
    ```

-----

### Step 4: Install and Verify Python Dependencies

Now, we'll install the Python libraries *inside* this new environment.

1.  **Install PyTorch for Blackwell (Correctly):**
    As you noted, we need PyTorch 2.7+ built for CUDA 12.8+. The command below installs the correct version using `pip`.

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install lightning
    ```

      * **Verify PyTorch:** Run `python` and enter these commands. You should see `True` and your Blackwell GPU's name.
        ```python
        import torch
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        # Exit python with exit()
        ```

2.  **Install Other Project Dependencies:**
    The original `requirements.txt` is outdated. We'll install the key packages manually to get modern, compatible versions.

    ```bash
    # For running the notebooks
    conda install -c conda-forge jupyterlab

    # For the Triton client and ML models
    pip install tritonclient[all]
    pip install xgboost scikit-learn pandas numpy kaggle
    ```

-----

### Step 5: Update the Triton Inference Server (Crucial)

The old Triton Docker container (`22.12-py3` or similar) **will not detect your GPU**. You must pull a modern container.

1.  **Pull a Blackwell-Compatible Triton Image:**
    NVIDIA container releases from `25.05` onward support the Blackwell architecture. We'll pull a recent one, such as `25.10-py3`.

    ```bash
    docker login nvcr.io
    docker pull nvcr.io/nvidia/tritonserver:25.10-py3
    ```

2.  **Edit the Inference Notebook:**
    Open the `3_triton_inference.ipynb` notebook in Jupyter Lab.

3.  **Find the `docker run` Command:**
    Search the notebook for the line that starts with `!docker run`. It will look something like this:

    > **Original Code (example):**
    > `!docker run ... nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver ...`

4.  **Replace the Image Tag:**
    Change **only** the image name and tag to the new one you just pulled.

    > **New Code:**
    > `!docker run ... nvcr.io/nvidia/tritonserver:25.10-py3 tritonserver ...`

-----

### Step 6: Run and Debug the Project

You are now performing a **code migration**. This project was written for libraries that are 2-3 years old. You should expect to encounter and fix errors.

Run the notebooks one by one in Jupyter Lab:

1.  **`0_download_data.ipynb`**: This should run without any issues.
2.  **`1_auto_regressive_rnn.ipynb`**: This runs PyTorch code. You might find minor API differences between the old PyTorch it was written for and the new PyTorch 2.7+.
3.  **`2_xgboost_end2end.ipynb`**: This runs RAPIDS `cuDF` and `XGBoost`. This is **highly likely** to have breaking changes. `cuDF` and `XGBoost` APIs have evolved significantly since RAPIDS 23.04. You will need to read the error messages and update the code to use the modern APIs from RAPIDS 25.10.
4.  **`3_triton_inference.ipynb`**: After updating the Docker tag, you may still face issues. The model configuration files (`config.pbtxt` for each model) might need to be updated to match the new Triton server's requirements, or the `tritonclient` Python API may have changed.

This guide provides the correct, modern environment. The final step of debugging the code against the new libraries will be up to you. Good luck\!

Would you like me to help you find the API documentation for RAPIDS or PyTorch if you get stuck on a specific error?

### Notebook updates
## For 0_download_data.ipynb
You will need to update the thrid cell with the path. Change:

```bash
PATH = '/raid/data/ml/kaggle/amex'
```
to
```bash
PATH = 'data/amex'
```
Cell 16, you will need to verify your Kaggle account and accept the terms. Go to this page - https://www.kaggle.com/c/amex-default-prediction, click the Late Submisssion button. If your account hasn't been verified, you will be walked through it and then you can click the Late Submission button and accept the terms in the window that pops-up

## 1_auto_regressive_rnn.ipynb

```bash
conda install -c conda-forge pytorch-lightning
```
## 3_triton_inference.ipynb
Need to update the notebook to load the Triton Server first and then load the client.

Please run the notebooks in the following order:
- 0_download_data.ipynb
- 1_auto_regressive_rnn.ipynb
- 2_xgboost_end2end.ipynb
- 3_triton_inference.ipynb
