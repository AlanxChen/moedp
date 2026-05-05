# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AlanxChen/moedp.git
cd moedp
```

## 2. Create the Environment

The example below uses CUDA 12.1. If your CUDA version is different, install the
matching PyTorch wheel from the official PyTorch instructions.

```bash
conda create -n moedp python=3.8 -y
conda activate moedp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 5. Install MoE-DP and Local Packages

```bash
cd moe-dp && pip install -e . && cd ..
cd third_party/gym-0.21.0 && pip install -e . && cd ../..
```

Install `robosuite-task-zoo` separately with `--no-deps` to avoid pulling
`mujoco-py`; this project uses `mujoco==2.3.2` instead.

```bash
pip install --no-deps "robosuite_task_zoo @ https://github.com/AlanxChen/robosuite-task-zoo/archive/522d4a012185278e51a65068ad19a0fb0bc9cf8d.tar.gz"
```



## 6. Configure MuJoCo Rendering

Add the following lines to your shell configuration file, such as `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

## 7. Verify the Installation

Make sure only MuJoCo `2.3.2` is installed:

```bash
pip list | grep -i mujoco
```

If multiple MuJoCo versions are listed, uninstall the extra versions and keep
`mujoco==2.3.2`.
