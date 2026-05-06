# MoE-DP: An MoE-Enhanced Diffusion Policy for Robust Long-Horizon Robotic Manipulation with Skill Decomposition and Failure Recovery

<p align="center" style="font-size: 18px">
   <a href="https://arxiv.org/abs/2511.05007">[Paper]</a>&emsp;<a href="https://moe-dp-website.github.io/MoE-DP-Website/">[Project Website]</a>
</p>

This repository is the official PyTorch implementation of **MoE-DP**. **MoE-DP**
is an MoE-enhanced diffusion policy for robust long-horizon robotic manipulation
with skill decomposition and failure recovery. 

# 🛠️ Installation Instructions

Please follow the full setup guide in [INSTALL.md](INSTALL.md). A minimal setup
is shown below.

First, clone the repository and create a conda environment. The example below
uses CUDA 12.1; please install the PyTorch wheel that matches your CUDA version.

```bash
git clone https://github.com/AlanxChen/moedp.git
cd moedp
conda create -n moedp python=3.8 -y
conda activate moedp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Next, install the Python dependencies and PyTorch3D.

```bash
pip install cmake==3.22
pip install -r requirements.txt
pip install --src third_party -r requirements_third_party.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Then, install MoE-DP and the required local packages.

```bash
cd moe-dp && pip install -e . && cd ..
cd third_party/gym-0.21.0 && pip install -e . && cd ../..
```

Finally, configure MuJoCo rendering for headless GPU evaluation. Add the
following lines to your shell configuration file, such as `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

Tips: please make sure only MuJoCo `2.3.2` is installed before launching large
training or evaluation jobs.

```bash
pip list | grep -i mujoco
```

## 📦 Dataset Download and Extraction

Please download the dataset from Google Drive:

`https://drive.google.com/file/d/1mhlNRfdQCfwmbJ8U1OSAkPaXVfe7SO-S/view?usp=drive_link`

After downloading and extracting the dataset, place it under:

```text
$YOUR_REPO_PATH/moe-dp/data
```

The expected dataset layout is:

```text
moe-dp/data/<task_name>/<task_name>.hdf5
```

For example:

```text
moe-dp/data/Kitchen_Cleanup_T0/Kitchen_Cleanup_T0.hdf5
```

## 💻 Code Usage

### Train a single task

If you would like to train MoE-DP on a single MimicGen task, please use
`scripts/train_policy.sh`.

```bash
bash scripts/train_policy.sh dp_unet_mlp_moe Kitchen_Cleanup_T0 0000 0 0
```

The script uses `n_demo=100` by default. The argument order is:

```bash
bash scripts/train_policy.sh <algorithm> <task_name> <run_tag> <seed> <gpu_id>
```

To train with the baseline diffusion policy, replace `dp_unet_mlp_moe` with
`dp_unet`.

To train on other tasks, replace `Kitchen_Cleanup_T0` with
`Coffee_Preparation_T0`, `Hammer_Cleanup_T0`, `Kitchen_T0`, `Mug_Cleanup_T0`,
or `Table_Cleanup_T0`.


### Train multiple tasks on multiple GPUs

If you would like to schedule multiple training jobs across several GPUs, please
use `scripts/train_taskmd_multi_gpu.sh`.

```bash
bash scripts/train_taskmd_multi_gpu.sh \
  --gpus 0,1,2,3 \
  --seeds 0 \
  --run-tag moe_multi
```

This script schedules the fixed current task list, launches both `dp_unet` and
`dp_unet_mlp_moe`, runs at most one job per GPU at a time, and writes logs
under `logs/taskmd_train/<run_tag>/`.

You can preview the generated jobs without launching training:

```bash
bash scripts/train_taskmd_multi_gpu.sh \
  --gpus 0,1 \
  --seeds 0 \
  --task-regex "Kitchen|Coffee" \
  --dry-run
```

### Evaluate a checkpoint

If you would like to evaluate a trained MoE-DP checkpoint, please use
`scripts/eval_policy_mimicgen_async.sh`.

```bash
bash scripts/eval_policy_mimicgen_async.sh \
  --alg dp_unet_mlp_moe \
  --task Kitchen_Cleanup_T0 \
  --checkpoint data/outputs/your_train_run/checkpoints/latest.ckpt \
  --tag eval1 \
  --seed 3 \
  --gpu 0 \
  --n-envs 25 \
  --n-test 50 \
  --n-test-vis 20
```

The main arguments are:

- `--alg`: policy config name, such as `dp_unet` or `dp_unet_mlp_moe`
- `--task`: MimicGen task name
- `--checkpoint`: checkpoint path to evaluate
- `--gpu`: CUDA device id
- `--n-envs`: number of parallel evaluation environments
- `--n-test`: number of evaluation episodes per setting
- `--n-test-vis`: number of rollout videos to record

For full argument help, run:

```bash
bash scripts/eval_policy_mimicgen_async.sh --help
```

### Visual rollout recording

The visual analysis pipeline is currently under development. We will add the
corresponding usage instructions and examples once this part is finalized.

## 📝 Citation

If you use our method or code in your research, please consider citing the paper
as follows:

```bibtex
@misc{cheng2025moedpmoeenhanceddiffusionpolicy,
      title={MoE-DP: An MoE-Enhanced Diffusion Policy for Robust Long-Horizon Robotic Manipulation with Skill Decomposition and Failure Recovery}, 
      author={Baiye Cheng and Tianhai Liang and Suning Huang and Maanping Shao and Feihong Zhang and Botian Xu and Zhengrong Xue and Huazhe Xu},
      year={2025},
      eprint={2511.05007},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.05007}, 
}
```

## 🙏 Acknowledgement

MoE-DP is licensed under the MIT license. This implementation builds on top of
several excellent open-source projects, including
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy),
[MimicGen](https://github.com/NVlabs/mimicgen_environments). We would like to
thank the authors and maintainers of these projects for open-sourcing their
codebases.
