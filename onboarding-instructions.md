# IsaacLab-Solo Onboarding Instructions

Goal: fork my IsaacLab-Solo repository, install the same Isaac Sim / Isaac Lab stack, and run one of my Solo12 policies.

## Reference Setup On My Machine

This is the local setup I use in my `env_isaaclab` conda environment:

- Remote: `ssh://git@gitlab.iri.upc.edu:2202/jbeltran/isaaclab-solo.git`
- Python: `3.11.14`
- Isaac Sim pip packages: `5.1.0.0`
- Isaac Lab editable packages:
  - `isaaclab==0.54.2`
  - `isaaclab_tasks==0.11.12`
  - `isaaclab_assets==0.2.4`
  - `isaaclab_rl==0.4.7`
- PyTorch: `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`
- RL libraries: `skrl==1.4.3`, `rsl-rl-lib==3.1.2`
- My current local NVIDIA driver: `580.159.03`

The important Isaac Sim version is therefore **Isaac Sim 5.1.0**.
Note: On the cluster  I'm using Isaac Sim 4.5 because the nvidia drivers are older.


Official references:

- Isaac Lab local installation: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>
- Isaac Lab install using Isaac Sim pip packages: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
- Isaac Sim 5.1 Python/pip installation: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html>

## 1. Fork And Clone The Repo

Fork my repository on GitLab (https://gitlab.iri.upc.edu/jbeltran/isaaclab-solo), then clone your fork locally. 
⚠️ Clone it in your home directory and name the folder `IsaacLab`. I had problems related to this before!

```bash
cd ~
git clone <your-fork-ssh-url> IsaacLab
cd ~/IsaacLab
```


This repo stores USD assets with Git LFS. Install Git LFS and pull the real asset files:

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```


## 2. Create The Conda Environment
Following these instructions:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
The only difference is that instead of cloning the official IsaacLab repo, you should clone the fork of my repo


Isaac Sim 5.x requires Python 3.11.

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
python -m pip install --upgrade pip
```

System requirements to check before installing:

```bash
ldd --version
nvidia-smi
```

Isaac Sim pip installation needs GLIBC `2.35+`. The current Isaac Lab docs recommend a recent NVIDIA production driver for Isaac Sim 5.1; my local driver is `580.159.03`.

## 3. Install Isaac Sim 5.1 And Isaac Lab

From inside `env_isaaclab`:

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

```
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install Isaac Lab from the fork checkout:

```bash
cd ~/IsaacLab
sudo apt install cmake build-essential
```

```
./isaaclab.sh --install # or "./isaaclab.sh -i"
```


## 4. Run One Of My Solo12 Policies

Download [checkpoint](https://drive.google.com/file/d/11bf5YmHNPFuXM1YOVRbBNoSP0PwBAfHI/view?usp=sharing), then put it somewhere local, for example:

```bash
mkdir -p ~/IsaacLab/checkpoints
```

Then change checkpoint path on the command below and run it:

```bash
cd ~/IsaacLab
conda activate env_isaaclab

./isaaclab.sh -p source/scripts/rsl_rl/play_direct_0325.py --task="solo12-v0" --checkpoint "/home/jordibelp/IsaacLab/logs/skrl/checkpoints/0604_vxfdhpup_model_20000.pt" --num_envs 1 --duration_s 2000 --cmd_init 0.5 0.0 0.0 --episode_length_s 80 --apply-force-ui env.initial_position=safe env.tricky_terrain=False 'env.forces_applied_to_base_curriculum=[15.0]' env.enable_observation_corruption=False --disable_training_gain_sync
```

You should see Solo jumping 😃 
