**Status:** This repository is built based on the openai baselines and meta_mb (add links)

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```
    
## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```


## Installation
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu==1.13.1 # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow==1.13.1
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). 
For mujoco150, install mujoco-py by running 
```
pip3 install mujoco-py==1.50.1.35
```

For mujoco200, which support simulation of deformable objects, install mujoco-py by running 
```
pip3 install -U 'mujoco-py<2.1,>=2.0'
```

### Docker
If not installed yet, [set up](https://docs.docker.com/install/) docker on your machine.
Pull our docker container ``vioichigo/base`` from docker-hub:

For mujoco200 and mujoco-py2.0, run:
```
docker pull vioichigo/base:soft
```

For mujoco150 and mujoco-py1.5, run:
```
docker pull vioichigo/base:try2
```

All the necessary dependencies are already installed inside the docker container.

### Setting up the doodad experiment launcher with EC2 support

Install AWS commandline interface

```
sudo apt-get install awscli
```

and configure the asw cli

```
aws configure
```

Clone the doodad repository 

```
git clone https://github.com/jonasrothfuss/doodad.git
```

Install the extra package requirements for doodad
```
cd doodad && pip install -r requirements.txt
```

Configure doodad for your ec2 account. First you have to specify the following environment variables in your ~/.bashrc: 
AWS_ACCESS_KEY, AWS_ACCESS_KEY, DOODAD_S3_BUCKET

Then run
```
python scripts/setup_ec2.py
```

Set S3_BUCKET_NAME in experiment_utils/config.py to your bucket name

## Experiments

### How to run experiments 
examples:

On your own machine:
```
python run_scripts/her_run_sweep.py
```
On docker:
```
python run_scripts/her_run_sweep.py --mode local_docker
```
On aws:
```
python run_scripts/her_run_sweep.py --mode ec2
```
To pull results from aws
```
python experiment_utils/sync_s3.py experiment_name
```
To check all the experiments running on aws
```
python experiment_utils/ec2ctl.py jobs
```
To kill experiments on aws
```
python experiment_utils/ec2ctl.py kill_f the_first_few_characters_of_your_experiment
```
OR
```
python experiment_utils/ec2ctl.py kill specific_full_name_of_an_experiment
```
### How to visualize results

```
python viskit/frontend.py any_folder_with_all_the_experiments_that_you_want_to_visualize
```



## Subpackages

- [DDPG](https://github.com/VioIchigo/tactile-baselines/tree/master/tactile_baselines/ddpg)
- [HER](https://github.com/VioIchigo/tactile-baselines/tree/master/tactile_baselines/her)
- [PPO2](https://github.com/VioIchigo/tactile-baselines/tree/master/tactile_baselines/ppo2) 
- [SAC](https://github.com/VioIchigo/tactile-baselines/tree/master/tactile_baselines/sac) 


