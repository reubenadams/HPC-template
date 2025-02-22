#################### SGE DIRECTIVES ####################

# Tells SGE to use bash as the interpreting shell for the job script.
#$ -S /bin/bash

# Sets a hard limit on how long your job can run. It's wallclock time (format hours:minutes:seconds), i.e. the time measured by a clock on the wall, rather than CPU time ()
#$ -l h_rt=1:0:0

# Request memory per core. Check in Process Memory in Use (MB) in wandb. Total memory is num cores x this value.
#$ -l mem=1G

# -l gpu=1  # Change to #$ if you want to use GPUs.
#$ -pe smp 4  # Number of cores requested.

# Only launch on A100 nodes.
# -ac allow=L

# Set the name of the job.
#$ -N ucabra5test

#$ -R y  # Reserve resources once they become free, until you have as much as you need. You definitely want this.
#$ -j y  # Merge the error and output streams into a single file (stdout, stderr). You definitely want this.

#$ -cwd  # All scripts have to work in a directory. This line says to use the directory we launched the script in.


#################### BASH COMMANDS ####################

date  # Print the current time.
# nvidia-smi  # Prints the GPUs available.

cd $HOME/HPC-template  # Get this by running pwd in terminal. It's just the name of your repo if you git cloned it into the home directory.

python3 --version
which python3
module load python/3.11.4
python3 --version
which python3

pip list
source .venv/bin/activate
python3 --version
pip list

# This is software you don't need.
# source $UCL_CONDA_PATH/etc/profile.d/conda.sh
# conda activate stackingbert

# module load beta-modules
# module unload gcc-libs/4.9.2
# module load gcc-libs/10.2.0
# module unload compilers/intel/2018/update3
# module load compilers/gnu/10.2.0

# export HF_DATASETS_CACHE=$HOME/Scratch/hf_dataset_cache  # Put datasets on scratch space. You're not using this as your datasets are small.
mkdir $TMPDIR/wandb_cache  # Put wandb data on tmp as wandb writes lots of small files. This might speed up the job, and is polite.
export WANDB_CACHE_DIR=$TMPDIR/wandb_cache
mkdir $TMPDIR/wandb
export WANDB_DIR=$TMPDIR/wandb
export WANDB_API_KEY=$(head -n 1 $HOME/HPC-template/wandb_api_key.txt)  # Setting the API key for wandb.

# Count is the number of runs to do. Syntax is `wandb agent username/project/sweep_id`, where sweep_id is what was returned by wandb.sweep
wandb agent teamreuben/HPC-template/h5euzs2d --count 2
# python main.py  # TODO: Should this be python3?
