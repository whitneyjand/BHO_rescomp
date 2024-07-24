#!/bin/bash --login

#SBATCH --time=36:00:00                        # walltime
#SBATCH --ntasks=4                             # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                              # number of nodes
#SBATCH --mem-per-cpu=1000M                    # memory per CPU core
#SBATCH -J "BHO"                               # job name
#SBATCH --mail-user=wanderson@math.byu.edu     # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

profile=slurm_${SLURM_JOB_ID}_$(hostname)

#This method for using ipyparallel with slurm was taken from https://k-d-w.org/blog/2015/05/using-ipython-for-parallel-computing-on-an-mpi-cluster-using-slurm/
echo "Creating profile ${profile}"
ipython profile create ${profile}


echo "Launching controller"
/home/whitjand/.conda/envs/rescomp/bin/ipcontroller --ip="*" --profile=${profile} --log-to-file &
sleep 10

echo "Launching engines"
srun /home/whitjand/.conda/envs/rescomp/bin/ipengine --profile=${profile} --location=$(hostname) --log-to-file &
sleep 30

echo "Launching job"

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE (1 - rho, 2 - pthin, 3 - profile)
mamba activate rescomp
python3 wa_bho.py $1 $2 "$profile"