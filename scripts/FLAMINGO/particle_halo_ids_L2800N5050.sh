#!/bin/bash -l
#
#SBATCH --array=1,6
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -J HYDRO_FIDUCIAL
#SBATCH -o ./logs/L2800N5040/particle_halo_ids.%x.lightcone%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 72:00:00

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3
module load python/3.12.4

source "/cosma/apps/dp004/${USER}/lightcone_env/bin/activate"

# Simulation to do (based on job name, e.g. sbatch --job-name=HYDRO_FIDUCIAL)
sim="L2800N5040/${SLURM_JOB_NAME}"

# Which lightcone to use - submit as a job array, e.g. sbatch --array=0-0 ...
lightcone_nr="${SLURM_ARRAY_TASK_ID}"

# Location of the lightcone particle data
lightcone_dir="/cosma8/data/dp004/flamingo/Runs/${sim}/particle_lightcones/"
lightcone_base="lightcone${lightcone_nr}"
echo "${lightcone_dir}/${lightcone_base}"

halo_lightcone_filenames="/cosma8/data/dp004/flamingo/Runs/${sim}/sorted_hbt_lightcone_halos/${lightcone_base}/lightcone_halos_%(file_nr)04d.hdf5"
echo "${halo_lightcone_filenames}"

soap_filenames="/cosma8/data/dp004/flamingo/Runs/${sim}/SOAP-HBT/halo_properties_%(snap_nr)04d.hdf5"
echo "${soap_filenames}"

# Where to write the output
output_dir="./lightcone_particle_halo_ids/S0200crit/${sim}/lightcone${lightcone_nr}/"
\mkdir -p "${output_dir}"
lfs setstripe --stripe-count=1 --stripe-size=8M "${output_dir}"

# match Gas and Star particles to haloes by smallesr r/R200c values, centrals-only. 
# use batches of 512 files at once. 
mpirun -- python3 -m mpi4py -m lightcone_io.particle_halo_ids \
       "${lightcone_dir}" \
       "${lightcone_base}" \
       "${halo_lightcone_filenames}" \
       "${soap_filenames}" \
       "${output_dir}" \
       --soap-so-name="SO/200_crit" \
       --centrals-only \
       --overlap-method=mass-weighted \
       --particle-files-per-batch=512 \
       --particle-types Stars Gas
