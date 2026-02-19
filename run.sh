#!/bin/bash
#
#SBATCH --job-name=lma
#
#SBATCH --ntasks=1
#SBATCH -p sched_mit_sloan_batch_r8
#SBATCH -o "logs/slurm-%a.out"
#SBATCH -e "logs/slurm-%a.out"
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=ERROR,END
#SBATCH --mail-user=jhays@mit.edu
#SBATCH --array=0-99


export JHAYS="/home/jhays"
# python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 3 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 5 
# python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 3 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 10 #--keep_existing_results
# python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 3 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1 #--keep_existing_results
# python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 3 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1e-1 #--keep_existing_results
python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 1 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1e-2 #--keep_existing_results
python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 1 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1e-3 #--keep_existing_results
python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 1 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1e-4 #--keep_existing_results
python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 1 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1e-5 #--keep_existing_results
python3 $JHAYS/strategic_candidacy_lmarena/rank_quality.py --B 1 --job_id $SLURM_ARRAY_TASK_ID --max_num_clones 1 --noisy_producer_rankings --noise_level 1e-6 #--keep_existing_results
