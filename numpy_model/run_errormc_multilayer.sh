sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/2-1/errormc/params_errormc.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/4-2-1/errormc/params_errormc.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/8-4-2-1/errormc/params_errormc.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/16-8-4-2-1/errormc/params_errormc.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/32-16-8-4-2-1/errormc/params_errormc.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/2-1/ann/params_ann.json --task fw_only &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/4-2-1/ann/params_ann.json --task fw_only &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/8-4-2-1/ann/params_ann.json --task fw_only &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/16-8-4-2-1/ann/params_ann.json --task fw_only &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/32-16-8-4-2-1/ann/params_ann.json --task fw_only &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/2-1/sacramento2018/params_sacramento2018.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/4-2-1/sacramento2018/params_sacramento2018.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/8-4-2-1/sacramento2018/params_sacramento2018.json --task fw_only --compare BP & 
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/16-8-4-2-1/sacramento2018/params_sacramento2018.json --task fw_only --compare BP &
sbatch ~/error-mc/toy_model/slurm_script.sh python ~/error-mc/toy_model/runner.py --params experiments/exp55_errormc_multilayer/32-16-8-4-2-1/sacramento2018/params_sacramento2018.json --task fw_only --compare BP 
