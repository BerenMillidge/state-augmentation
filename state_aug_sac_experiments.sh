SCRATCH_HOME=/disk/scratch
USER=s1686853
SCRATCH_BASE=${SCRATCH_HOME}/${USER}
SAVE_BASE=/home/s1686853/state_augmentation
mkdir -p ${SAVE_BASE}

python sac_script.py --logdir ${SCRATCH_BASE}/baseline_1 --savedir ${SAVE_BASE}/baseline_1
python sac_script.py --logdir ${SCRATCH_BASE}/baseline_2 --savedir ${SAVE_BASE}/baseline_2
python sac_script.py --logdir ${SCRATCH_BASE}/baseline_3 --savedir ${SAVE_BASE}/baseline_3

python sac_script.py --logdir ${SCRATCH_BASE}/aug_5_with_reward_1 --savedir ${SAVE_BASE}/aug_5_with_reward_1 --n_augments 5
python sac_script.py --logdir ${SCRATCH_BASE}/aug_5_with_reward_2 --savedir ${SAVE_BASE}/aug_5_with_reward_2 --n_augments 5
python sac_script.py --logdir ${SCRATCH_BASE}/aug_5_with_reward_3 --savedir ${SAVE_BASE}/aug_5_with_reward_3 --n_augments 5

python sac_script.py --logdir ${SCRATCH_BASE}/aug_10_with_reward_1 --savedir ${SAVE_BASE}/aug_10_with_reward_1 --n_augments 10
python sac_script.py --logdir ${SCRATCH_BASE}/aug_10_with_reward_2 --savedir ${SAVE_BASE}/aug_10_with_reward_2 --n_augments 10
python sac_script.py --logdir ${SCRATCH_BASE}/aug_10_with_reward_3 --savedir ${SAVE_BASE}/aug_10_with_reward_3 --n_augments 10

python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_reward_1 --savedir ${SAVE_BASE}/aug_20_with_reward_1 --n_augments 20
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_reward_2 --savedir ${SAVE_BASE}/aug_20_with_reward_2 --n_augments 20
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_reward_3 --savedir ${SAVE_BASE}/aug_20_with_reward_3 --n_augments 20

python sac_script.py --logdir ${SCRATCH_BASE}/aug_10_1_std_with_reward_1 --savedir ${SAVE_BASE}/aug_10_1_std_with_reward_1 --n_augments 10 --augment_std 1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_10_1_std_with_reward_2 --savedir ${SAVE_BASE}/aug_10_1_std_with_reward_2 --n_augments 10 --augment_std 1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_10_1_std_with_reward_3 --savedir ${SAVE_BASE}/aug_10_with_1_std_reward_3 --n_augments 10 --augment_std 1

python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_01_std_reward_1 --savedir ${SAVE_BASE}/aug_20_with_01_std_reward_1 --n_augments 20 --augment_std 0.1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_01_std_reward_2 --savedir ${SAVE_BASE}/aug_20_with_01_std_reward_2 --n_augments 20 --augment_std 0.1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_01_std_reward_3 --savedir ${SAVE_BASE}/aug_20_with_01_std_reward_3 --n_augments 20 --augment_std 0.1

python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_01_std_reward_1 --savedir ${SAVE_BASE}/aug_20_with_01_std_reward_1 --n_augments 20 --augment_std 0.1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_01_std_reward_2 --savedir ${SAVE_BASE}/aug_20_with_01_std_reward_2 --n_augments 20 --augment_std 0.1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_01_std_reward_3 --savedir ${SAVE_BASE}/aug_20_with_01_std_reward_3 --n_augments 20 --augment_std 0.1

python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_05_std_reward_1 --savedir ${SAVE_BASE}/aug_20_with_05_std_reward_1 --n_augments 20 --augment_std 1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_05_std_reward_2 --savedir ${SAVE_BASE}/aug_20_with_05_std_reward_2 --n_augments 20 --augment_std 1
python sac_script.py --logdir ${SCRATCH_BASE}/aug_20_with_05_std_reward_3--savedir ${SAVE_BASE}/aug_20_with_05_std_reward_3 --n_augments 20 --augment_std 1

