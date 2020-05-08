# necessary experiments for initial tests -- see whether it will work on the edinburgh computers

import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
seeds = 1
base_call = "python sac_script.py --env_name Pendulum-v0"
output_file = open(generated_name, "w")
exp_name="prelim"
n_augments = [5,10,20,50]
augment_stds = [0.01,0.1,0.05,0.5]
reward_augment_stds = [0,0.01,0.05,0.1,0.5,1]

for n in n_augments:
    lpath = log_path + "/"+str(exp_name) + "/n_augments" + str(n)
    spath = save_path + "/" + str(exp_name) + "/n_augments" + str(n)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " "  + " --n_augments " + str(n) + " --augment_std " + str(0.05)
    print(final_call)
    print(final_call,file=output_file)

for s in augment_stds:
    lpath = log_path + "/"+str(exp_name) + "/augment_stds" + str(s)
    spath = save_path + "/" + str(exp_name) + "/augment_stds" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " "  + " --augment_std " + str(s) + " --n_augments " + str(10)
    print(final_call)
    print(final_call,file=output_file)

for r in reward_augment_stds:
    lpath = log_path + "/"+str(exp_name) + "/reward_augment_stds" + str(r)
    spath = save_path + "/" + str(exp_name) + "/reward_augment_stds" + str(r)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " "  + " --augment_std " + str(0.05) + "--n_augments " + str(10) +" --reward_std " + str(r)
    print(final_call)
    print(final_call,file=output_file)


for n in n_augments:
    lpath = log_path + "/"+str(exp_name) + "/jitter_n_augments" + str(n)
    spath = save_path + "/" + str(exp_name) + "/jitter_n_augments" + str(n)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " "  + " --n_augments " + str(n) + " --augment_std " + str(0.05) + " --sample_jitter True"
    print(final_call)
    print(final_call,file=output_file)

for s in augment_stds:
    lpath = log_path + "/"+str(exp_name) + "/jitter_augment_stds" + str(s)
    spath = save_path + "/" + str(exp_name) + "/jitter_augment_stds" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " "  + " --augment_std " + str(s) + " --n_augments " + str(10) + " --sample_jitter True"
    print(final_call)
    print(final_call,file=output_file)

for r in reward_augment_stds:
    lpath = log_path + "/"+str(exp_name) + "/jitter_reward_augment_stds" + str(r)
    spath = save_path + "/" + str(exp_name) + "/jitter_reward_augment_stds" + str(r)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " "  + " --augment_std " + str(0.05) + "--n_augments " + str(10) +" --reward_std " + str(r) + " --sample_jitter True"
    print(final_call)
    print(final_call,file=output_file)