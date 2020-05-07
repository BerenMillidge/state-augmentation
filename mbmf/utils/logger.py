import os
import json
from datetime import datetime
import pprint
import numpy as np

class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = "log_" + logdir + "_" + str(seed) + "/"
        self.metrics_path = self.path + "metrics.json"
        self.mean_path = self.path + "means.npy"
        self.stds_path = self.path + "stds.npy"
        
        self.print_path = self.path + "out.txt"
        os.makedirs(self.path, exist_ok=True)
        self.metrics = {}
        self._init_print()
        self._setup_metrics()
        
        self._cem_means = []
        self._cem_stds = []
        self.cem_means_all = []
        self.cem_stds_all = []

    def log(self, string):
        f = open(self.print_path, "a")
        f.write("\n")
        f.write(str(string))
        f.close()
        print(string)

    def log_episode(self, reward, steps):
        # TODO
        def process_val(val):
            if isinstance(val, list):
                return val[0]
            else:
                return val
        self.metrics["rewards"].append(reward)
        self.metrics["steps"].append(steps)
        total_steps = sum([process_val(step) for step in self.metrics["steps"]])
        self.log(f"reward: {reward} | steps: {steps} | total_steps {total_steps}")
        
    def save(self):
        self._save_json(self.metrics_path, self.metrics)
        np.save(self.mean_path, self.cem_means_all)
        np.save(self.stds_path, self.cem_stds_all)
        self.log("Saved _metrics_")

    def get_video_path(self, episode):
        return self.video_dir + "{}.mp4".format(episode)

    def _init_print(self):
        f = open(self.print_path, "w")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f.write(current_time)
        f.close()

    def log_cem_stats(self, means, stds):
        self._cem_means.append(means.detach().cpu().numpy())
        self._cem_stds.append(stds.detach().cpu().numpy())

    def flush_cem_stats(self):
        self.cem_means_all.append(self._cem_means)
        self.cem_stds_all.append(self._cem_stds)
        self._cem_means = []
        self._cem_stds = []

    def _setup_metrics(self):
        self.metrics = {
            "rewards": [],
            "steps": [],
            "cem_means": [],
            "cem_stds": [],
        }

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)
