# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import numpy as np


class RewardMetricAnalyzer:

    def __init__(self, env, writer):
        reward_info = env.get_reward_info()
        self.writer = writer
        self.data_tags = list(reward_info[0].keys())
        self.data_size = 0
        self.data_mean = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_square_sum = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_min = np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_max = -np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        self.metrics = dict()

    def add_reward_info(self, info):
        self.data_size += len(info)

        for i in range(len(self.data_tags)):
            for j in range(len(info)):
                self.data_square_sum[i] += info[j][self.data_tags[i]]*info[j][self.data_tags[i]]
                self.data_mean[i] += info[j][self.data_tags[i]]
                self.data_min[i] = min(self.data_min[i], info[j][self.data_tags[i]])
                self.data_max[i] = max(self.data_max[i], info[j][self.data_tags[i]])

    def analyze_and_plot(self, step):
        self.data_mean /= self.data_size
        data_std = np.sqrt((self.data_square_sum - self.data_size * self.data_mean * self.data_mean) / (self.data_size + 1 + 1e-16))

        ensemble_avg = dict()
        ensemble_std = dict()
        ensemble_min = dict()
        ensemble_max = dict()
        for data_id in range(len(self.data_tags)): # much better to see!
            self.writer.add_scalar('rewards_avg/' + self.data_tags[data_id], self.data_mean[data_id], global_step=step)
            self.writer.add_scalar('rewards_std/' + self.data_tags[data_id], data_std[data_id], global_step=step)
            self.writer.add_scalar('rewards_min/' + self.data_tags[data_id], self.data_min[data_id], global_step=step)
            self.writer.add_scalar('rewards_max/' + self.data_tags[data_id], self.data_max[data_id], global_step=step)

            # self.writer.add_scalars('rewards_ext/' + self.data_tags[data_id],{
            #     'min' : self.data_min[data_id],
            #     'max' : self.data_max[data_id]
            # }, global_step=step)

            ensemble_avg[self.data_tags[data_id]] = self.data_mean[data_id]
            ensemble_std[self.data_tags[data_id]] = data_std[data_id]
            ensemble_min[self.data_tags[data_id]] = self.data_min[data_id]
            ensemble_max[self.data_tags[data_id]] = self.data_max[data_id]

        # commented out since it messes up the org.
        # self.writer.add_scalars('ensemble/avg', ensemble_avg, global_step=step)
        # self.writer.add_scalars('ensemble/std', ensemble_std, global_step=step)
        # self.writer.add_scalars('ensemble/min', ensemble_min, global_step=step)
        # self.writer.add_scalars('ensemble/max', ensemble_max, global_step=step)

        self.data_size = 0
        self.data_mean = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_square_sum = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_min = np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_max = -np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)

    def update_metrics(self,env,additional_metrics_dict):
        self.metrics = env.get_metrics()
        self.metrics = self.metrics | additional_metrics_dict

    def plot_metrics(self, env, step):
        for metric_tag in self.metrics:
            self.writer.add_scalar('_metric/'+metric_tag, self.metrics[metric_tag], global_step=step)

