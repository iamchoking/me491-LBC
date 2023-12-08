# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'

        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl


        self.num_obs = self.wrapper.getObDims()[0]
        self.num_acts = self.wrapper.getActionDims()[0]
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)

        self.opp_num_obs = self.wrapper.getObDims()[1]
        self.opp_num_acts= self.wrapper.getActionDims()[1]
        self._opp_observation = np.zeros([self.num_envs, self.opp_num_obs], dtype=np.float32)
        self.opp_actions = np.zeros([self.num_envs, self.opp_num_acts], dtype=np.float32)



        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)



        # /// Scaling
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)

        self.opp_count = 0.0
        self.opp_mean = np.zeros(self.opp_num_obs, dtype=np.float32)
        self.opp_mean = np.zeros(self.opp_num_obs, dtype=np.float32)
        self.opp_var  = np.zeros(self.num_obs,     dtype=np.float32)


    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action, opp_action):
        self.wrapper.step(action, opp_action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5, opp_dir_name="", opp_iteration=0, opp_count=1e5):
        self.wrapper.getObStatistics(self.mean,self.var,self.count,self.opp_mean,self.opp_var,self.opp_count)

        if dir_name != "":
            mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
            var_file_name = dir_name + "/var" + str(iteration) + ".csv"
            count_file_name = dir_name + "/count"+str(iteration)

            try:
                count_file = open(count_file_name,'r').read()
                self.count = float(count_file.strip())
                count_file.close()
            except:
                print("[ENV-LOAD] Failed to Retrieve [count] for <"+dir_name+">. using supplied value instead: "+str(count))
                self.count = count

            self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
            self.var = np.loadtxt(var_file_name, dtype=np.float32)

        if opp_dir_name != "":
            opp_mean_file_name = opp_dir_name + "/mean" + str(opp_iteration) + ".csv"
            opp_var_file_name = opp_dir_name + "/var" + str(opp_iteration) + ".csv"
            opp_count_file_name = opp_dir_name + "/count" + str(opp_iteration)

            try:
                opp_count_file = open(opp_count_file_name,'r').read()
                self.opp_count = float(opp_count_file.strip())
                opp_count_file.close()
            except:
                print("[ENV-LOAD] Failed to Retrieve [count] for <"+dir_name+">. using supplied value instead: "+str(opp_count))
                self.opp_count = opp_count

            self.opp_mean = np.loadtxt(opp_mean_file_name, dtype=np.float32)
            self.opp_var = np.loadtxt(opp_var_file_name, dtype=np.float32)

        self.wrapper.setObStatistics(self.mean, self.var, self.count, self.opp_mean, self.opp_var, self.opp_count)


    def save_scaling(self, dir_name, iteration): # (opponent doesn't need to save scaling info)
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        count_file_name = dir_name + "/count"+iteration
        self.wrapper.getObStatistics(self.mean, self.var, self.count, self.opp_mean, self.opp_var, self.opp_count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

        count_file = open(count_file_name, 'w+')
        count_file.write(str(self.count))
        count_file.close()


    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation,self._opp_observation, update_statistics)
        return self._observation,self._opp_observation

    def get_reward_info(self):
        return self.wrapper.getRewardInfo()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
