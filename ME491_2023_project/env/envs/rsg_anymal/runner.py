from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from ME491_2023_project.helper.raisim_gym_helper import (
    ConfigurationSaver, load_param,load_param_selfplay, tensorboard_launcher)
from ME491_2023_project.env.bin.rsg_anymal import NormalSampler
from ME491_2023_project.env.bin.rsg_anymal import RaisimGymEnv
from ME491_2023_project.env.RewardAnalyzer import RewardAnalyzer
import os
import math
import time
import ME491_2023_project.algo.ppo.module as ppo_module
import ME491_2023_project.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse

# task specification
task_name = "ME491_2023_project"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-v', '--oppweight', help='path for opponent weight', type=str, default='')
parser.add_argument('-l', '--lr', help='prescribed starting learning rate',type=float,default=-1)
parser.add_argument('-c','--config', help='config file name (without .yaml) in preset-cfg',type=str,default='none')

args = parser.parse_args()
mode = args.mode
weight_path = args.weight
opp_weight_path = args.oppweight
starting_lr = args.lr
cfg_name = args.config


# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
if cfg_name == '' or cfg_name == 'none':
    print("[RUNNER] No cfg file was given. Loading default from: "+task_path+"/cfg.yaml")
    cfg_path = task_path + "/cfg.yaml"
else:
    print("[RUNNER] Loading config: "+cfg_name)
    cfg_path = task_path + "/preset-cfg/"+cfg_name+".yaml"
# print("[CHRIS] self-play start!! (runner l43)")
cfg = YAML().load(open(cfg_path, 'r'))

is_dummy = cfg['environment']['training_dummy_opponent']

if (not is_dummy) and cfg['environment']['training_mode'] == 1 and (opp_weight_path == '' or opp_weight_path == 'null'):
    print("[SELF-TRAIN] No opponent weight provided. Proceeding with dummy opponent")
    is_dummy = True


# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])


# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

opp_ob_dim = env.opp_num_obs
opp_act_dim= env.opp_num_acts

print(opp_ob_dim)
print(opp_act_dim)

num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           5.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

# opponent only needs actor
opp_actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, opp_ob_dim, opp_act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(opp_act_dim,
                                                                           env.num_envs,
                                                                           5.0,
                                                                           NormalSampler(opp_act_dim),
                                                                           cfg['seed']),
                         device)


saver = ConfigurationSaver(log_dir=home_path + "/ME491_2023_project/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/runner.py", task_path + "/Environment.hpp"])

tensorboard_launcher(saver.data_dir+"/..", False)  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.998,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

if starting_lr > 0: # reset the optimizer
    print("[PPO] Using Prescribed Learning Rate: "+ str(starting_lr))
    ppo.optimizer = torch.optim.Adam([*ppo.actor.parameters(), *ppo.critic.parameters()], lr=starting_lr)
    # self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)

reward_analyzer = RewardAnalyzer(env, ppo.writer)

# if mode == 'retrain':
#     load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
if cfg['environment']['training_mode'] == 0:
    if mode == 'retrain':
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
else:
    if is_dummy:
        load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    else:
        load_param_selfplay(weight_path, opp_weight_path, env, actor, critic, ppo.optimizer, saver.data_dir, opp_actor)

env.turn_off_visualization()

def ppo_outer_loop(updates = 5000):
    for update in range(updates):
        start = time.time()
        env.reset()
        reward_sum = 0
        done_sum = 0
        average_dones = 0.

        if (update % cfg['environment']['eval_every_n'] == 0 and update != 0) or cfg['environment']['eval_every_n'] == 1:
            print("[RUNNER] Visualizing and evaluating the current policy")
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')
            # we create another graph just to demonstrate the save/load method
            loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
            loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

            # do the same for opponent
            opp_loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'],nn.LeakyReLU, opp_ob_dim, opp_act_dim)

            if cfg['environment']['training_mode'] == 1 and not is_dummy:
                opp_loaded_graph.load_state_dict(torch.load(opp_weight_path)['actor_architecture_state_dict'])

            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            for step in range(n_steps): # visualization (dry) run
                with torch.no_grad():
                    frame_start = time.time()
                    obs,opp_obs = env.observe(False)
                    # print(obs[0]) # CHECK OBS HERE
                    action = loaded_graph.architecture(torch.from_numpy(obs).cpu()).cpu().detach().numpy()
                    if cfg['environment']['training_mode'] != 1:
                        opp_action = np.zeros([1,1],dtype=np.float32)
                    else:
                        opp_action = opp_loaded_graph.architecture(torch.from_numpy(opp_obs).cpu()).cpu().detach().numpy()
                        if is_dummy:
                            opp_action = np.zeros_like(opp_action)
                    # print(action)
                    # print(opp_action)

                    # print("action: " + str(action.get_device()))
                    # print("opp_action: " + str(opp_action.get_device()))

                    reward, dones = env.step(action,opp_action)
                    reward_analyzer.add_reward_info(env.get_reward_info())
                    frame_end = time.time()
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

            env.stop_video_recording()
            env.turn_off_visualization()

            reward_analyzer.analyze_and_plot(update)
            env.reset()
            env.save_scaling(saver.data_dir, str(update))
            print("[RUNNER] Visualization Complete")

        # time measurements
        cur_time = time.time()
        t_observe = 0
        t_action  = 0
        t_envstep = 0
        t_ppostep = 0
        t_summing = 0
        t_backprop = 0
        # actual training
        for step in range(n_steps):
            obs,opp_obs = env.observe()
            t_observe += time.time()-cur_time
            cur_time = time.time()

            with torch.no_grad():
                action = ppo.act(obs)
                opp_action = opp_actor.sample(torch.from_numpy(opp_obs).to(device))[0]
                if cfg['environment']['training_dummy_opponent']:
                    # feed zeros if it is a dummy opponent!
                    opp_action = np.zeros_like(opp_action)

            t_action += time.time()-cur_time
            cur_time = time.time()

            reward, dones = env.step(action,opp_action)
            t_envstep += time.time()-cur_time
            cur_time = time.time()

            ppo.step(value_obs=obs, rews=reward, dones=dones)
            t_ppostep += time.time()-cur_time
            cur_time = time.time()

            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)
            t_summing += time.time()-cur_time
            cur_time = time.time()

        # take st step to get value obs
        obs,opp_obs = env.observe()
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))
        t_backprop = time.time() - cur_time

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

        end = time.time()

        print('[ITERATION-SUMMARY]-------------------------------------')
        print('[#{:>6}] MODE {:>2}'.format(update,cfg['environment']['training_mode']))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('--------------------------------------------------------')
        print('{:<7}|{:<7}|{:<7}|{:<7}|{:<7}|{:<7}|{:<7}|'.format('observe','action','envstep','ppostep','sum','grad','etc.'))
        print('{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|'.format(t_observe,t_action ,t_envstep,t_ppostep,t_summing,t_backprop,(end-start)-sum((t_observe,t_action,t_envstep,t_ppostep,t_summing,t_backprop))))
        print('--------------------------------------------------------\n')


ppo_outer_loop()



