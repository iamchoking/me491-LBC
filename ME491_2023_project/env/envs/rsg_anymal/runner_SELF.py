from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from ME491_2023_project.helper.raisim_gym_helper import ConfigurationSaver, load_param_selfplay, tensorboard_launcher
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

args = parser.parse_args()
mode = args.mode
weight_path = args.weight
opp_weight_path = args.oppweight


# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# print("[CHRIS] self-play start!! (runner l43)")

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])



# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

opp_ob_dim = env.opp_num_obs
opp_act_dim= env.opp_num_acts

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

reward_analyzer = RewardAnalyzer(env, ppo.writer)

# if mode == 'retrain':
#     load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
load_param_selfplay(weight_path,opp_weight_path,env,actor,critic,ppo.optimizer,saver.data_dir,opp_actor)

env.turn_off_visualization()

def ppo_outer_loop(updates = 5000):
    for update in range(1000000):
        start = time.time()
        env.reset()
        reward_sum = 0
        done_sum = 0
        average_dones = 0.

        if (update % cfg['environment']['eval_every_n'] == 0 and update != 0) or cfg['environment']['eval_every_n'] == 1:
            print("Visualizing and evaluating the current policy")
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
            opp_loaded_graph.load_state_dict(torch.load(opp_weight_path)['actor_architecture_state_dict'])

            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            for step in range(n_steps): # visualization (dry) run
                with torch.no_grad():
                    frame_start = time.time()
                    obs,opp_obs = env.observe(False)
                    # print(obs[0]) # CHECK OBS HERE
                    action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                    opp_action = opp_loaded_graph.architecture(torch.from_numpy(opp_obs).cpu())

                    # print("action: " + str(action.get_device()))
                    # print("opp_action: " + str(opp_action.get_device()))

                    reward, dones = env.step(action.cpu().detach().numpy(),opp_action.cpu().detach().numpy())
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
            print("Visualization Complete")

        # time measurements
        curTime = time.time()
        t_observe = 0
        t_action  = 0
        t_envstep = 0
        t_ppostep = 0
        t_summing = 0
        t_backprop = 0
        # actual training
        for step in range(n_steps):
            obs,opp_obs = env.observe()
            t_observe += time.time()-curTime
            curTime = time.time()

            with torch.no_grad():
                action = ppo.act(obs)
                opp_action = opp_actor.sample(torch.from_numpy(opp_obs).to(device))[0]

            t_action += time.time()-curTime
            curTime = time.time()

            reward, dones = env.step(action,opp_action)
            t_envstep += time.time()-curTime
            curTime = time.time()

            ppo.step(value_obs=obs, rews=reward, dones=dones)
            t_ppostep += time.time()-curTime
            curTime = time.time()

            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)
            t_summing += time.time()-curTime
            curTime = time.time()

        # take st step to get value obs
        obs,opp_obs = env.observe()
        ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        average_ll_performance = reward_sum / total_steps
        average_dones = done_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))
        t_backprop = time.time() - curTime

        # curriculum update. Implement it in Environment.hpp
        env.curriculum_callback()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('----------------------------------------------------')
        print('{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|'.format('observe','action','envstep','ppostep','summing','backprop','etc.'))
        print('{:1.8f}|{:1.8f}|{:1.8f}|{:1.8f}|{:1.8f}|{:1.8f}|{:1.8f}|'.format(t_observe,t_action ,t_envstep,t_ppostep,t_summing,t_backprop,(end-start)-sum((t_observe,t_action,t_envstep,t_ppostep,t_summing,t_backprop))))
        print('----------------------------------------------------\n')


ppo_outer_loop()