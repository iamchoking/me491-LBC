from ruamel.yaml import YAML, dump, RoundTripDumper
from ME491_2023_project.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from ME491_2023_project.helper.raisim_gym_helper import (tensorboard_launcher, copy_files)
from ME491_2023_project.env.bin.rsg_anymal import NormalSampler
from ME491_2023_project.env.bin.rsg_anymal import RaisimGymEnv
from ME491_2023_project.env.RewardMetricAnalyzer import RewardMetricAnalyzer
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
import logging


class Game:
    def __init__(self,
                 cfg=None,
                 targ_path=None,
                 opp_path=None,
                 rules="vanilla",
                 reward="ffa",
                 stride="fine",
                 viz_only=False,
                 no_viz  =False,
                 new_name_bare=None,
                 logger=None
                 ):
        self.viz_only = viz_only
        self.task_dir = os.path.dirname(os.path.realpath(__file__))
        self.home_dir = self.task_dir.rsplit("/", 4)[0]
        self.league_dir = self.home_dir + "/ME491_2023_project/league"
        if cfg is None:
            print("[GAME] No [cfg.yaml] Provided! Using default cfg : ")
            self.cfg = YAML().load(open(self.league_dir + "/configs/vanilla.yaml", "r"))
        else:
            self.cfg = cfg
        if targ_path is None:
            print("[GAME] No Target Provided! Using default target: ")
            self.targ_path = self.league_dir + "/athletes/STABLE32/x0STABLE32_1200.pt"
        else:
            self.targ_path = targ_path
        if opp_path is None:
            print("[GAME] No Opponent Provided! Using default opponent: ")
            self.opp_path = self.league_dir + "/athletes/AGILE/x0AGILE_1020.pt"
        else:
            self.opp_path = opp_path

        self.timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if new_name_bare is None:
            self.targ_name = self.targ_path.rsplit('/', 1)[1].rsplit('_', 1)[0]
        else:
            self.targ_name = 'x0'+new_name_bare

        self.targ_dir = self.targ_path.rsplit('/', 1)[0]
        self.targ_iter = int(self.targ_path.rsplit('_', 1)[1].rsplit('.', 1)[0])

        self.opp_name = self.opp_path.rsplit('/', 1)[1].rsplit('_', 1)[0]
        self.opp_dir = self.opp_path.rsplit('/', 1)[0]
        self.opp_iter = int(self.opp_path.rsplit('_', 1)[1].rsplit('.', 1)[0])

        # number of updates for this game (constant difference btw self.targ_iter)
        self.update_num = 0

        self.name = self.targ_name + "_vs_" + self.opp_name + "__" + rules + "-" + reward + "-" + stride

        # create game save directory
        if viz_only:
            self.game_dir = self.league_dir + "/viz/" + self.name + "__" + self.timestr
        else:
            self.game_dir = self.league_dir + "/data/" + self.name + "__" + self.timestr  # [!!] this becomes "game_dir"
        os.makedirs(self.game_dir)

        # initialize logging
        if logger is None:
            self.logger = logging.getLogger(__name__)

            f_handler = logging.FileHandler(self.game_dir + '/log.txt')
            c_handler = logging.StreamHandler()

            self.logger.setLevel(logging.DEBUG)
            f_handler.setLevel(logging.DEBUG)
            c_handler.setLevel(logging.DEBUG)

            f_handler.setFormatter(logging.Formatter('[%(levelname)s::%(asctime)s] %(message)s'))
            c_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

            self.logger.addHandler(f_handler)
            self.logger.addHandler(c_handler)
        else:
            self.logger = logger
            self.logger.setLevel(logging.DEBUG)

            f_handler = logging.FileHandler(self.game_dir + '/log.txt')
            f_handler.setFormatter(logging.Formatter('[%(levelname)s::%(asctime)s] %(message)s'))
            self.logger.addHandler(f_handler)


        self.targ_ver = 00
        try:
            self.targ_ver = int(self.targ_name[1])
        except ValueError:
            self.logger.warning("Couldn't Extract Version Number from target name: " + self.targ_name)

        # <PRE-GAME Routine>: Save and Log details
        # [!!] not using configuration saver.
        # save details for the game

        # target and opponent networks + submission things
        # items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir
        # + "Environment.hpp", weight_dir + "AnymalController_20190673.hpp"]
        targ_mean_path = self.targ_dir + "/mean" + str(self.targ_iter) + ".csv"
        targ_var_path = self.targ_dir + "/var" + str(self.targ_iter) + ".csv"
        opp_mean_path = self.opp_dir + "/mean" + str(self.opp_iter) + ".csv"
        opp_var_path = self.opp_dir + "/var" + str(self.opp_iter) + ".csv"

        copy_files([self.targ_path, targ_mean_path, targ_var_path, self.task_dir + "/Environment.hpp",
                    self.task_dir + "/AnymalController_20190673.hpp"], self.game_dir + "/target_" + self.timestr)
        copy_files([self.opp_path, opp_mean_path, opp_var_path, self.task_dir + "/Environment.hpp",
                    self.task_dir + "/AnymalController_20190673.hpp"], self.game_dir + "/opponent_" + self.timestr)

        # config file(s) (consolitated into one [_cfg.yaml])
        YAML().dump(self.cfg, open(self.game_dir + "/_cfg.yaml", "w"))
        copy_files([self.task_dir + "/league_game.py", self.task_dir + "/Environment.hpp",
                    self.task_dir + "/AnymalController_20190673.hpp"], self.game_dir)

        # <PRE-GAME Routine> preliminaries done!
        self.logger.info("[READY] Preprocessing COMPLETE")

        # create cpp simulation environment
        if no_viz:
            self.cfg['environment']['render'] = False

        self.env = VecEnv(RaisimGymEnv(self.home_dir + "/rsc", dump(self.cfg['environment'], Dumper=RoundTripDumper)))
        self.env.seed(self.cfg['seed'])

        # shortcuts
        self.targ_ob_dim = self.env.num_obs
        self.targ_act_dim = self.env.num_acts  # stored and used later for saving
        opp_ob_dim = self.env.opp_num_obs
        opp_act_dim = self.env.opp_num_acts

        # base params
        self.num_threads = self.cfg['environment']['num_threads']
        self.n_steps = math.floor(self.cfg['environment']['max_time'] / self.cfg['environment']['control_dt'])
        self.total_steps = self.n_steps * self.env.num_envs

        # create torch objects and load
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create target dummy
        targ_p_mlp = ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, self.targ_ob_dim,
                                    self.targ_act_dim)
        targ_p_dist = ppo_module.MultivariateGaussianDiagonalCovariance(self.targ_act_dim, self.env.num_envs, 5.0,
                                                                        NormalSampler(self.targ_act_dim),
                                                                        self.cfg['seed'])
        targ_actor = ppo_module.Actor(targ_p_mlp, targ_p_dist, self.device)

        targ_vf_mlp = ppo_module.MLP(self.cfg['architecture']['value_net'], nn.LeakyReLU, self.targ_ob_dim, 1)
        targ_critic = ppo_module.Critic(targ_vf_mlp, self.device)

        if self.cfg['learning']['kl'] < 0:
            self.cfg['learning']['kl'] = 0.01
        # load target (only put important values in self!)
        self.targ_ppo = PPO.PPO(actor=targ_actor,
                                critic=targ_critic,
                                num_envs=self.cfg['environment']['num_envs'],
                                num_transitions_per_env=self.n_steps,
                                num_learning_epochs=4,
                                gamma=0.995,  # [!!] changed from .998 to .995
                                lam=0.95,
                                num_mini_batches=4,
                                device=self.device,
                                log_dir=self.game_dir,
                                shuffle_batch=False,
                                desired_kl=self.cfg['learning']['kl'],
                                )

        self.logger.info("[PPO] Desired KL Divergence is " + str(self.cfg['learning']['kl']))

        # load the ppo from path
        checkpoint = torch.load(self.targ_path)
        self.targ_ppo.actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
        self.targ_ppo.actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
        self.targ_ppo.critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
        self.targ_ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # reward / metric analysis tool (for writing to tensorboard)
        self.analyzer = RewardMetricAnalyzer(self.env, self.targ_ppo.writer)

        # create opponent dummy
        self.opp_mlp = ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, opp_ob_dim, opp_act_dim)

        # load opponent
        self.opp_mlp.load_state_dict(torch.load(self.opp_path)['actor_architecture_state_dict'])

        # load observation scaling from files of pre-trained model
        self.env.load_scaling(dir_name=self.targ_dir, iteration=self.targ_iter, opp_dir_name=self.opp_dir,
                              opp_iteration=self.opp_iter)

        # tensorboard_launcher(self.game_dir+"/..", False)

        if self.cfg['learning']['lr'] > 0:  # reset the optimizer
            self.logger.info("[PPO] Replacing Optimizer With Learning Rate: " + str(self.cfg['learning']['lr']))
            # self.logger.info("[PPO] Previous State Dict : " + str(ppo.optimizer.state_dict()))
            self.targ_ppo.optimizer.load_state_dict(
                torch.optim.Adam([*self.targ_ppo.actor.parameters(), *self.targ_ppo.critic.parameters()],
                                 lr=self.cfg['learning']['lr']).state_dict())
            # self.logger.info("[PPO] New State Dict" + str(targ_ppo.optimizer.state_dict()))

        self.targ_ppo.learning_rate = self.targ_ppo.optimizer.param_groups[0]['lr']
        # !! needed to negate leftover values!

        self.env.turn_off_visualization()

        self.logger.info("[GAME] Game for " + self.targ_name + "(version" + str(self.targ_ver) + ") ready!")
        self.logger.info("     Opponent: " + self.opp_path)
        self.logger.info("     Config : \n" + str(self.cfg))
        self.logger.info("[GAME] Here we go!!")

    def save_target(self, to_dir=None, name=None):
        if to_dir is None:
            to_dir = self.game_dir
        if name is None:
            name = self.targ_name
        self.logger.info("[SAVE] Saving the current game : " + self.name)

        dest_path = to_dir + "/" + name + "_" + str(self.targ_iter) + '.pt'

        torch.save({
            'actor_architecture_state_dict': self.targ_ppo.actor.architecture.state_dict(),
            'actor_distribution_state_dict': self.targ_ppo.actor.distribution.state_dict(),
            'critic_architecture_state_dict': self.targ_ppo.critic.architecture.state_dict(),
            'optimizer_state_dict': self.targ_ppo.optimizer.state_dict(),
        }, dest_path)
        self.env.save_scaling(to_dir, str(self.targ_iter))

        self.logger.info("[SAVE] Saved current target network(s) at : " + dest_path)

        return dest_path

    def evaluate(self, eval_steps=-1, visualize=True):
        if eval_steps == -1:
            eval_steps = self.n_steps

        save_path = self.save_target()
        # to check save/load, create the evaluation graph from saved file
        loaded_graph = ppo_module.MLP(self.cfg['architecture']['policy_net'], nn.LeakyReLU, self.targ_ob_dim,
                                      self.targ_act_dim)
        loaded_graph.load_state_dict(torch.load(save_path)['actor_architecture_state_dict'])

        if visualize:
            self.logger.info("[EVAL-VIZ] Visualization ON")
            self.env.turn_on_visualization()
            if self.viz_only:
                self.env.start_video_recording(self.timestr + "_" + self.name + '_VIZ.mp4')
            else:
                self.env.start_video_recording(self.timestr + "_" + self.name + "_update-" + str(self.update_num) + '.mp4')

        for frame in range(eval_steps):  # visualization (dry) run
            with torch.no_grad():
                frame_start = time.time()
                obs, opp_obs = self.env.observe(False)
                # self.logger.info(obs[0]) # CHECK OBS HERE

                action = loaded_graph.architecture(torch.from_numpy(obs).cpu()).cpu().detach().numpy()
                opp_action = self.opp_mlp.architecture(torch.from_numpy(opp_obs).cpu()).cpu().detach().numpy()

                # print("Actor obs:")
                # print(obs)
                # print("Opponent obs:")
                # print(opp_obs)
                # return

                __, __ = self.env.step(action, opp_action)
                self.analyzer.add_reward_info(self.env.get_reward_info())
                frame_end = time.time()

                if visualize:
                    wait_time = self.cfg['environment']['control_dt'] - (frame_end - frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

        if visualize:
            self.env.stop_video_recording()
            self.logger.info("[EVAL-SAVE] Visualization Complete")
            self.env.turn_off_visualization()

        self.analyzer.analyze_and_plot(self.update_num)
        self.env.reset()
        self.logger.info(
            '[EVAL-SAVE] Update #' + str(self.update_num) + '( Iteration #' + str(self.targ_iter) + ') Saved.')

    def ppo_outer_loop(self, updates=5000, do_viz=True):

        # flow: update (with periodic evals) -> finish
        if not do_viz:
            self.logger.warning("[GAME] This game was called with no visualization!!")

        for update in range(updates): # TODO implement (keyboard interrupt)

            do_update = update % self.cfg['environment']['eval_every_n'] == 0 and update != 0
            if do_update or self.cfg['environment']['eval_every_n'] == 1:
                self.evaluate(visualize=do_viz)
                self.logger.info("[GAME-OUTER] Evaluation Complete. Starting Update # " + str(self.update_num))

            self.ppo_training_update()

            self.update_num += 1
            self.targ_iter += 1

            # stopping conditions
            metrics = self.env.get_metrics()
            if metrics['100_win'] > 0.8:
                self.logger.info("[GAME-OUTER][WIN] Game session finished (win-rate exceeds 0.8)")
                break
            # too much gap in power
            # do at least 50 though...
            elif metrics['score'] > 50 and self.update_num < 100 and self.update_num > 50:
                self.logger.info("[GAME-OUTER][TRIVIAL] Game Session finished (score quickly exceeds 50)")
                break
            if update == updates - 1:
                self.logger.info("[GAME-OUTER][FINISH] Game session finished (max-updates reached)")
                break

        fin_path = self.finish()
        self.logger.info("[GAME-FINISH] Game Concluded. Final result saved at ["+fin_path+"]")
        return fin_path

    # one update step for PPO training
    def ppo_training_update(self):
        device = self.device

        start = time.time()
        cur_time = start

        n_steps = self.n_steps
        total_steps = self.env.num_envs * n_steps

        t_observe = 0
        t_action = 0
        t_envstep = 0
        t_ppostep = 0
        t_summing = 0
        # t_backprop = 0

        reward_sum = 0
        done_sum = 0
        # average_dones = 0.

        env = self.env
        ppo = self.targ_ppo
        opp_mlp = self.opp_mlp

        # actual training
        for step in range(n_steps):
            obs, opp_obs = env.observe()
            t_observe += time.time() - cur_time
            cur_time = time.time()

            action = ppo.act(obs)
            opp_action = opp_mlp.architecture(torch.from_numpy(opp_obs).cpu()).detach().numpy()
            # opp_action = opp_mlp.architecture(torch.from_numpy(opp_obs).cpu()).cpu().detach().numpy()

            t_action += time.time() - cur_time
            cur_time = time.time()

            reward, dones = env.step(action, opp_action)
            t_envstep += time.time() - cur_time
            cur_time = time.time()

            ppo.step(value_obs=obs, rews=reward, dones=dones)
            t_ppostep += time.time() - cur_time
            cur_time = time.time()

            done_sum = done_sum + np.sum(dones)
            reward_sum = reward_sum + np.sum(reward)
            t_summing += time.time() - cur_time
            cur_time = time.time()

        # take st step to get value obs
        obs, opp_obs = self.env.observe()
        # ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
        ppo.update(actor_obs=obs, value_obs=obs,
                   log_this_iteration=(self.update_num % self.cfg['environment']['plot_metric_n'] == 0),
                   update=self.update_num)
        average_ll_performance = reward_sum / total_steps
        average_dones = done_sum / total_steps

        ppo.actor.update()
        env.curriculum_callback()

        ppo.actor.distribution.enforce_minimum_std((torch.ones(12) * 0.2).to(device))
        t_backprop = time.time() - cur_time

        # logging additional metrics
        if self.update_num % self.cfg['environment']['plot_metric_n'] == 0:
            self.analyzer.update_metrics(self.env, {
                "average_reward": average_ll_performance,
                "average_T_per_episode": average_dones * total_steps / self.env.num_envs,
            })
            self.analyzer.plot_metrics(self.env, self.update_num)
            self.logger.info("[PLOT] Metrics Plotted")

        end = time.time()

        self.logger.info('[UPDATE-#{:>6}] {:>30}'.format(self.update_num, self.game_dir.split('/')[-1]))
        self.logger.info('<Learning>')
        self.logger.info('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        self.logger.info(
            '{:<40} {:>6}'.format("dones per self.env: ",
                                  '{:0.6f}'.format(average_dones * total_steps / self.env.num_envs)))
        self.logger.info('{:<40} {:>6}'.format("learning rate: ",
                                               '{:0.6f}'.format(ppo.optimizer.state_dict()['param_groups'][0]['lr'])))
        self.logger.info('{:<40} {:>6}'.format("kl stride: ", '{:0.6f}'.format(ppo.kl)))
        self.logger.info('')
        self.logger.info('<Performance>')
        self.logger.info('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        self.logger.info('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        self.logger.info('{:<40} {:>6}'
                         .format("real time factor: ",
                                 '{:6.0f}'.format(total_steps / (end - start) * self.cfg['environment']['control_dt'])))
        self.logger.info('--------------------------------------------------------')
        self.logger.info('{:<7}|{:<7}|{:<7}|{:<7}|{:<7}|{:<7}|{:<7}|'
                         .format('observe', 'action', 'envstep', 'ppostep', 'sum', 'grad', 'etc.'))
        self.logger.info('{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|'
                         .format(t_observe, t_action, t_envstep, t_ppostep, t_summing, t_backprop,
                                 (end - start) - sum(
                                     (t_observe, t_action, t_envstep, t_ppostep, t_summing, t_backprop))))
        self.logger.info('--------------------------------------------------------\n')

    # save the current version in the database
    def finish(self):
        # plot metrics for the final point
        self.analyzer.plot_metrics(self.env, self.update_num)
        if self.env.get_metrics()['100_win'] < 0.5:
            self.logger.warning('Final winrate does not exceed half after finish (may be non-optimal)!')

        if self.update_num < 3:
            self.logger.warning("Too few updates. This game will not be recorded as a version.")
            return self.targ_path
        name_bare = self.targ_name[2:]
        finish_dir = self.league_dir + "/athletes/" + name_bare

        # save to game_dir just in case
        self.save_target()

        final_path = self.save_target(finish_dir,'x'+str(self.targ_ver+1) + name_bare)

        copy_files([self.task_dir + "/league_game.py", self.task_dir + "/Environment.hpp",
                    self.task_dir + "/AnymalController_20190673.hpp"], finish_dir)

        # TODO placholder
        self.cfg['training record'] = self.timestr

        final_cfg_path = finish_dir+"/cfg_x"+ str(self.targ_ver) + "-x" + str(self.targ_ver+1) + "_" + self.name + ".yaml"
        YAML().dump(self.cfg,open(final_cfg_path,'w'))

        # TODO do some logging and stuff (profile.yaml and stuff)

        self.targ_ver += 1

        return final_path

    def end(self):
        self.logger.handlers = []
        self.env.close()

if __name__ == "__main__":  # TODO parse args and start training (+ catch keyboard interrupts)
    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--target', help='target weight path', type=str, default='')
    parser.add_argument('-y', '--opponent', help='path for opponent weight', type=str, default='')
    parser.add_argument('-l', '--lr', help='prescribed starting learning rate', type=float, default=-1)
    parser.add_argument('-k', '--kl', help='desired kl divergence (how much stride is desired)', type=float, default=-1)
    parser.add_argument('-c', '--config', help='config file path', type=str, default='')
    parser.add_argument('-v', '--visualize', help='just visualize without training', type=bool, default=False)
    parser.add_argument('-n', '--newname', help='come up with a new name for an athlete',type=str,default='')
    parser.add_argument('-u','--updates', help='max number of updates',type=int,default=750)
    args = parser.parse_args()

    main_targ_path = args.target
    main_opp_path = args.opponent
    main_starting_lr = args.lr
    main_cfg_path = args.config
    main_prescribed_kl = args.kl
    main_do_visualize = args.visualize
    main_newname = args.newname
    main_updates = args.updates

    if main_cfg_path == '':
        main_cfg_path = "ME491_2023_project/league/default_cfg.yaml"
        print("[GAME-MAIN] No cfg provided. loading from : " + main_cfg_path)
    main_cfg = YAML().load(open(main_cfg_path, 'r'))
    if main_prescribed_kl > 0:
        main_cfg['learning']['kl'] = main_prescribed_kl
    if main_starting_lr > 0:
        main_cfg['learning']['lr'] = main_starting_lr

    if main_do_visualize:
        main_cfg['environment']['num_envs'] = 1
        main_cfg['environment']['eval_every_n'] = 1
        main_cfg['environment']['max_time'] = 100
        print('[VIS-ONLY] Simple Visualization Mode')
        game = Game(main_cfg, main_targ_path, main_opp_path, 'main', 'exec', 'game', True)
        game.evaluate()
    else:
        # adding a new name
        if main_newname != '':
            game = Game(main_cfg, main_targ_path, main_opp_path, 'main', 'exec', 'game',new_name_bare=main_newname)
        else:
            game = Game(main_cfg, main_targ_path, main_opp_path, 'main', 'exec', 'game')
        # given my experience, 750 iters is a good place to stop (if you're poor on resources)
        game.ppo_outer_loop(main_updates)


