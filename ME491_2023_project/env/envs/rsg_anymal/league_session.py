import os

from league_game import *
import random
import copy

# run this in home path!
league_dir = "ME491_2023_project/league"
athlete_dir = league_dir + "/athletes"
config_dir = league_dir + "/configs/preset"
types = ['STABLE32', 'BABY', 'PASSIVE', 'AGILE']

# initialize logging
session_logger = logging.getLogger(__name__)

f_handler = logging.FileHandler(league_dir + '/data/SESSION.txt')
c_handler = logging.StreamHandler()

session_logger.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_handler.setLevel(logging.DEBUG)

f_handler.setFormatter(logging.Formatter('[%(levelname)s::%(asctime)s] %(message)s'))
c_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

session_logger.addHandler(f_handler)
session_logger.addHandler(c_handler)


def session(unit_updates=750, viz_duration=30, mock=False):
    for target_type in types:
        opponent_type = random.choice(types)
        opponent_dir = athlete_dir + "/" + opponent_type
        target_dir = athlete_dir + "/" + target_type

        possible_targets = list(filter(lambda path: path[-3:] == ".pt", os.listdir(target_dir)))
        possible_targets.sort(reverse=True)

        possible_opponents = list(filter(lambda path: path[-3:] == ".pt", os.listdir(opponent_dir)))
        possible_opponents.sort(reverse=True)

        if (len(possible_targets) > 10):
            session_logger.error("Something went wrong. Terminating...")
            raise ValueError("Something is Wrong... There are too many iterations")
        target_name = possible_targets[0]
        target_path = target_dir + "/" + target_name

        if target_type == opponent_type:
            opponent_name = possible_opponents[1]
        else:
            opponent_name = possible_opponents[0]

        opponent_path = opponent_dir + "/" + opponent_name

        target_version = int(target_name[1])
        opponent_version = int(opponent_name[1])

        config_path = config_dir + "/" + random.choice(os.listdir(config_dir))
        config = YAML().load(open(config_path, 'r'))

        # print(target_path)
        # print(target_version)
        # print(opponent_path)
        # print(opponent_version)
        # print(config_path)
        # print(config)

        session_logger.info("[SESSION-START] TARGET: " + target_name + " OPPONENT: " + opponent_name + " CONFIG: " +
                            config_path.rsplit('/', 1)[0])

        session_logger.info("[SESSION] Visualizing Session --- ")
        viz_config = copy.deepcopy(config)
        viz_config['environment']['num_envs'] = 1
        viz_config['environment']['eval_every_n'] = 1
        viz_config['environment']['max_time'] = viz_duration
        pre_viz_game = Game(viz_config, target_path, opponent_path, config['rules'], config['reward'], config['stride'],
                            viz_only=True)
        pre_viz_game.evaluate(visualize=True)

        session_logger.info("[SESSION] Starting Learning Session --- ")
        learning_game = Game(config, target_path, opponent_path, config['rules'], config['reward'], config['stride'])
        new_target_path = "NONE"
        try:
            new_target_path = learning_game.ppo_outer_loop(unit_updates, do_viz=False)
            learning_game.evaluate()
        except KeyboardInterrupt:
            session_logger.warning("[SESSION] Keyboard Interrupt Detected! Publishing Data Now...")
            learning_game.finish()
        session_logger.info("[SESSION] Learning Finished after " + str(learning_game.update_num) + " updates")

        session_logger.info("[SESSION] New model created: ["  + new_target_path + "]")

        post_viz_game = Game(viz_config, new_target_path, opponent_path, config['rules'], config['reward'],
                             config['stride'], viz_only=True)
        post_viz_game.evaluate(visualize=True)

        session_logger.info("[SESSION] Learning step from " + target_name + " to " + new_target_name + " complete.")

    session_logger.info("[SESSION] ROUND COMPLETE")


if __name__ == "__main__":
    while True:
        session(unit_updates=10, mock=True)
