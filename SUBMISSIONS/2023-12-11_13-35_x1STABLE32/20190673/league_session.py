import os

from league_game import *
import random
import copy

# run this in home path!
league_dir = "ME491_2023_project/league"
athlete_dir = league_dir + "/athletes"
config_dir = league_dir + "/configs/preset"
types = ['BABY', 'STABLE32', 'PASSIVE', 'AGILE']

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


def session(unit_updates=750, viz_duration=30, given_target = None,given_opponent=None,given_config=None):

    # choose target
    if given_target is not None:
        target_type = given_target

    else:
        target_type = random.choice(types)

    target_dir = athlete_dir + "/" + target_type
    possible_targets = list(filter(lambda path: path[-3:] == ".pt", os.listdir(target_dir)))
    possible_targets.sort(reverse=True)

    if len(possible_targets) > 10:
        session_logger.error("Something went wrong. Terminating...")
        raise ValueError("Something is Wrong... There are too many iterations")
    target_name = possible_targets[0]

    # choose opponent
    if given_opponent is not None:
        opponent_type = given_opponent
        opponent_dir = athlete_dir + "/" + opponent_type

    else:
        opponent_type = random.choice(types)
        opponent_dir = athlete_dir + "/" + opponent_type
        while (target_type == "BABY") and (opponent_type == "BABY"):
            opponent_type = random.choice(types)

    possible_opponents = list(filter(lambda path: path[-3:] == ".pt", os.listdir(opponent_dir)))
    possible_opponents.sort(reverse=True)

    if target_type == opponent_type:
        opponent_name = possible_opponents[1]
    else:
        opponent_name = possible_opponents[0]

    # choose config
    if given_config is not None:
        config_name = given_config
    else:
        config_name = random.choice(os.listdir(config_dir))


    target_version = int(target_name[1])
    opponent_version = int(opponent_name[1])

    # generate paths
    opponent_path = opponent_dir + "/" + opponent_name
    target_path = target_dir + "/" + target_name
    config_path = config_dir + "/" + config_name

    config = YAML().load(open(config_path, 'r'))

    # print(target_path)
    # print(target_version)
    # print(opponent_path)
    # print(opponent_version)
    # print(config_path)
    # print(config)

    session_logger.info("[SESSION-START] TARGET: " + target_name + " OPPONENT: " + opponent_name + " CONFIG: " +
                        config_path.rsplit('/', 1)[1])

    session_logger.info("[SESSION] Visualizing Session --- ")
    viz_config = copy.deepcopy(config)
    viz_config['environment']['num_envs'] = 1
    viz_config['environment']['eval_every_n'] = 1
    viz_config['environment']['max_time'] = viz_duration
    pre_viz_game = Game(viz_config, target_path, opponent_path, config['rules'], config['reward'], config['stride'],
                        viz_only=True)
    pre_viz_game.evaluate(visualize=True)
    pre_viz_game.end()

    session_logger.info("[SESSION] Starting Learning Session --- ")
    learning_game = Game(config, target_path, opponent_path, config['rules'], config['reward'], config['stride'])
    new_target_path = "NONE"
    try:
        new_target_path = learning_game.ppo_outer_loop(unit_updates, do_viz=False)
        session_logger.info("[SESSION] Doinng Final Evaluation")
        learning_game.evaluate(visualize=False)
        session_logger.info("[SESSION] Learning Complete")

    except KeyboardInterrupt:
        session_logger.warning("[SESSION] Keyboard Interrupt Detected! Publishing Data Now...")
        new_target_path = learning_game.finish()
    learning_game.end()

    session_logger.info("[SESSION] Learning Finished after " + str(learning_game.update_num) + " updates")

    post_viz_game = Game(viz_config, new_target_path, opponent_path, config['rules'], config['reward'],
                         config['stride'], viz_only=True)
    post_viz_game.evaluate(visualize=True)
    post_viz_game.end()
    session_logger.info("[SESSION] Learning step for <" + new_target_path + "> complete.")

    session_logger.info("[SESSION] STAGE COMPLETE")


if __name__ == "__main__":
    prescribed_routine = [
        ['AGILE','STABLE32','ffa-fine.yaml',300],
        ['PASSIVE','STABLE32','runaway-fine.yaml',300],
        ['STABLE32','AGILE','ffa-fine.yaml',300],
        ['STABLE32','PASSIVE','standard-fine.yaml',750],
        ['STABLE32','STABLE32','ffa-fine-yaml',500],
    ]

    while True:
        # random training
        # for type in types:
        #     session(unit_updates=750,viz_duration=20,given_target=type)

        # precribed training
        for [target,opponent,config,updates] in prescribed_routine:
            session(updates,20,target,opponent,config)
