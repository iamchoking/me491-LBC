from shutil import copyfile
import datetime
import os
import ntpath
import torch


class ConfigurationSaver:
    def __init__(self, log_dir, save_items):
        if log_dir == '/dev/null':
            print("[SAVER] log_dir set to destroy itself")
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                if base_file_name.split(".")[-1] == "yaml":
                    print("[SAVER] saved " + base_file_name + "as cfg.yaml")
                    copyfile(save_item, self._data_dir + '/' + "cfg.yaml")
                else:
                    copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir


def tensorboard_launcher(directory_path, open_browser=True):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    if open_browser:
        webbrowser.open_new(url)


def load_param(weight_path, env, actor, critic, optimizer, data_dir):
    if weight_path == "":
        print("[HELPER] Can't find the pre-trained weight, continuing with fresh model. (or, provide a pre-trained weight with --weight switch)")
    print("[HELPER] Retraining from the checkpoint:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
    var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
    items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp", weight_dir + "AnymalController_20190673.hpp"]

    copy_files(items_to_save,to_dir=data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1])

    # load observation scaling from files of pre-trained model
    env.load_scaling(weight_dir, iteration_number)

    # load actor and critic parameters from full checkpoint
    checkpoint = torch.load(weight_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def copy_files(items_to_save,to_dir):
    if items_to_save is None:
        return
    os.makedirs(to_dir)
    for item_to_save in items_to_save:
        copyfile(item_to_save, to_dir+'/'+item_to_save.rsplit('/', 1)[1])

def load_param_selfplay(weight_path, opp_weight_path, env, actor, critic, optimizer, data_dir,opp_actor):

    if opp_weight_path == "":
        raise Exception("\n[HELPER] Can't find the opponent weight, please provide a opponent weight with --oppweight switch\n")
    else:
        print("[HELPER] Loading opponent from checkpoint:", opp_weight_path+"\n")
        opp_iteration_number = opp_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
        opp_weight_dir = opp_weight_path.rsplit('/', 1)[0] + '/'

        opp_checkpoint = torch.load(opp_weight_path)
        opp_actor.architecture.load_state_dict(opp_checkpoint['actor_architecture_state_dict'])
        opp_actor.distribution.load_state_dict(opp_checkpoint['actor_distribution_state_dict'])

        items_to_save = [opp_weight_path, opp_weight_dir + 'mean' + opp_iteration_number + '.csv', opp_weight_dir + 'var' +opp_iteration_number+'.csv',opp_weight_dir + "Environment.hpp", opp_weight_dir + "AnymalController_20190673.hpp"]
        copy_files(items_to_save,to_dir= data_dir + '/opponent_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1])


    if weight_path == "":
        print("[HELPER] No Specified pre-trained weight for player. Proceeding with new network\n")
        iteration_number = 0
        weight_dir = ""
        items_to_save = None
    else:
        print("[HELPER] Retraining from the checkpoint:", weight_path+"\n")
        # BOOKMARK
        iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
        weight_dir = weight_path.rsplit('/', 1)[0] + '/'

        mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
        var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
        items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp", weight_dir + "AnymalController_20190673.hpp"]
        checkpoint = torch.load(weight_path)
        actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
        actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
        critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        copy_files(items_to_save,to_dir= data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1])

    # load observation scaling (ignores if directory is "")
    env.load_scaling(
        dir_name=weight_dir         , iteration=iteration_number,
        opp_dir_name=opp_weight_dir , opp_iteration=opp_iteration_number
    )

