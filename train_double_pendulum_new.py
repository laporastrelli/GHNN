import os
import json
import ghnn
import argparse

def run(root_data_dir, max_epochs, dataset="Data_double_pendulum"):

    # create ouptut root directory
    root_nn_path = os.path.join('..', 'GHNN', 'outputs', 'NeuralNets_GHNN')
    if not os.path.exists(root_nn_path):
        os.mkdir(root_nn_path)

    # create dataset specific ouput directory
    dataset_nn_path = os.path.join(root_nn_path, dataset)
    if not os.path.isdir(dataset_nn_path):
        os.mkdir(dataset_nn_path)

    # get list of models to train
    nn_types = ['GHNN', 'SympNet', 'HenonNet', 'MLP']

    # iterate over all models
    for nn_type in nn_types:

        print(f"Training {nn_type} on {dataset} .. ")

        # create model-specific output directory
        nn_path = os.path.join(dataset_nn_path, nn_type)
        if not os.path.isdir(nn_path):
            os.mkdir(nn_path)

        # load model-specifc default settings json file
        if nn_type == 'SympNet':
            with open(os.path.join('ghnn', 'training', 'default_G_SympNet.json')) as file_:
                settings = json.load(file_)
        else:
            with open(os.path.join('ghnn', 'training', f'default_{nn_type}.json')) as file_:
                settings = json.load(file_)

        data_path = os.path.join(root_data_dir, dataset, 'doub_pend_training.h5.1')
        settings['data_path'] = data_path
        del(settings['bodies'])
        del(settings['dims'])
        settings['feature_names'] = ['q_A', 'q_B','p_A', 'p_B']
        settings['label_names'] = ['q_A', 'q_B','p_A', 'p_B']
        settings['batch_size'] = 200
        settings['max_epochs'] = max_epochs
        settings['period_q'] = 3.141592653589793

        run_ints = []
        existing_runs = os.listdir(os.path.join(nn_path))
        for existing_run in existing_runs:
            run_int = int(existing_run.split('_')[1])
            run_ints.append(run_int)
        if len(run_ints) == 0:
            start_idx = 0
            print(f'There are no existing runs in the folder. Starting from run {start_idx}!')
        else:
            start_idx = max(run_ints) + 1
            print(f'There are existing runs in the folder. Starting from run {start_idx}!')

        for i in range(start_idx, start_idx + 5):
            settings['seed'] = i
            run_path = os.path.join(nn_path, f'nn_{i}')
            if not os.path.exists(run_path):
                os.mkdir(run_path)
            else:
                raise FileExistsError(f"Run directory {run_path} already exists. This should not be possible. Please check your code.")
            with open(os.path.join(run_path, 'settings.json'), 'w') as file_:
                json.dump(settings, file_, indent=4, separators=(',', ': '))

            wd = os.getcwd()
            os.chdir(run_path)
            ghnn.training.train_from_folder()
            os.chdir(wd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, required=True)
    args = parser.parse_args()

    root_data_dir = '/data2/users/lr4617/data/GHNN'
    
    run(root_data_dir, args.max_epochs)
