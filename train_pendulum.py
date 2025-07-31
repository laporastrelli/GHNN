import os
import json
import ghnn
import argparse

def run(dataset, root_data_dir, period_q, max_epochs):

    # 1) LOGISTICS: create ouptut root directory
    root_nn_path = os.path.join('..', 'GHNN', 'outputs', 'NeuralNets_GHNN')
    if not os.path.exists(root_nn_path):
        os.mkdir(root_nn_path)

    # 1) LOGISTICS: create dataset specific ouput directory
    dataset_nn_path = os.path.join(root_nn_path, dataset)
    if not os.path.isdir(dataset_nn_path):
        os.mkdir(dataset_nn_path)

    # 1) LOGISTICS: get list of models to train
    nn_types = ['GHNN', 'SympNet', 'HenonNet', 'MLP']

    # 1) LOGISTICS: iterate over all models
    for nn_type in nn_types:

        print(f"Training {nn_type} on {dataset} .. ")

        # 1) LOGISTICS: create model-specific output directory
        nn_path = os.path.join(dataset_nn_path, nn_type)
        if not os.path.isdir(nn_path):
            os.mkdir(nn_path)

        # 2) JSON SETTINGS: load model-specifc default settings json file
        if nn_type == 'SympNet':
            with open(os.path.join('ghnn', 'training', 'default_G_SympNet.json')) as file_:
                settings = json.load(file_)
        elif nn_type == 'MLP_wsymp_2':
            with open(os.path.join('ghnn', 'training', 'default_MLP_wsymp.json')) as file_:
                settings = json.load(file_)
                settings['p_range'] = [-2.5, 2.5]
                settings['q_range'] = [-3, 3]
                settings['p_steps'] = 9
                settings['q_steps'] = 9
                settings['symp_lambda'] = 100
        else:
            with open(os.path.join('ghnn', 'training', f'default_{nn_type}.json')) as file_:
                settings = json.load(file_)
        
        # 2) JSON SETTINGS: update settings with dataset-specific parameters
        data_path = os.path.join(root_data_dir, dataset, 'pend_training.h5.1')
        settings['data_path'] = data_path
        del(settings['bodies'])
        del(settings['dims'])
        settings['feature_names'] = ['q_A','p_A']
        settings['label_names'] = ['q_A','p_A']
        settings['batch_size'] = 200
        settings['max_epochs'] = max_epochs
        if period_q == 0:
            period_q = None
        if dataset.find("half")!= -1 or dataset.find("DHN")!= -1:
            print("Using Half Pendulum!")
            settings['t_in_T'] = False
            if period_q is not None:
                settings['period_q'] = period_q/2
            else:
                settings['period_q'] = None
        else: 
            settings['t_in_T'] = False
            settings['period_q'] = period_q            
        
        # 3) LOGISTICS-2: create a new run directory
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
        
        # 3) LOGISTICS-2: create 5 runs with different seeds
        for i in range(start_idx, start_idx + 5):
            settings['seed'] = i
            run_path = os.path.join(nn_path, f'nn_{i}')
            if not os.path.exists(run_path):
                os.mkdir(run_path)
            else:
                raise FileExistsError(f"Run directory {run_path} already exists. This should not be possible. Please check your code.")
            with open(os.path.join(run_path, 'settings.json'), 'w') as file_:
                json.dump(settings, file_, indent=4, separators=(',', ': '))

            # 4) RUN TRAINING: run the training
            wd = os.getcwd()
            os.chdir(run_path)
            ghnn.training.train_from_folder()
            os.chdir(wd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--period_q", type=float, required=True,
                        default=3.141592653589793)
    args = parser.parse_args()

    root_data_dir = '/data2/users/lr4617/data/GHNN'
    
    run(args.dataset, root_data_dir, args.period_q, args.max_epochs)