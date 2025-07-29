import os
import math
import ghnn
import numpy as np
import argparse
import json

def generate_nn_paths_list(root_ghnn_dir, dataset, names, max_epochs, period_q):
    nn_paths_list = []
    for name in names:
        nn_paths = []
        path_to_model = os.path.join(root_ghnn_dir, 'outputs', 'NeuralNets_GHNN', 
                                     dataset, name)
        runs = os.listdir(path_to_model)
        for i, _ in enumerate(runs):
            path_to_run = os.path.join(root_ghnn_dir, 'outputs', 'NeuralNets_GHNN',
                                       dataset, name, "nn_{}".format(i))
            config_file = os.path.join(path_to_run, 'settings.json')
            with open(config_file, 'r') as f:
                config = json.load(f)
            if config["max_epochs"] == max_epochs and config["period_q"] == period_q:
                nn_paths.append(path_to_run)
        nn_paths_list.append(nn_paths)
    return nn_paths_list


def analyze(dataset, max_epochs, period_q):
    ghnn_dir = '/data2/users/lr4617/GHNN'
    results_dir = os.path.join(ghnn_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    if dataset.find("double")!=-1:
        store_name = 'doub_pend_all_runs.h5.1'
    else:
        store_name = 'pend_all_runs.h5.1'
    data_path = os.path.join('/data2/users/lr4617/data/GHNN', dataset)

    # best_names = ['MLP', 'SympNet', 'HenonNet', 'GHNN']
    best_names = ['SympNet', 'GHNN', 'GHNNLatentPhysics']

    figures_path = os.path.join(results_dir, f'Figures_{dataset}')
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    nn_paths_list = generate_nn_paths_list(ghnn_dir, dataset, best_names, max_epochs, period_q)

    ghnn.plotting.plot_data_mse_moments(data_path,
                                        store_name, 
                                        nn_paths_list, 
                                        'mean_var', 
                                        test='doub_pend_training.h5.1' if dataset.find("double")!=-1 else \
                                             'pend_training.h5.1', 
                                        period_q=period_q, 
                                        t_in_T=False, 
                                        save_name=os.path.join(figures_path, 
                                                  f'doub_pend_mse_comparison_p{period_q}_e{max_epochs}.png' \
                                                  if dataset.find("double")!=-1 else \
                                                  f'pend_mse_comparison_p{period_q}_e{max_epochs}.png')
                                        )
    results_list = os.listdir(figures_path)
    results_files = [f'pos_mse_{model_name}_m.npy' for model_name in best_names]
    pos_mses = {}
    for result in results_list:
        if result in results_files:
            result_arr = np.load(os.path.join(figures_path, result))
            model_name = result.split('_')[2]
            pos_mses[model_name] = np.sum(result_arr)

    best_model, best_score = min(pos_mses.items(), key=lambda kv: kv[1])
    
    # write to disk
    out_path = os.path.join(figures_path, 'best_model.txt')
    with open(out_path, 'w') as f:
        f.write(f"best_model: {best_model}\n")
        f.write(f"best_score: {best_score}\n")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--period_q", type=float, default=3.141592653589793)
    args = parser.parse_args()

    print('##############################################################################')
    print(f'Analyzing runs with period_q={args.period_q} and max_epochs={args.max_epochs}')
    print('##############################################################################')

    analyze(args.dataset, args.max_epochs, args.period_q)