import os
import math
import ghnn

def generate_nn_paths_list(root_ghnn_dir, dataset_str, names):
    nn_paths_list = []
    for name in names:
        nn_paths = []
        for i in range(1,3):
            if dataset_str.find("DHN")!= -1:
                print(f"Loading model trained on Data_{dataset_str} ...")
                nn_paths.append(os.path.join(root_ghnn_dir, 'outputs', 'NeuralNets_GHNN', 
                                             f'Data_{dataset_str}', name, f'nn_{i}'))
            else:
                nn_paths.append(os.path.join(root_ghnn_dir, 'outputs', 'NeuralNets_GHNN', 
                                             f'{dataset_str}_pendulum', name, f'nn_{i}'))
        nn_paths_list.append(nn_paths)
    return nn_paths_list

if __name__ == '__main__':
    ghnn_dir = '/data2/users/lr4617/GHNN'
    results_dir = os.path.join(ghnn_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    which = 'half_rand'
    store_name = 'pend_all_runs.h5.1'
    data_path = os.path.join('/data2/users/lr4617/data/GHNN', 'Data_' + which)

    all_names = ['MLP', 'MLP_wsymp', 'MLP_wsymp_2', 'SympNet', 
                 'HenonNet', 'double_HenonNet', 'GHNN']
    best_names = ['MLP', 'SympNet', 'HenonNet', 'double_HenonNet', 'GHNN']

    figures_path = os.path.join(results_dir, 'Figures_' + which)
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    nn_paths_list = generate_nn_paths_list(ghnn_dir, which, best_names)
    ghnn.plotting.plot_data_mae_moments(data_path, 
                                        store_name, 
                                        nn_paths_list, 
                                        'mean_var', 
                                        test='pend_training.h5.1', 
                                        period_q=math.pi, 
                                        t_in_T=False, 
                                        save_name=os.path.join(figures_path, 'pend_mae_comparison.png'))
    ghnn.plotting.plot_data_mse_moments(data_path,
                                        store_name, 
                                        nn_paths_list, 
                                        'mean_var', 
                                        test='pend_training.h5.1', 
                                        period_q=math.pi, 
                                        t_in_T=False, 
                                        save_name=os.path.join(figures_path, 'pend_mse_comparison.png'))

    kwargs = {'energy': True, 'mse': True, 'phase_space': True, 'period_q': math.pi}
    nn_paths = [nn_path[0] for nn_path in nn_paths_list]
    ghnn.plotting.predict_pendulum(data_path, 
                                   store_name, 
                                   6, 
                                   nn_paths, 
                                   save_name=os.path.join(figures_path, 'pend_plots_comparison.png'), 
                                   **kwargs)


    """Other possible things to do:
        >>> num_runs = 500
        >>> ghnn.plotting.plot_loss_moments(nn_paths_list, 'mean_var')
        >>> ghnn.plotting.plot_data_mae_moments(data_path, store_name, nn_paths_list, 'mean_var', period_q=math.pi, t_in_T=True)

        >>> ghnn.plotting.plot_loss(nn_paths)
        >>> ghnn.plotting.plot_data_mae(data_path, store_name, nn_paths, max_time=max_time, period_q=math.pi, t_in_T=True)

        >>> ghnn.plotting.predict_pendulum_rand(data_path, store_name, num_runs, nn_paths[0], **kwargs)
        >>> ghnn.plotting.predict_pendulum_rand(data_path, store_name, num_runs, nn_paths, **kwargs)
    """
