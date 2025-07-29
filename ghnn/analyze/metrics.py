"""Functions to calculate different metrics of predictions."""
import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ghnn.nets.helpers import net_from_dir, predict_trajectories

__all__ = ['calculate_error', 'dataset_mse', 'dataset_mae', 'dataset_mse_pos']

def calculate_error(data, predictions, feature_names, interp='linear'):
    """Calulcates the error at each timestep for one trajectory using interpolation for the data.

    Args:
        data (pd.DataFrame): Data of the trajectory (including time).
        predictions (pd.DataFrame): Predcitions from an NN (including time).
        feature_names (str[]): Names of the features of the NN.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.

    Returns:
        pd.DataFrame: The error at each timestep for all features.
    """
    f = interp1d(data['time'].values, data[feature_names].values, axis=0, kind=interp, fill_value='extrapolate')
    error = f(predictions['time'].values) - predictions[feature_names].values
    return pd.DataFrame(error, columns=feature_names)

def dataset_mse(data_path, store_name, nn_path, max_time=None, period_q=None, t_in_T=False, interp='linear', test=None):
    """Calulcates the mean of the mean square error at each timestep for all trajectories in a dataset.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_path (str): Path to where the NN is saved.
        max_time (float): Maximal time until when to calculate the error.
        period_q (float): If not None inputs are mapped to [-period_q, period_q].
        t_in_T (bool): Whether to use one period as max time.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        test (str, path-like object): HDF5 training store where the test data can be found.
          If None all data is used.

    Returns:
        pd.DataFrame: Times and mean of the MSE at each timestep.
    """
    settings = {'period_q': period_q, 
                'interp': interp, 
                'test': test, 
                't_in_T': t_in_T}
    if not os.path.isdir(os.path.join(nn_path, 'metrics')):
        os.makedirs(os.path.join(nn_path, 'metrics'))

    '''
    if os.path.isfile(os.path.join(nn_path, 'metrics', 'mse.txt')):
        with open(os.path.join(nn_path, 'metrics', 'mse.txt'), 'r') as file_:
            lines = file_.readlines()
        cont = [json.loads(line[12:-1]) for line in lines if line[:10]=='# Settings']
        if settings in cont:
            result = np.loadtxt(os.path.join(nn_path, 'metrics', 'mse.txt'))
            i = cont.index(settings)
            result = pd.DataFrame(result[i*2:i*2+2].T, columns=['time', 'mse'])
            if max_time != None:
                result = result[result['time'] <= max_time]
            return result
    '''

    my_net = net_from_dir(nn_path)
    data = pd.read_hdf(os.path.join(data_path, store_name), '/all_runs')
    if test:
        test_runs = pd.read_hdf(os.path.join(data_path, test), '/test_features')
        test_runs = test_runs['run'].unique()
        data = data[np.isin(data.index.get_level_values(0), test_runs)]
    inp = data.xs(0, level='timestep')

    runs = data.index.get_level_values(0).unique()
    max_time_data = [0.]*len(runs)
    for i, run in enumerate(runs):
        max_time_data[i] = data.loc[run].iloc[-1]['time']

    predictions = predict_trajectories(my_net, 
                                       inp, 
                                       max_time_data, 
                                       period_q=period_q)

    if period_q != None:
        q = my_net.get_pos_features()
        data[q] = (data[q] + period_q) % (2 * period_q) - period_q

    if t_in_T:
        result = pd.DataFrame(np.linspace(0, 1, 101), columns=['time'])
        mean = np.zeros(101)
    else:
        ind = max_time_data.index(max(max_time_data))
        result = pd.DataFrame(predictions.loc[runs[ind]]['time'])
        mean = np.zeros(max(predictions.index.get_level_values(1))+1)

    for run in runs:
        error = calculate_error(data.loc[run], predictions.loc[run],
                                my_net.settings['feature_names'], interp=interp)

        if period_q != None:
            q = my_net.get_pos_features()
            error = error.abs()
            error[q] = period_q - (error[q] - period_q).abs()

        if t_in_T:
            f = interp1d(np.linspace(0, 1, len(error.values)), error.values, axis=0, kind=interp)
            error = pd.DataFrame(f(np.linspace(0, 1, 101)), columns=my_net.settings['feature_names'])

        error = error.values**2
        error = error.mean(axis=-1)
        mean[:len(error)] += error
    result['mse'] = mean/len(runs)

    with open(os.path.join(nn_path, 'metrics', 'mse.txt'), 'a') as file_:
        file_.write(f'# Settings: {json.dumps(settings)}\n')
        file_.write('  '.join([f'{time:.5e}' for time in result['time']])+'\n')
        file_.write('  '.join([f'{mse:.5e}' for mse in result['mse']])+'\n')

    if max_time != None:
        result = result[result['time'] <= max_time]

    return result

def dataset_mse_pos(data_path,
                    store_name,
                    nn_path,
                    max_time=None,
                    period_q=None,
                    t_in_T=False,
                    interp='linear',
                    test=None):
    """
    Same as dataset_mse, but only computes the MSE on the first feature
    (e.g. the first pendulum position), rather than over all features.
    """
    settings = {'period_q': period_q, 'interp': interp, 'test': test, 't_in_T': t_in_T}
    metrics_dir = os.path.join(nn_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    
    cache_file = os.path.join(metrics_dir, 'mse.txt')
    '''
    # --- check cache ---
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            lines = f.readlines()
        # find prior matching settings
        cached = [json.loads(line[12:-1]) 
                  for line in lines if line.startswith('# Settings')]
        if settings in cached:
            idx = cached.index(settings)
            raw = np.loadtxt(cache_file)
            block = raw[idx*2:idx*2+2].T
            df = pd.DataFrame(block, columns=['time','mse'])
            if max_time is not None:
                df = df[df['time'] <= max_time]
            return df
    '''
            
    # --- load network & data ---
    my_net = net_from_dir(nn_path)

    all_data = pd.read_hdf(os.path.join(data_path, store_name), '/all_runs')
    if test is not None:
        test_feats = pd.read_hdf(os.path.join(data_path, test), '/test_features')
        test_runs = test_feats['run'].unique()
        all_data = all_data[np.isin(all_data.index.get_level_values(0), test_runs)]

    inp    = all_data.xs(0, level='timestep')
    runs   = all_data.index.get_level_values(0).unique()
    max_t  = [all_data.loc[r].iloc[-1]['time'] for r in runs]

    # get predictions for each run at those timepoints
    predictions = predict_trajectories(my_net, inp, max_t, period_q=period_q)

    # wrap positions modulo 2π if needed
    if period_q is not None:
        q_feats = my_net.get_pos_features()
        all_data[q_feats] = (all_data[q_feats] + period_q) % (2*period_q) - period_q

    # prepare output DataFrame scaffold
    if t_in_T:
        times = np.linspace(0,1,101)
        df    = pd.DataFrame(times, columns=['time'])
        accu  = np.zeros_like(times)
    else:
        # pick the run with the longest T
        i_long = int(np.argmax(max_t))
        times  = predictions.loc[runs[i_long]]['time'].values
        df     = pd.DataFrame(times, columns=['time'])
        accu   = np.zeros(times.shape)

    # Decide which column(s) to track
    if store_name.find("doub") != -1:
        # Double pendulum → two “position” columns, e.g. ['q_A','q_B']
        first_feats = my_net.settings['feature_names'][:2]
    else:
        # Single pendulum → just one “position” column, in a 1‐element list
        first_feats = [ my_net.settings['feature_names'][0] ]

    # --- main loop: accumulate squared errors on those column(s) ---
    for run in runs:
        err_df = calculate_error(
            all_data.loc[run],
            predictions.loc[run],
            first_feats,   # pass a flat list of strings
            interp=interp
        )

        # If we need to wrap angles period‐wise, do it per column
        if period_q is not None:
            for col in first_feats:
                e = err_df[col].abs()
                err_df[col] = period_q - (e - period_q).abs()

        if t_in_T:
            # 1) Build an interpolator over [0,1] for each column in first_feats
            f = interp1d(
                np.linspace(0,1,len(err_df)),
                err_df[first_feats].values,  # shape = (len(err_df), len(first_feats))
                axis=0,
                kind=interp,
                fill_value='extrapolate'
            )
            # 2) Sample it on 101 equally spaced points in [0,1]
            samp2d = f(np.linspace(0,1,101))  # → shape = (101, len(first_feats))

            # 3) Compute mean‐squared error across all columns at each timepoint
            mse_per_timestep = np.mean(samp2d**2, axis=1)  # shape = (101,)

        else:
            # No re‐interpolation: just take raw errors for each column
            arr2d = err_df[first_feats].values  # shape = (n_timesteps, len(first_feats))
            mse_per_timestep = np.mean(arr2d**2, axis=1)  # shape = (n_timesteps,)

        # 4) Now accu is 1D of length >= n_timesteps. Add the 1D MSE‐array into it:
        accu[: len(mse_per_timestep)] += mse_per_timestep

    # take mean over runs
    df['mse'] = accu / len(runs)

    # cache to disk
    with open(cache_file, 'a') as f:
        f.write(f'# Settings: {json.dumps(settings)}\n')
        f.write('  '.join(f'{t:.5e}' for t in df['time']) + '\n')
        f.write('  '.join(f'{m:.5e}' for m in df['mse']) + '\n')

    # trim by max_time if requested
    if max_time is not None:
        df = df[df['time'] <= max_time]

    return df

def dataset_mae(data_path, 
                store_name, 
                nn_path, 
                max_time=None, 
                period_q=None, 
                t_in_T=False, 
                interp='linear', 
                test=None):
    
    """Calulcates the mean of the mean absolute error at each timestep for all trajectories in a dataset.

    Args:
        data_path (str, path-like object): Path to where the HDF5 store is saved.
        store_name (str, path-like object): Name of the HDF5 store.
        nn_path (str): Path to where the NN is saved.
        max_time (float): Maximal time until when to calculate the error.
        period_q (float): If not None inputs are mapped to [-period_q, period_q].
        t_in_T (bool): Whether to use one period as max time.
        interp (str): Type of iterpolation for times in between data points.
          Can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
        test (str, path-like object): HDF5 training store where the test data can be found.
          If None all data is used.

    Returns:
        pd.DataFrame: Times and mean of the MAE at each timestep.
    """
    settings = {'period_q': period_q, 'interp': interp, 'test': test, 't_in_T': t_in_T}
    if not os.path.isdir(os.path.join(nn_path, 'metrics')):
        os.makedirs(os.path.join(nn_path, 'metrics'))
    if os.path.isfile(os.path.join(nn_path, 'metrics', 'mae.txt')):
        
        print('-------------------------')
        print(nn_path)
        print('-------------------------')

        with open(os.path.join(nn_path, 'metrics', 'mae.txt'), 'r') as file_:
            lines = file_.readlines()
        cont = [json.loads(line[12:-1]) for line in lines if line[:10]=='# Settings']
        if settings in cont:
            result = np.loadtxt(os.path.join(nn_path, 'metrics', 'mae.txt'))
            i = cont.index(settings)
            result = pd.DataFrame(result[i*2:i*2+2].T, columns=['time', 'mae'])
            if max_time != None:
                result = result[result['time'] <= max_time]
            return result

    my_net = net_from_dir(nn_path)

    # load df with all runs
    data = pd.read_hdf(os.path.join(data_path, store_name), '/all_runs')
    
    # if a test directory is provided get the runs id from the test data
    if test:
        test_runs = pd.read_hdf(os.path.join(data_path, test), '/test_features')
        test_runs = test_runs['run'].unique()
        data = data[np.isin(data.index.get_level_values(0), test_runs)]
    inp = data.xs(0, level='timestep')

    # get the last time of each run
    runs = data.index.get_level_values(0).unique()
    max_time_data = [0.]*len(runs)
    for i, run in enumerate(runs):
        max_time_data[i] = data.loc[run].iloc[-1]['time']

    # get predictions for all runs
    predictions = predict_trajectories(my_net, inp, max_time_data, period_q=period_q)

    if period_q != None:
        q = my_net.get_pos_features()
        data[q] = (data[q] + period_q) % (2 * period_q) - period_q

    if t_in_T:
        result = pd.DataFrame(np.linspace(0, 1, 101), columns=['time'])
        mean = np.zeros(101)
    else:
        ind = max_time_data.index(max(max_time_data))
        result = pd.DataFrame(predictions.loc[runs[ind]]['time'])
        mean = np.zeros(max(predictions.index.get_level_values(1))+1)

    for run in runs:
        error = calculate_error(data.loc[run], predictions.loc[run],
                                my_net.settings['feature_names'], interp=interp)

        if period_q != None:
            q = my_net.get_pos_features()
            error = error.abs()
            error[q] = period_q - (error[q] - period_q).abs()

        if t_in_T:
            f = interp1d(np.linspace(0, 1, len(error.values)), error.values, axis=0, kind=interp)
            error = pd.DataFrame(f(np.linspace(0, 1, 101)), columns=my_net.settings['feature_names'])

        error = np.abs(error.values)
        error = error.mean(axis=-1)
        mean[:len(error)] += error
    result['mae'] = mean/len(runs)

    with open(os.path.join(nn_path, 'metrics', 'mae.txt'), 'a') as file_:
        file_.write(f'# Settings: {json.dumps(settings)}\n')
        file_.write('  '.join([f'{time:.5e}' for time in result['time']])+'\n')
        file_.write('  '.join([f'{mae:.5e}' for mae in result['mae']])+'\n')

    if max_time != None:
        result = result[result['time'] <= max_time]

    return result

def dataset_mae_q(data_path,
                  store_name,
                  nn_path,
                  max_time=None,
                  period_q=None,
                  t_in_T=False,
                  interp='linear',
                  test=None):
    """
    Calculates the MAE over only the position features (q's) 
    for all runs in the test set, at each timestep.

    Returns:
        pd.DataFrame with columns ['time', 'mae_q'].
    """
    # --- Load and filter raw data ---
    data = pd.read_hdf(os.path.join(data_path, store_name), '/all_runs')
    if test is not None:
        tf = pd.read_hdf(os.path.join(data_path, test), '/test_features')
        runs = tf['run'].unique()
        data = data[np.isin(data.index.get_level_values(0), runs)]

    # --- Initial states (timestep 0) ---
    inp = data.xs(0, level='timestep')

    # --- Determine each run's final time ---
    runs = data.index.get_level_values(0).unique()
    max_times = [ data.loc[r].iloc[-1]['time'] for r in runs ]

    # --- Load the network and predict trajectories ---
    net = net_from_dir(nn_path)
    preds = predict_trajectories(net, inp, max_times, period_q=period_q)

    # --- Optionally wrap raw q's into [−period_q,period_q] ---
    if period_q is not None:
        qcols = net.get_pos_features()
        data[qcols] = (data[qcols] + period_q) % (2*period_q) - period_q

    # --- Build the common time‐grid ---
    if t_in_T:
        times = np.linspace(0, 1, 101)
    else:
        # pick the run with the longest prediction
        idx = np.argmax(max_times)
        ref_run = runs[idx]
        times = preds.loc[ref_run]['time'].values

    # --- Prepare accumulator ---
    sum_q = np.zeros_like(times)

    feat_names = net.settings['feature_names']
    qcols     = net.get_pos_features()
    q_idx     = [feat_names.index(q) for q in qcols]

    # --- Loop over runs and accumulate q‐errors ---
    for run in runs:
        true_df = data.loc[run]
        pred_df = preds.loc[run]

        # interpolate & get per‐feature errors
        err_df = calculate_error(true_df, pred_df, feat_names, interp=interp).abs()

        # circularly wrap q‐errors if requested
        if period_q is not None:
            err_df[qcols] = period_q - (err_df[qcols] - period_q).abs()

        # resample to fractional‐period grid if needed
        if t_in_T:
            f = interp1d(np.linspace(0,1,len(err_df)), err_df.values,
                         axis=0, kind=interp)
            err_arr = f(times)
        else:
            err_arr = err_df.values

        # average only over q‐dimensions
        sum_q += err_arr[:, q_idx].mean(axis=1)

    # --- Finalize ---
    df_q = pd.DataFrame({
        'time':  times,
        'mae': sum_q / len(runs)
    })

    # --- Truncate if asked ---
    if max_time is not None:
        df_q = df_q[df_q['time'] <= max_time]

    return df_q
