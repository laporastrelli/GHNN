import os
from itertools import product
import numpy as np
import pandas as pd
import ghnn
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

import sys

def save_pendulum_video(angles, 
                        filename, 
                        length=1.0, 
                        fps=10, 
                        codec='libx264'):
    """
    angles: 1D array of shape (T,) giving the pendulum angle at each timestep
    filename: path to write the .mp4 file
    length: pendulum rod length
    fps: frames per second in output video
    codec: ffmpeg codec
    """
    T = len(angles)
    # Precompute cartesian trajectory
    x = length * np.sin(angles)
    y = -length * np.cos(angles)

    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-length * 1.1, length * 1.1)
    ax.set_ylim(-length * 1.1, 0.1 * length)
    ax.set_aspect('equal')
    ax.axis('off')

    # Pendulum line artist
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        line.set_data([0, x[i]], [0, y[i]])
        return (line,)

    # Use imageio to write frames directly to MP4
    writer = imageio.get_writer(
        filename, format='mp4', mode='I', fps=fps, codec=codec
    )

    for i in range(T):
        # 1) update the pendulum position
        update(i)
        # 2) draw the updated figure to the Agg backend
        fig.canvas.draw()
        # 3) grab the RGBA buffer from the canvas
        buf = fig.canvas.buffer_rgba()           # shape: (H, W, 4)
        img = np.asarray(buf)
        # 4) drop the alpha channel get an (H, W, 3) RGB image
        img = img[..., :3]
        # 5) append the frame
        writer.append_data(img)

    writer.close()
    plt.close(fig)


def main(args):

    # build dataset path
    root_ghnn_data_dir = '/data2/users/lr4617/data/GHNN'
    dataset_str = 'Data'
    if args.half:
        dataset_str += '_half'
    else:
        dataset_str += '_full'
    dataset_str += f'_{args.init}'
    if args.extrapolation:
        dataset_str += '_extrapolation'
    if args.horizon > 1:
        dataset_str += f'_horizon_{args.horizon}'
    data_path = os.path.join(root_ghnn_data_dir, dataset_str)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        os.mkdir(os.path.join(data_path, "vis"))
    print('###################################')
    print(f"Created data directory at {data_path}")
    print('###################################')
    
    # simulation parameters
    num_runs = args.num_runs
    store_name = 'pend_all_runs.h5.1'
    nu_q = 0.95
    nu_p = 0
    dt = 0.01
    tol, max_steps = (0.0001, 100)
    seed = 0

    # integrator parameters
    integrator = ('Symplectic Euler', True)
    integrator = ghnn.integrators.integrator_from_name(integrator[0], integrator[1])
    
    # dataset paramaters
    kwargs = {'validation_share': 0.1,
              'test_share': 0.1,
              'seed': seed, 
              'horizon': args.horizon}
    save_kwargs = {'complib': 'zlib', 'complevel': 1}

    # generate initial conditions for the pendulum
    ghnn.data.generate_pendulum_inputs(data_path, 
                                       num_runs, 
                                       store_name, 
                                       nu_q=nu_q, 
                                       nu_p=nu_p, 
                                       seed=seed,
                                       half=args.half,
                                       init_pos=args.init)

    # generate num_runs number of runs
    for run_num in range(num_runs):
        data = pd.read_hdf(os.path.join(data_path, store_name), '/run' + str(run_num)).iloc[0]
        constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
        bodies = constants['bodies']

        calculations = []
        m = constants['mass']
        g = constants['g']
        l = constants['length']
        grad_q = ghnn.gradients.Pendulum_grad_q(m, g, l)
        grad_p = ghnn.gradients.Pendulum_grad_p(m, l)

        p = data[['p_'+body for body in bodies]].values
        q = data[['q_'+body for body in bodies]].values
        calculations.append(np.concatenate((q,p)))
        
        # set maximum value of T based on extrapolation mode
        if args.extrapolation:
            T = args.max_T
        else:
            T = 1e10
        t = 0
        while t < T:
            p_old, q_old = p, q
            p, q = integrator(p_old, q_old, dt, grad_p, grad_q)

            diff = np.array([2*tol])
            j = 1
            while (tol < np.linalg.norm(diff) and 2**j <= max_steps):
                p_new, q_new = p, q
                p, q = p_old, q_old
                for k in range(2**j):
                    p, q = integrator(p, q, dt/(2**j), grad_p, grad_q)
                diff = np.concatenate((p-p_new, q-q_new))
                j += 1
                if max_steps < 2**j:
                    print('Warning: Too many steps required!!!')

            if not args.extrapolation:
                if (np.sign(q) - np.sign(q_old))[0] != 0 and T == 1e10:
                    T = 4 * (t + abs(q_old[0]) / (abs(q[0]) + abs(q_old[0])) * dt)

            calculations.append(np.concatenate((q,p)))
            t += dt

        if run_num < 5:
            angles_to_plot = np.array(calculations[::10])
            filename = os.path.join(data_path, "vis", f"example_video_{run_num}.mp4")
            save_pendulum_video(angles_to_plot[:, 0], filename)

        columns = [qp+'_'+body for (qp, body) in product(['q', 'p'], bodies)]
        data = pd.DataFrame(calculations, columns=columns)
        data['time'] = data.index * dt
        data.to_hdf(os.path.join(data_path, store_name), key='/run' + str(run_num), format='fixed', **save_kwargs)

        constants['step_size'] = dt
        constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **save_kwargs)

    # combine all runs in "all_runs"
    ghnn.data.combine(data_path, store_name, num_runs)

    # create train/val/test sets
    ghnn.data.create_pendulum_training_dataframe(data_path, 
                                                 store_name, 
                                                 'h_01_training.h5.1', 
                                                 num_runs, 
                                                 1e-1, 
                                                 **kwargs)

    # for each set type save truncated features and corresponding labels
    save_name = os.path.join(data_path, 'pend_training.h5.1')
    for d_type in ['', 'val_', 'test_']:
        feat      = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), f'/{d_type}features')
        lab       = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), f'/{d_type}labels')
        constants = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), '/constants')
        
        runs = feat['run'].unique()

        keep = []
        for run in runs:
            run_data = feat[feat['run'] == run]
            if args.extrapolation:
                L = len(run_data)
                until = L // 6
            else:
                until = np.where(np.diff(np.sign(run_data['q_A'])))[0][0] + 1

            keep += list(run_data.iloc[:until].index)

        print('###################################')
        print(f"FEATURES trajectories length: {until}")
        print(f"LABELS   trajectories length: {L}")
        print('###################################')

        feat = feat.loc[keep]
        lab = lab.loc[feat.index]

        feat.to_hdf(save_name, key=f'/{d_type}features', format='fixed', **save_kwargs)
        lab.to_hdf(save_name, key=f'/{d_type}labels', format='fixed', **save_kwargs)
    
    # save constants to Dataframe (together with features and labels for each set)
    constants.to_hdf(save_name, key='/constants', format='fixed', **save_kwargs)

    # delete intermediate files
    os.remove(os.path.join(data_path, 'h_01_training.h5.1'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create single pendulum datasets"
    )
    parser.add_argument('--half', type=int, default=1,
                        help="Decide whether the maximum period should be full or half.")
    parser.add_argument('--init', type=str, default='rand', 
                        help="Decide whther the initial posyion should be fixed or random")
    parser.add_argument('--extrapolation', action="store_true", default=True,
                        help="Decide whether the test maximum longer than one period.")
    parser.add_argument('--num_runs', type=int, default=1000,
                        help='Total number of runs to generate')
    parser.add_argument('--max_T', type=int, default=72, 
                        help='Test trajectories duration (for extrapolation=False)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Number of timesteps to predict in the future')
    
    args = parser.parse_args()
    
    print('###################################')
    print('Generating pendulum dataset with the following properties:')
    print(args)
    print('###################################')

    main(args)