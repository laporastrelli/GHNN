import os
from itertools import product
import numpy as np
import pandas as pd
import ghnn

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

import sys

# function for diplaying pendulum video
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
        # 4) drop the alpha channel → get an (H, W, 3) RGB image
        img = img[..., :3]
        # 5) append the frame
        writer.append_data(img)

    writer.close()
    plt.close(fig)


# define all parameters
which_half = "rand"
data_path = os.path.join('..', f'Data_half_{which_half}')
num_runs = 1000
store_name = 'pend_all_runs.h5.1'
nu_q = 0.95
nu_p = 0
dt = 0.02
tol, max_steps = (0.0001, 100)
seed = 0

# define integrator
integrator = ('Symplectic Euler', True)
integrator = ghnn.integrators.integrator_from_name(
              integrator[0], 
              integrator[1]
              )
# define keword arguments
kwargs = {'validation_share': 0.1,
          'test_share': 0.1,
          'seed': seed}
save_kwargs = {'complib': 'zlib', 'complevel': 1}

# create data path (if it does not exist)
if not os.path.exists(data_path):
    os.mkdir(data_path)

# get initial states for the pendulum
ghnn.data.generate_pendulum_inputs(data_path, 
                                   num_runs, 
                                   store_name, 
                                   nu_q=nu_q, 
                                   nu_p=nu_p, 
                                   seed=seed, 
                                   half=which_half)

# iterate over all runs
for run_num in range(num_runs):

    # read the initial state and constants from the saved 'pend_all_runs.h5.1' file
    data = pd.read_hdf(os.path.join(data_path, store_name), 
                       '/run' + str(run_num)).iloc[0]
    constants = pd.read_hdf(os.path.join(data_path, store_name), '/constants')
    bodies = constants['bodies']

    # helper variables
    calculations = []
    m = constants['mass']
    g = constants['g']
    l = constants['length']
    
    # grad functions for integrator
    grad_q = ghnn.gradients.Pendulum_grad_q(m, g, l)
    grad_p = ghnn.gradients.Pendulum_grad_p(m, l)

    # get initial state and store it in helper variable
    p = data[['p_'+body for body in bodies]].values
    q = data[['q_'+body for body in bodies]].values
    calculations.append(np.concatenate((q,p)))

    # set the time step and the initial time point
    T = 144
    t = 0

    # run the simulation
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

        calculations.append(np.concatenate((q,p)))
        t += dt

    # create column names for the data frame
    columns = [qp+'_'+body for (qp, body) in product(['q', 'p'], bodies)]
    data = pd.DataFrame(calculations, columns=columns)    
    data['time'] = data.index * dt
    data.to_hdf(os.path.join(data_path, store_name), key='/run' + str(run_num), format='fixed', **save_kwargs)
    constants['step_size'] = dt
    constants.to_hdf(os.path.join(data_path, store_name), key='/constants', format='fixed', **save_kwargs)

# save all runs in one file 
# here store_name = 'pend_all_runs.h5.1'
ghnn.data.combine(data_path, store_name, num_runs)

# create the training data
ghnn.data.create_pendulum_training_dataframe(data_path, 
                                             store_name, 
                                             'h_01_training.h5.1', 
                                             num_runs, 
                                             2e-1, 
                                             **kwargs)

save_name = os.path.join(data_path, 'pend_training.h5.1')

for d_type in ['', 'val_', 'test_']:
    feat = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), f'/{d_type}features')
    lab = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), f'/{d_type}labels')
    constants = pd.read_hdf(os.path.join(data_path, f'h_01_training.h5.1'), '/constants')
    runs = feat['run'].unique()

    keep = []
    for run in runs:
        run_data = feat[feat['run']==run]
        L = len(run_data)

        print('------------------------------------')
        print(L)
        print('------------------------------------')

        until = L // 6
        keep += list(run_data.iloc[:until].index)

    feat = feat.loc[keep]
    lab  = lab .loc[feat.index]

    feat.to_hdf(save_name, key=f'/{d_type}features', format='fixed', **save_kwargs)
    lab .to_hdf(save_name, key=f'/{d_type}labels',   format='fixed', **save_kwargs)

constants.to_hdf(save_name, key='/constants', format='fixed', **save_kwargs)

os.remove(os.path.join(data_path, 'h_01_training.h5.1'))
