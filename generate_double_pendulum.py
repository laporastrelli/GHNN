import os
import ghnn
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_double_pendulum_video(
    angles: np.ndarray,
    filename: str,
    lengths: tuple[float, float] = (1.0, 1.0),
    fps: int = 10,
    codec: str = 'libx264'
):
    """
    angles: array of shape (T, 2) giving [theta1, theta2] at each timestep
    filename: output .mp4 path
    lengths: (l1, l2) lengths of the first and second rod
    fps: frames per second
    codec: ffmpeg codec
    """
    T = angles.shape[0]
    l1, l2 = lengths
    # compute positions
    theta1 = angles[:, 0]
    theta2 = angles[:, 1]
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # setup figure
    fig, ax = plt.subplots(figsize=(4,4))
    max_range = l1 + l2
    ax.set_xlim(-max_range*1.1, max_range*1.1)
    ax.set_ylim(-max_range*1.1, 0.1*max_range)
    ax.set_aspect('equal')
    ax.axis('off')

    # artists
    line, = ax.plot([], [], 'o-', lw=2)

    def update(i):
        xs = [0, x1[i], x2[i]]
        ys = [0, y1[i], y2[i]]
        line.set_data(xs, ys)
        return (line,)

    # writer
    writer = imageio.get_writer(filename, format='mp4', mode='I', fps=fps, codec=codec)

    for i in range(T):
        update(i)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)

############################################
generate_data = True
############################################

data_path = os.path.join('/data2/users/lr4617/data/GHNN', 
                         'Data_double_pendulum')
num_runs = 2000
store_name = 'doub_pend_all_runs.h5.1'
nu_q = 0.5
nu_p = 0
T = 72
dt = 0.01
step_size = 1e-1
converge = (0.001, 100)
seed = 0

kwargs = {'validation_share': 0.1,
          'test_share': 0.1,
          'seed': seed}

if generate_data:
    ghnn.data.double_pendulum(data_path, 
                            num_runs, 
                            store_name=store_name, 
                            nu_q=nu_q, 
                            nu_p=nu_p, 
                            T=T, 
                            dt=dt, 
                            converge=converge, 
                            seed=seed)

    ghnn.data.create_pendulum_training_dataframe(data_path, 
                                                store_name, 
                                                'doub_pend_training.h5.1', 
                                                num_runs, 
                                                step_size, 
                                                max_time=12, 
                                                **kwargs)



###################################################################################
train = pd.read_hdf(os.path.join(data_path, 'doub_pend_training.h5.1'), key='/features')
grp = train.groupby('run')
for run_id, df_run in grp:
    angles = df_run[[col for col in train.columns if col.startswith('q_')]].values
    out_file = os.path.join(data_path, "vis_initial", f'double_pendulum_run_{run_id}.mp4')
    save_double_pendulum_video(angles, out_file, lengths=(1.0,1.0), fps=10)
    if run_id == 4:
        break

with pd.HDFStore(os.path.join(data_path, 'doub_pend_all_runs.h5.1'), mode='r') as all_data:
    print('keys: ', all_data.keys()[:10])
    keys_og = all_data.keys()[:10]

val = pd.read_hdf(os.path.join(data_path, 'doub_pend_all_runs.h5.1'), key='/run0')
angles = val[[col for col in val.columns if col.startswith('q_')]].values
out_file = os.path.join(data_path, "vis_initial", f'double_pendulum_run_0_val.mp4')
save_double_pendulum_video(angles, out_file, lengths=(1.0,1.0), fps=10)


###################################################################################
raw_store = os.path.join(data_path, 'doub_pend_all_runs.h5.1')
train_store = os.path.join(data_path, 'doub_pend_training.h5.1')

# 1) visualize training runs
vis_train_dir = os.path.join(data_path, "vis_training")
os.makedirs(vis_train_dir, exist_ok=True)

print(f"Loading training runs from {train_store}")
train_df = pd.read_hdf(train_store, key="/features")
angle_cols = [c for c in train_df.columns if c.startswith("q_")]

cnt=0
for run_id, df_run in train_df.groupby("run"):
    angles = df_run[angle_cols].values  # shape (T, 2)
    print(f"Visualizing run {run_id} with shape {angles.shape}")
    out_file = os.path.join(
        vis_train_dir, f"double_pendulum_run_{run_id}.mp4"
    )
    save_double_pendulum_video(
        angles,
        out_file,
        lengths=(1.0, 1.0),
        fps=10,
    )
    cnt += 1
    if cnt >= 5:  # limit to first 10 runs for quick visualization
        break
print("✅ Finished visualizing training runs.")

# 2) visualize raw runs
vis_raw_dir = os.path.join(data_path, "vis_raw")
os.makedirs(vis_raw_dir, exist_ok=True)

print(f"Loading raw runs from {raw_store}")
with pd.HDFStore(raw_store) as store:
    cnt = 0
    for key in store.keys():  # keys like '/run0', '/run1', …
        if not key.startswith("/run"):
            continue
        df_raw = store[key]
        angle_cols = [c for c in df_raw.columns if c.startswith("q_")]
        if not angle_cols:
            continue

        angles = df_raw[angle_cols].values  # shape (T, ≥2)
        print(f"Visualizing {key} with shape {angles.shape}")
        run_id = key.strip("/")
        out_file = os.path.join(
            vis_raw_dir, f"double_pendulum_raw_{run_id}.mp4"
        )
        save_double_pendulum_video(
            angles[::10],
            out_file,
            lengths=(1.0, 1.0),
            fps=10,
        )
        cnt += 1
        if cnt >= 5:
            break
print("✅ Finished visualizing raw runs.")