import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

from simulation import Variables

E_arr = []
t_arr = []
with open(Variables.dumpfile, 'rb') as fr:
    try:
        for i in range(Variables.n_t):
            E, t = (pickle.load(fr))
            E_arr.append(E)
            t_arr.append(t)
    except EOFError:
        pass

E = np.array(E_arr)
t = np.array(t_arr)

minimum = np.min(E)
maximum = np.max(E)

fig, ax = plt.subplots()
plt.style.use('dark_background')

def update(frame):
    ax.clear()
    curr_E = E[frame]
    ax.imshow(curr_E, cmap="inferno", vmin=minimum*0.1, vmax=maximum*0.1, extent=(Variables.boundary[0], Variables.boundary[1], Variables.boundary[0], Variables.boundary[1]))
    ax.set_title(f"Time t={t[frame]:.2f} s")

animation = FuncAnimation(fig, update, frames=len(E), interval=1/Variables.fps*1000)

def progress_callback(current_frame, total_frames):
    progress = current_frame / total_frames
    print(f"{progress*100:.1f}%")

animation.save("result.gif", writer="ffmpeg", progress_callback=progress_callback)