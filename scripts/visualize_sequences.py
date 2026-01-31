import numpy as np
import matplotlib.pyplot as plt

def plot_sequence(joints, title):
    xs = joints[:, :, 0]
    ys = joints[:, :, 1]

    plt.figure(figsize=(6, 6))
    for j in range(xs.shape[1]):
        plt.plot(xs[:, j], ys[:, j], alpha=0.6)

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    clean = np.load("data/joints_clean/sample_3.npy")
    occluded = np.load("data/joints_occluded/sample_3_occluded.npy")

    plot_sequence(clean, "Clean Hand Trajectories")
    plot_sequence(occluded, "Occluded + Noisy Hand Trajectories")
