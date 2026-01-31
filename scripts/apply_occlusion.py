import numpy as np
from pathlib import Path

def apply_occlusion(joints,
                    joint_dropout_prob=0.3,
                    frame_dropout_prob=0.1,
                    noise_std=0.01):
    occluded = joints.copy()
    T, J, _ = occluded.shape

    mask = np.ones((T, J, 1))

    for t in range(T):
        # Drop entire frame
        if np.random.rand() < frame_dropout_prob:
            occluded[t] = 0
            mask[t] = 0
            continue

        for j in range(J):
            if np.random.rand() < joint_dropout_prob:
                occluded[t, j] = 0
                mask[t, j] = 0
            else:
                occluded[t, j] += np.random.normal(0, noise_std, 3)

    return occluded, mask


if __name__ == "__main__":
    in_dir = Path("data/joints_clean")
    out_dir = Path("data/joints_occluded")
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in in_dir.glob("*.npy"):
        joints = np.load(file)
        occluded, mask = apply_occlusion(joints)

        np.save(out_dir / f"{file.stem}_occluded.npy", occluded)
        np.save(out_dir / f"{file.stem}_mask.npy", mask)

        print(f"Processed {file.name}")
