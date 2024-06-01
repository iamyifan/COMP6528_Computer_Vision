import os

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    if not os.path.exists("data/"):
        os.makedirs("data/")

    img_name = "stereo2012a"
    img = plt.imread(os.path.join("images/", f"{img_name}.jpg"))

    plt.imshow(img)
    # graphical user interface to get 6 points
    # function doc: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ginput.html
    uv = plt.ginput(6, timeout=0.0)
    np.save(os.path.join("data/", f"{img_name}_uv.npy"), uv)

    # change the XYZ array according to your selection of uv coordinates
    xyz = np.array([
        [0, 14, 7],  # YZ plane p1
        [7, 14, 0],  # XY plane p1
        [0, 7, 7],  # YZ plane p2
        [7, 7, 0],  # XY plane p2
        [7, 0, 14],  # XZ plane p1
        [14, 0, 7],  # XZ plane p2
    ])
    np.save(os.path.join("data/", f"{img_name}_xyz.npy"), xyz)
