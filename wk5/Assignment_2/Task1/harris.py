"""
CLAB Task-1: Harris Corner Detector
Yifan Luo (u7351505)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import det, inv


def conv2(img, conv_filter):
    """Convolution operation.

    Args:
        img (numpy.ndarray): The input image.
        conv_filter (numpy.ndarray): The input convolution filter.

    Returns:
        numpy.ndarray: The convolution operation result.
    """
    # flip the filter
    f_size_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_size_1 - 1, -1, -1), :][:, range(f_size_2 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    """Generate a Gaussian filter given its shape and standard deviation.

    Args:
        shape (tuple, optional): The shape of the Gaussian filter. Defaults to (3, 3).
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 0.5.

    Returns:
        h (numpy.ndarray): The generated Gaussian kernel.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]  # coordinates of the filter
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))  # Gaussian kernel 13 * 13
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # normalise the sum of h to 1
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2  # parameter for the Gaussian filter in fspecial
img_path = 'Task1/Harris-2.jpg'
window_size_M = 3  # window size for the calculation of M
window_size_R = 3  # neighbourhood size for local maxima of R
k = 0.05         # empirically determined constant ranged from 0.04 to 0,06
thresh = 1e-8    # threshold in thresholding process on R

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # vertical Sobel filter
dy = dx.transpose()                                  # horizontal Sobel filter

# Load pictures
bw = plt.imread(img_path)   # (row, col, rgb) for Harris-[1345].jpg; (row, col) for Harris-2.jpg
bw = np.array(bw * 255, dtype='uint8')
if bw.ndim == 3:    # convert RGB images (Harris-[1345].jpg) to grayscale images
    bw = cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)

# Computer x and y derivatives of image
Ix = conv2(bw, dx)  # horizontal derivative
Iy = conv2(bw, dy)  # vertical derivative

# Generate a Gaussian filter for smoothing the second-order derivatives
g = fspecial(shape=(max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma=sigma)
Iy2 = conv2(np.power(Iy, 2), g)  # smoothed second-order vertical derivative (Iy)
Ix2 = conv2(np.power(Ix, 2), g)  # smoothed second-order horizontal derivative (Ix)
Ixy = conv2(Ix * Iy, g)          # smoothed production of Iy and Ix


######################################################################
# Task: Compute the Harris Cornerness
######################################################################


def construct_R(Ix2, Iy2, Ixy, window_size=3, k=0.05):
    """Calculate the cornerness for each pixel.

    Args:
        Ix2 (numpy.ndarray): The second-order derivative in horizontal direction.
        Iy2 (numpy.ndarray): The second-order derivative in vertical direction.
        Ixy (numpy.ndarray): The product of second-order derivatives in both directions.
        window_size (int, optional): The size of neighborhood for M=avg(sum_neighborhood([[Ix2, Ixy], [Ixy, Iy2]])). Defaults to 3.
        k (float, optional): The empirical parameter of cornerness computation R=det(M)-k*trace(M)^2
        
    Return:
        numpy.ndarray: The Harris response matrix R.
    """
    R = np.empty_like(Ix2)  # cornerness for each pixel
    pad_width = (window_size - 1) // 2
    # zero padding for each derivatives 
    Ix2_p = np.pad(Ix2, pad_width, mode='constant', constant_values=0)
    Iy2_p = np.pad(Iy2, pad_width, mode='constant', constant_values=0)
    Ixy_p = np.pad(Ixy, pad_width, mode='constant', constant_values=0)
    for r in range(R.shape[0]):
        for c in range(R.shape[1]):
            # compute each entry for constrcuting the matrix M, suppose w(x,y)=1/window_size**2
            Ix2_sum = np.sum(Ix2_p[r:r + window_size, c:c + window_size]) / window_size ** 2
            Iy2_sum = np.sum(Iy2_p[r:r + window_size, c:c + window_size]) / window_size ** 2
            Ixy_sum = np.sum(Ixy_p[r:r + window_size, c:c + window_size]) / window_size ** 2
            # construct the matrix M
            M = np.array([[Ix2_sum, Ixy_sum], 
                          [Ixy_sum, Iy2_sum]])
            # calculate the cornerness for each pixel
            R[r, c] = det(M) - k * (np.trace(M) ** 2)
    return R


R = construct_R(Ix2, Iy2, Ixy, window_size_M, k)


######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################


def thresh_non_max_sup(R, thresh, window_size):
    """Conduct the thresholding and non-maximum suppression on the input cornerness R.

    Args:
        R (numpy.ndarray): The input cornerness for each pixel.
        thresh (float): The threshold which only keeps R>thresh
        window_size (int): The neighborhood size for the local maxima.

    Returns:
        (numpy.ndarray, numpy.ndarray): Return the index after thresholding and non-max suppression.
    """
    # R_t = np.copy(R)
    # R_t[R_t < thresh] = 0
    
    # R_nm = np.zero_like(R)
    # pad_width = (window_size - 1) // 2
    # R_t_p = np.pad(R_t, pad_width, mode='minimum')
    # for r in range(R_nm.shape[0]):
    #     for c in range(R_nm.shape[1]):
    #         local_max = R_t_p[r:r + 2 * pad_width + 1, c:c + 2 * pad_width + 1].max()
    #         if R_t[r, c] == local_max:
    #             R_t[r, c] = 1
    
    # thresholding: select index where the cornerness is larger than threshold
    thresh_idx_x, thresh_idx_y = np.where(R > thresh)
    thresh_idx = np.array([(x, y) for (x, y) in zip(thresh_idx_x, thresh_idx_y)])
    # non-max suppression: select index where is the local maxima cornerness
    non_max_idx = []
    R_p = np.pad(R, pad_width=(window_size - 1) // 2, mode='reflect')
    for r, c in thresh_idx:
        # for each pixel, find its local maxima of cornerness
        local_max = R_p[r:r + window_size, c:c + window_size].max()
        # print(R[r, c], R_p[r:r + window_size, c:c + window_size])
        # only keep index of local maxima
        if R[r, c] == local_max:
            non_max_idx.append([r, c])
    return thresh_idx, np.asarray(non_max_idx)


# R results after thresholding and non-max suppression
corner_thresh, corner_thresh_non_max = thresh_non_max_sup(R, thresh*R.max(), window_size_R)
# R results using the built-in function cv2.cornerHarris
R_cv = cv2.cornerHarris(bw, blockSize=window_size_M, ksize=3, k=k)
corner_thresh_cv, corner_thresh_non_max_cv = thresh_non_max_sup(R_cv, thresh*R_cv.max(), window_size_R)


def plot_R(img_path, R, corner_thresh, corner_thresh_non_max):
    # results comparison between:
    img = plt.imread(img_path)
    fig, ax = plt.subplots(1, 4, figsize=(8, 4))
    fig.suptitle(img_path.split('/')[1] + '\n' + 
                "threshold:" + str(thresh) + " k:" + str(k))
    # 1. original image
    ax[0].imshow(img, cmap='gray' if np.ndim(img) == 2 else 'viridis')
    ax[0].axis('off')
    ax[0].set_title("Original")
    # 2. cornerness R
    ax[1].imshow(R, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title("Cornerness")
    # 3. corners after threholding with non-max suppression
    ax[2].imshow(img, cmap='gray' if np.ndim(img) == 2 else 'viridis')
    ax[2].scatter(corner_thresh[:, 1], corner_thresh[:, 0], color='green', s=0.8, marker='o')
    ax[2].axis('off')
    ax[2].set_title("Thresh")
    # 4. corners after threholding with non-max suppression
    ax[3].imshow(img, cmap='gray' if np.ndim(img) == 2 else 'viridis')
    ax[3].scatter(corner_thresh_non_max[:, 1], corner_thresh_non_max[:, 0], color='green', s=0.8, marker='o')
    ax[3].axis('off')
    ax[3].set_title("Thresh + Non-Max")
    plt.show()
    
    
plot_R(img_path, R, corner_thresh, corner_thresh_non_max)
plot_R(img_path, R_cv, corner_thresh_cv, corner_thresh_non_max_cv)


def trans_mat_out_shape(img, theta):
    """Return the transformation matrix and the corresponding output shape.

    Args:
        img (numpy.ndarray): 
        theta (float): The rotation degree. The value should in the range of [0, 360].

    Returns:
        (np.ndarray, (int, int)): The generated transformation matrix, and the shape of output image (n_row, n_col).
    """
    radians = theta * np.pi / 180
    trans_mat = np.array([[np.cos(radians), -np.sin(radians)],
                          [np.sin(radians), np.cos(radians)]])
    corner_top_left = np.array([0, 0]).T
    corner_top_right = np.array([0, img.shape[1] - 1]).T
    corner_bot_left = np.array([img.shape[0] - 1, 0]).T
    corner_bot_right = np.array([img.shape[0] - 1, img.shape[1] - 1]).T
    
    if 0 <= theta < 90 or 180 <= theta < 270:
        n_r = np.abs((trans_mat @ corner_top_right)[0] - (trans_mat @ corner_bot_left)[0])
        n_c = np.abs((trans_mat @ corner_bot_right)[1] - (trans_mat @ corner_top_left)[1])
    if 90 <= theta < 180 or 270 <= theta <= 360:
        n_r = np.abs((trans_mat @ corner_bot_right)[0] - (trans_mat @ corner_top_left)[0])
        n_c = np.abs((trans_mat @ corner_bot_left)[1] - (trans_mat @ corner_top_right)[1])
    output_shape = (int(np.ceil(n_r)), int(np.ceil(n_c))) if img.ndim == 2 else (int(np.ceil(n_r)), int(np.ceil(n_c)), 3)
    return trans_mat, output_shape  


def inv_warp(img, inv_trans_mat, out_size):
    """Inverse image warpping.

    Args:
        img (numpy.ndarray): The image to be rotated.
        inv_trans_mat (numpy.ndarray): The inverse of transformation matrix used to rotate the image.
        out_size (Tuple(int, int)): The size of output rotated image.
    
    Returns:
        numpy.ndarray: The rotated image.
    """
    n_row, n_col = out_size[0], out_size[1]
    cent_x, cent_y = img.shape[0] / 2, img.shape[1] / 2
    rot_u, rot_v = inv(inv_trans_mat) @ np.array([cent_x, cent_y]).T
    cent_u, cent_v = n_row / 2, n_col / 2
    shift_u, shift_v = rot_u - cent_u, rot_v - cent_v
    res = np.zeros(out_size)
    cond = lambda x, y: (0 <= x < img.shape[0]) and (0 <= y < img.shape[1])
    for u in range(n_row):
        for v in range(n_col):
            x, y = inv_trans_mat @ np.array([u + shift_u, v + shift_v]).T
            if cond(x, y):
                if int(x) == x and int(y) == y:
                    res[u, v] = img[int(x), int(y)]
                else:
                    # bilinear interpolation 
                    y1, y2 = int(np.floor(y)), int(np.ceil(y))
                    x1, x2 = int(np.floor(x)), int(np.ceil(x))
                    if cond(x1, y2) and cond(x2, y2) and cond(x1, y1) and cond(x2, y1):
                        if x1 == x2:
                            res[u, v] = img[x1, y2] * (y - y1) / (y2 - y1) + img[x1, x1] * (y2 - y) / (y2 - y1)
                        elif y1 == y2:
                            res[u, v] = img[x2, y2] * (x2 - x) / (x2 - x1) + img[x1, y2] * (x - x1) / (x2 - x1)
                        else:
                            R1 = img[x1, y2] * ((x2 - x) / (x2 - x1)) + img[x2, y2] * ((x - x1) / (x2 - x1))
                            R2 = img[x1, y1] * ((x2 - x) / (x2 - x1)) + img[x2, y1] * ((x - x1) / (x2 - x1))
                            res[u, v] = R1 * (y - y1) / (y2 - y1) + R2 * (y2 - y) / (y2 - y1)
                    
    return res.astype('uint8')


img = plt.imread(img_path)
fig, ax = plt.subplots(2, 4)
retate_imgs = []
for i, theta in enumerate([0, 90, 180, 270]):
    trans_mat, output_shape = trans_mat_out_shape(img, theta=theta)  
    rotate_img = inv_warp(img, inv(trans_mat), output_shape)
    retate_imgs.append(rotate_img)
    ax[0, i].imshow(rotate_img, cmap='gray' if rotate_img.ndim == 2 else 'viridis')
    ax[0, i].axis('off')
    ax[0, i].set_title(str(theta) + " degrees")
    bw = cv2.cvtColor(rotate_img, cv2.COLOR_RGB2GRAY) if rotate_img.ndim == 3 else rotate_img
    Ix = conv2(bw, dx)  # horizontal derivative
    Iy = conv2(bw, dy)  # vertical derivative
    Iy2 = conv2(np.power(Iy, 2), g)  # smoothed second-order vertical derivative (Iy)
    Ix2 = conv2(np.power(Ix, 2), g)  # smoothed second-order horizontal derivative (Ix)
    Ixy = conv2(Ix * Iy, g)          # smoothed production of Iy and Ix
    R = construct_R(Ix2, Iy2, Ixy, window_size_M, k)
    corner_thresh, corner_thresh_non_max = thresh_non_max_sup(R, thresh*R.max(), window_size_R)
    ax[1, i].imshow(rotate_img, cmap='gray' if np.ndim(rotate_img) == 2 else 'viridis')
    ax[1, i].scatter(corner_thresh_non_max[:, 1], corner_thresh_non_max[:, 0], color='green', s=0.8, marker='o')
    ax[1, i].axis('off')
fig.suptitle(img_path.split('/')[1])
plt.show()

