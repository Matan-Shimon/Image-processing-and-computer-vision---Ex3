import math
import sys
from typing import List
from numpy import linalg as LA
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 100

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    if im1.ndim > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if im2.ndim > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    if im1.shape != im2.shape:
        return "The images must be in the same size"

    if win_size % 2 == 0:
        return "win_size must be an odd number"

    origin_index = []
    u_v = []
    filter = np.array([[1, 0, -1]])
    Ix = cv2.filter2D(im1, -1, filter, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im1, -1, filter.transpose(), borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1
    mid = np.round(win_size / 2).astype(int)
    for x in range(mid, im1.shape[0] - mid, step_size):
        for y in range(mid, im1.shape[1] - mid, step_size):
            sx, ex = x - mid, x + mid + 1
            sy, ey = y - mid, y + mid + 1

            A = np.zeros((pow(win_size, 2), 2))
            b = np.zeros((pow(win_size, 2), 1))

            A[:, 0] = Ix[sx: ex, sy: ey].flatten()
            A[:, 1] = Iy[sx: ex, sy: ey].flatten()
            b[:, 0] = -It[sx: ex, sy: ey].flatten()

            b = A.transpose() @ b
            A = A.transpose() @ A
            eigen_values, v = np.linalg.eig(A)

            eigen_values.sort()
            big = eigen_values[1]
            small = eigen_values[0]

            if big >= small > 1 and big / small < 100:
                ans = np.dot(np.linalg.pinv(A), b)
                origin_index.append([y, x])
                u_v.append(ans)

    return np.array(origin_index).reshape(len(origin_index), 2), -np.array(u_v).reshape(len(u_v), 2)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    if img1.ndim > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    if img1.shape != img2.shape:
        return "The images must be in the same size"

    ans = np.zeros((img1.shape[0], img1.shape[1], 2))
    gp1 = [img1]
    gp2 = [img2]
    for i in range(k-1):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
    gp1.reverse()
    gp2.reverse()

    for i in range(k):
        points, directions = opticalFlow(gp1[i], gp2[i], stepSize, winSize)
        for j in range(len(points)):
            y, x = points[j][0], points[j][1]
            u, v = directions[j][0], directions[j][1]
            if i != k-1:
                ans[x * 2][y * 2][0] = u + 2 * ans[x][y][0]
                ans[x * 2][y * 2][1] = v + 2 * ans[x][y][1]
            else:
                ans[x][y][0] *= 2
                ans[x][y][0] += u
                ans[x][y][1] *= 2
                ans[x][y][1] += v

    return ans


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    EPS = 0.000001
    min_error = np.inf
    final_ans = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    good_features = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    cv_lk_pyr = cv2.calcOpticalFlowPyrLK(im1, im2, good_features, None)[0]
    directions = cv_lk_pyr - good_features
    for index in range(len(directions)):
        u = directions[index, 0, 0]
        v = directions[index, 0, 1]
        check = final_ans
        check[0][2] = u
        check[1][2] = v
        moved_img = cv2.warpPerspective(im1, check, im1.shape[::-1])
        mse = np.square(im2 - moved_img).mean()
        if mse < min_error:
            min_error = mse
            final_ans = check
        if mse < EPS:
            break

    return final_ans


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    EPS = 0.000001
    min_error = np.inf
    final_rotation = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype=np.float32)
    directions = opticalFlow(im1, im2)[1]
    for u, v in directions:
        if u == 0:
            angle = 0
        else:
            angle = np.arctan(v / u)
        check = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=np.float32)
        rotated_img = cv2.warpPerspective(im1, check, im1.shape[::-1])
        mse = np.square(im2 - rotated_img).mean()
        if mse < min_error:
            min_error = mse
            final_rotation = check
            final_rotated_img = rotated_img.copy()
        if mse < EPS:
            break

    translation = findTranslationLK(final_rotated_img, im2)
    final_ans = translation @ final_rotation
    return final_ans


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    if im1.ndim > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if im2.ndim > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1_height = im1.shape[0]
    im1_width = im1.shape[1]
    im2_height = im2.shape[0]
    im2_width = im2.shape[1]
    ans = np.zeros(im2.shape)
    for i in range(1, im2_height):
        for j in range(1, im2_width):
            check = np.linalg.inv(T).dot(np.array([i, j, 1]))
            found_x, found_y = int(round(check[0])), int(round(check[1]))
            if 0 <= found_x < im1.shape[0] and 0 <= found_y < im1.shape[1]:
                ans[i, j] = im1[found_x, found_y]

    return ans


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    gauss_pyramid = [img]
    for i in range(1, levels):
        blurred = cv2.GaussianBlur(gauss_pyramid[-1], (5, 5), 0)
        blurred = blurred[::2, ::2]
        gauss_pyramid.append(blurred)

    return gauss_pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    img_pyramids = [img]
    for i in range(levels-1):
        img_pyramids.append(cv2.pyrDown(img_pyramids[-1]))
    img_pyramids.reverse()
    expanded = []
    for i in range(levels-1):
        expanded.append(cv2.pyrUp(img_pyramids[i], dstsize=(img_pyramids[i+1].shape[1], img_pyramids[i+1].shape[0])))
    img_pyramids.reverse()
    expanded.reverse()
    laplacians = []
    for i in range(levels-1):
        laplacians.append(img_pyramids[i] - expanded[i])
    laplacians.append(img_pyramids[len(img_pyramids)-1])
    return laplacians


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    smallest_pyr_image = lap_pyr[-1]
    lap_pyr.reverse()
    for i in range(1, len(lap_pyr)):
        smallest_pyr_image = cv2.pyrUp(smallest_pyr_image, dstsize=(lap_pyr[i].shape[1], lap_pyr[i].shape[0]))
        smallest_pyr_image += lap_pyr[i]
    lap_pyr.reverse()
    return smallest_pyr_image


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    naive_blend = img_1 * mask + img_2 * (1 - mask)
    img_1_lp = laplaceianReduce(img_1, levels)
    img_2_lp = laplaceianReduce(img_2, levels)
    mask_gp = [mask]
    for i in range(levels-1):
        mask_gp.append(cv2.pyrDown(mask_gp[-1]))
    img_2_lp.reverse()
    img_1_lp.reverse()
    mask_gp.reverse()
    blending = []
    for i in range(levels):
        new_img = img_1_lp[i] * mask_gp[i] + (1 - mask_gp[i]) * img_2_lp[i]
        blending.append(new_img)
    blending.reverse()
    final_blend = laplaceianExpand(blending)
    return naive_blend, final_blend
