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
    Ix = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3,
                   borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3,
                   borderType=cv2.BORDER_DEFAULT)

    if im1.shape != im2.shape:
        return "images shape must be equal"
    if win_size % 2 != 1:
        return "win_size must be an odd number!"
    # if the images is an RGB type
    if im1.ndim > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if im2.ndim > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    assert (win_size % 2 == 1)
    assert (im1.shape == im2.shape)

    Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    Gy = Gx.transpose()
    w = win_size // 2
    Ix = cv2.filter2D(im2, -1, Gx, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, Gy, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    u_v = []
    j_i = []
    k = 0
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):

            Nx = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Ny = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Nt = It[i - w:i + w + 1, j - w:j + w + 1].flatten()

            A = np.array([[sum(Nx[k] ** 2 for k in range(len(Nx))), sum(Nx[k] * Ny[k] for k in range(len(Nx)))],
                          [sum(Nx[k] * Ny[k] for k in range(len(Nx))), sum(Ny[k] ** 2 for k in range(len(Ny)))]])

            b = np.array([[-1 * sum(Nx[k] * Nt[k] for k in range(len(Nx))),
                           -1 * sum(Ny[k] * Nt[k] for k in range(len(Ny)))]]).reshape(2, 1)

            ev1, ev2 = np.linalg.eigvals(A)
            if ev2 < ev1:  # sort them
                temp = ev1
                ev1 = ev2
                ev2 = temp
            if ev2 >= ev1 > 1 and ev2 / ev1 < 100:  # check the conditions
                velo = np.dot(np.linalg.pinv(A), b)
                u = velo[0][0]
                v = velo[1][0]
                u_v.append(np.array([u, v]))
            else:
                k += 1
                # print('ev1: {0} ev2: {1}', ev1, ev2, k)
                u_v.append(np.array([0.0, 0.0]))

            j_i.append(np.array([j, i]))
    return np.array(j_i), np.array(u_v)


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
    # p = []
    # print(img1.shape)
    # uvs = np.zeros((img2.shape, 2))
    # print(uvs.shape)
    # for i in range(k):
    #     p.append(np.array([img1.copy(), img2.copy()]))
    #     img1 = cv2.pyrDown(img1, dstsize=(img1.shape[1] // 2, img1.shape[0] // 2))
    #     img2 = cv2.pyrDown(img2, dstsize=(img2.shape[1] // 2, img2.shape[0] // 2))
    # for level in range(k - 1, -1, -1):
    #     pyr1, pyr2 = p[level]
    #     pts, uv = opticalFlow(pyr1, pyr2, max(
    #         int(stepSize * pow(2, -level)), 1), winSize)
    #     converted_points = pts * np.power(2, level)
    #     try:
    #         uvs[converted_points[:, 1], converted_points[:, 0]] += 2 * uv
    #     except:
    #         pass
    # return uvs
    gp1 = gaussianPyr(img1, k)
    gp2 = gaussianPyr(img2, k)
    points, directions = opticalFlow(gp1[0], gp2[0], stepSize, winSize)
    U_V = directions
    for i in range(1, k+1):
        points, directions = opticalFlow(gp1[i], gp2[i], stepSize, winSize)
        directions_i = directions + 2 * U_V
        U_V = directions_i
    return np.darray(U_V)


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


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
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts_img2_tmp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts_img2 = cv2.perspectiveTransform(pts_img2_tmp, T)
    points = np.concatenate((pts_img1, pts_img2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    new_img = cv2.warpPerspective(im2, H_translation.dot(T), (x_max - x_min, y_max - y_min))
    new_img[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = im1

    return new_img


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
    gArr = cv2.getGaussianKernel(5, -1)
    gKernel = gArr @ gArr.transpose()
    for i in range(1, levels):
        It = cv2.filter2D(gauss_pyramid[i - 1], -1, gKernel)
        It = It[::2, ::2]
        gauss_pyramid.append(It)

    return gauss_pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaker = cv2.getGaussianKernel(5, -1)
    upsmaple_kernel = gaker @ gaker.transpose()
    upsmaple_kernel *= 4
    gau_pyr = gaussianPyr(img, levels)
    gau_pyr.reverse()
    lap_pyr = [gau_pyr[0]]
    for i in range(1, len(gau_pyr)):
        if len(gau_pyr[i - 1].shape) == 2:
            out = np.zeros((2 * gau_pyr[i - 1].shape[0], 2 * gau_pyr[i - 1].shape[1]), dtype=gau_pyr[i - 1].dtype)
        else:
            out = np.zeros((2 * gau_pyr[i - 1].shape[0], 2 * gau_pyr[i - 1].shape[1], img.shape[2]), dtype=gau_pyr[i - 1].dtype)
        out[::2, ::2] = gau_pyr[i - 1]
        expanded = cv2.filter2D(out, -1, upsmaple_kernel, borderType=cv2.BORDER_REPLICATE)
        if gau_pyr[i].shape != expanded.shape:
            x = expanded.shape[0] - gau_pyr[i].shape[0]
            y = expanded.shape[1] - gau_pyr[i].shape[1]
            expanded = expanded[x::, y::]
            diff_img = gau_pyr[i] - expanded
        else:
            diff_img = gau_pyr[i] - expanded
        lap_pyr.append(diff_img)

    lap_pyr.reverse()
    return lap_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaker = cv2.getGaussianKernel(5, -1)
    upsmaple_kernel = gaker @ gaker.transpose()
    upsmaple_kernel *= 4
    lap_pyr.reverse()
    r_img = [lap_pyr[0]]
    for i in range(1, len(lap_pyr)):
        if len(r_img[i - 1].shape) == 2:
            out = np.zeros((2 * r_img[i - 1].shape[0], 2 * r_img[i - 1].shape[1]), dtype=r_img[i - 1].dtype)
        else:
            out = np.zeros((2 * r_img[i - 1].shape[0], 2 * r_img[i - 1].shape[1], r_img[i - 1].shape[2]), dtype=r_img[i - 1].dtype)
        out[::2, ::2] = r_img[i - 1]
        temp = cv2.filter2D(out, -1, upsmaple_kernel, borderType=cv2.BORDER_REPLICATE)
        if lap_pyr[i].shape != temp.shape:
            x = temp.shape[0] - lap_pyr[i].shape[0]
            y = temp.shape[1] - lap_pyr[i].shape[1]
            new_img = temp[x::, y::] + lap_pyr[i]
        else:
            new_img = temp + lap_pyr[i]
        r_img.append(new_img)
    lap_pyr.reverse()
    r_img.reverse()
    return r_img[0]


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
    mask_lp = gaussianPyr(mask, levels)
    img_2_lp.reverse()
    img_1_lp.reverse()
    mask_lp.reverse()
    r_imgs = []
    for i in range(0, len(img_2_lp)):
        new_img = mask_lp[i] * img_1_lp[i] + (1 - mask_lp[i]) * img_2_lp[i]
        r_imgs.append(new_img)
    r_imgs.reverse()

    return naive_blend, laplaceianExpand(r_imgs)
