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
    return 314669342

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
    # if the image is RGB, convert to gray
    if im1.ndim > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if im2.ndim > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    # if the images don't have the same shape we will throw an error
    if im1.shape != im2.shape:
        return "The images must be in the same size"
    # win size must be odd
    if win_size % 2 == 0:
        return "win_size must be an odd number"

    origin_index = []
    u_v = []
    filter = np.array([[1, 0, -1]])
    # getting the derivatives
    Ix = cv2.filter2D(im1, -1, filter, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im1, -1, filter.transpose(), borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1  # getting the change
    mid = np.round(win_size / 2).astype(int)
    for x in range(mid, im1.shape[0] - mid, step_size):
        for y in range(mid, im1.shape[1] - mid, step_size):
            # calculating the pixels relative to the middle to get the window bounds
            min_x, max_x = x - mid, x + mid + 1
            min_y, max_y = y - mid, y + mid + 1

            A = np.zeros((pow(win_size, 2), 2))
            b = np.zeros((pow(win_size, 2), 1))

            # flattering
            A[:, 0] = Ix[min_x: max_x, min_y: max_y].flatten()
            A[:, 1] = Iy[min_x: max_x, min_y: max_y].flatten()
            b[:, 0] = -It[min_x: max_x, min_y: max_y].flatten()

            # dot product calculation
            b = A.transpose() @ b
            A = A.transpose() @ A
            # getting the eigen values
            eigen_values, v = np.linalg.eig(A)
            # sorting the eigen values
            eigen_values.sort()
            big = eigen_values[1]
            small = eigen_values[0]
            # checking by the formula we have been given
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
    # if the image is RGB, convert to gray
    if img1.ndim > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # if the images don't have the same shape we will throw an error
    if img1.shape != img2.shape:
        return "The images must be in the same size"
    # creating an answer array in the img1 shape and adding another dimension for u and v
    ans = np.zeros((img1.shape[0], img1.shape[1], 2))
    # getting the image pyramids
    gp1 = [img1]
    gp2 = [img2]
    for i in range(k-1):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
    # reverse to start from the smallest image
    gp1.reverse()
    gp2.reverse()

    for i in range(k):
        # getting the points and directions
        points, directions = opticalFlow(gp1[i], gp2[i], stepSize, winSize)
        for j in range(len(points)):
            y, x = points[j][0], points[j][1]  # getting the location
            u, v = directions[j][0], directions[j][1]  # getting the u and v relative to that pixels
            if i != k-1:  # the formula we have been given
                ans[x * 2][y * 2][0] = u + 2 * ans[x][y][0]
                ans[x * 2][y * 2][1] = v + 2 * ans[x][y][1]
            else:  # if we are in the biggest pyramid, we need to do it differently so we won't get out of bounds
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
    # calculating cv lk to get more accurate results
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    good_features = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    cv_lk_pyr = cv2.calcOpticalFlowPyrLK(im1, im2, good_features, None)[0]
    directions = cv_lk_pyr - good_features  # getting the directions
    for index in range(len(directions)):
        u = directions[index, 0, 0]
        v = directions[index, 0, 1]
        check = final_ans
        check[0][2] = u
        check[1][2] = v
        moved_img = cv2.warpPerspective(im1, check, im1.shape[::-1])  # warping the image
        mse = np.square(im2 - moved_img).mean()  # calculating the error relative to image 2
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
    final_rotated_img = 0
    for u, v in directions:
        if u == 0:
            angle = 0
        else:
            angle = np.arctan(v / u)  # getting the angle
        check = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=np.float32)
        rotated_img = cv2.warpPerspective(im1, check, im1.shape[::-1])  # rotating the image
        mse = np.square(im2 - rotated_img).mean()  # calculating the error relative to image 2
        if mse < min_error:
            min_error = mse
            final_rotation = check
            final_rotated_img = rotated_img.copy()
        if mse < EPS:
            break

    translation = findTranslationLK(final_rotated_img, im2)  # finding the translation from the rotated image to im2
    final_ans = translation @ final_rotation  # dot product for getting the rigid matrix
    return final_ans


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    # the function can take a lot of time to run.
    EPS = 0.001
    relevant_gap = 100
    min_error = np.inf
    final_ans = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
    # getting through a lot of gap
    for y in range(-relevant_gap, relevant_gap):
        for x in range(-relevant_gap, relevant_gap):
            check = np.array([[1, 0, y],
                              [0, 1, x],
                              [0, 0, 1]], dtype=np.float32)
            moved_img = cv2.warpPerspective(im1, check, im1.shape[::-1])  # moving the image
            mse = np.square(moved_img - im2).mean()  # calculating the error
            if mse < min_error:
                min_error = mse
                final_ans = check
            if mse < EPS:
                return final_ans

    return final_ans


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    # the function can take a lot of time to run.
    EPS = 0.000001
    min_error = np.inf
    final_rotation = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)
    final_rotated_img = 0
    # going through all of the angles
    for alpha in range(360):
        t = np.array([[math.cos(alpha), -math.sin(alpha), 0],
                      [math.sin(alpha), math.cos(alpha), 0],
                      [0, 0, 1]], dtype=np.float32)
        rotated_img = cv2.warpPerspective(im1, t, im1.shape[::-1])  # rotating the image
        mse = np.square((im2 - rotated_img)).mean()  # calculating the error
        if mse < min_error:
            min_error = mse
            final_rotation = t
            final_rotated_img = rotated_img.copy()

    translation = findTranslationCorr(final_rotated_img, im2)  # finding the translation from the rotated
    final_ans = translation @ final_rotation  # dot product for rigid matrix
    return final_ans


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
            check = np.array([i, j, 1]) @ np.linalg.inv(T)  # multiplying in the inverse matrix
            found_x, found_y = int(round(check[0])), int(round(check[1]))  # getting the right x and y
            if 0 <= found_x < im1_height and 0 <= found_y < im1_width:
                ans[i, j] = im1[found_x, found_y]  # inserting the answer matrix

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
        blurred = cv2.GaussianBlur(gauss_pyramid[-1], (5, 5), 0)  # blurring the image
        blurred = blurred[::2, ::2]  # getting the second pixel each time
        gauss_pyramid.append(blurred)  # adding to the list

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
        img_pyramids.append(cv2.pyrDown(img_pyramids[-1]))  # using cv pyramid to get more accurate results
    img_pyramids.reverse()  # reverse to start from the smallest image
    expanded = []
    for i in range(levels-1):
        # expanding the image
        expanded.append(cv2.pyrUp(img_pyramids[i], dstsize=(img_pyramids[i+1].shape[1], img_pyramids[i+1].shape[0])))
    img_pyramids.reverse()
    expanded.reverse()
    laplacians = []
    for i in range(levels-1):
        # computing the laplacian image and adding to the answer list
        laplacians.append(img_pyramids[i] - expanded[i])
    laplacians.append(img_pyramids[len(img_pyramids)-1])
    return laplacians


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    smallest_pyr_image = lap_pyr[-1]  # getting the smallest image
    lap_pyr.reverse()  # reverse to start from the smallest image
    for i in range(1, len(lap_pyr)):
        # expand the image
        smallest_pyr_image = cv2.pyrUp(smallest_pyr_image, dstsize=(lap_pyr[i].shape[1], lap_pyr[i].shape[0]))
        # adding the right laplacian
        smallest_pyr_image += lap_pyr[i]
    lap_pyr.reverse()  # reverse back the lap list
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
    # the formula we have been taught
    naive_blend = img_1 * mask + img_2 * (1 - mask)
    # getting the laplacians list for each image
    img_1_lp = laplaceianReduce(img_1, levels)
    img_2_lp = laplaceianReduce(img_2, levels)
    mask_gp = [mask]
    for i in range(levels-1):
        # calculating mask pyramid
        mask_gp.append(cv2.pyrDown(mask_gp[-1]))  # using cv pyramid to get more accurate results
    img_2_lp.reverse()
    img_1_lp.reverse()
    mask_gp.reverse()
    blending = []
    for i in range(levels):
        new_img = img_1_lp[i] * mask_gp[i] + (1 - mask_gp[i]) * img_2_lp[i]  # the formula we have been taught
        blending.append(new_img)  # adding the calculation
    blending.reverse()
    final_blend = laplaceianExpand(blending)  # calculating laplacian
    return naive_blend, final_blend
