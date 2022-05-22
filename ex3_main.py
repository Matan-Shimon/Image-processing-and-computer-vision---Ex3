import matplotlib.pyplot as plt
import numpy as np

from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float32)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float32), img_2.astype(np.float32), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float32)
    cv_warp = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    start = time.time()
    uv = opticalFlowPyrLK(img_1.astype(np.float32), cv_warp.astype(np.float32), 10, stepSize=20, winSize=5)
    end = time.time()
    print("Time: {:.2f}".format(end - start))
    points = np.where(np.not_equal(uv[:, :], np.zeros((2))))
    uv = uv[points[0], points[1]]
    print(np.median(uv, 0))
    print(np.mean(uv, 0))
    plt.imshow(cv_warp, cmap='gray')
    plt.quiver(points[1], points[0], uv[:, 0], uv[:, 1], color='r')
    plt.show()


def compareLK(img_path):
    print("Compare LK")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 3],
                  [0, 1, -3],
                  [0, 0, 1]], dtype=np.float32)
    cv_warp = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    start = time.time()
    regular_uv = opticalFlow(img_1.astype(np.float32), cv_warp.astype(np.float32), step_size=20, win_size=5)[1]
    median_regular = np.median(regular_uv, 0)
    end = time.time()
    print("Time of regular lk: {:.2f}".format(start - end))
    start = time.time()
    pyr_uv = opticalFlowPyrLK(img_1.astype(np.float32), cv_warp.astype(np.float32), 7, stepSize=20, winSize=5)
    end = time.time()
    print("Time of regular lk: {:.2f}".format(start - end))
    median_pyr = np.ma.median(np.ma.masked_where(pyr_uv == np.zeros((2)), pyr_uv), axis=(0, 1)).filled(0)
    right_values = np.array([3, -3])
    regular = np.power(np.median(regular_uv, 0) - right_values, 2).sum() / 2
    pyr = np.power(median_pyr - right_values, 2).sum() / 2
    print('regular median:', median_regular)
    print('pyramid median:', median_pyr)
    print('regular average mistake:', regular)
    print('pyramid average mistake:', pyr)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def translationlkdemo(img_path):
    print("Image Warping demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    translation = np.array([[1, 0, -30],
                            [0, 1, 50],
                            [0, 0, 1]], dtype=np.float32)
    cv_warp = cv2.warpPerspective(img_1, translation, img_1.shape[::-1])
    start = time.time()
    my_translation = findTranslationLK(img_1, cv_warp)
    end = time.time()
    print("Time: {:.2f}".format(start - end))
    my_warp = cv2.warpPerspective(img_1, translation, img_1.shape[::-1])
    print("mse = ", np.square(cv_warp - my_warp).mean())
    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].imshow(img_1)
    ax[0].set_title('original')
    ax[1].imshow(cv_warp)
    ax[1].set_title('cv translation')
    ax[2].imshow(my_warp)
    ax[2].set_title('my translation')
    plt.show()


def rigidlkdemo(img_path):
    print("rigid lk demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    translation = np.array([[1, 0, -4],
                            [0, 1, 3],
                            [0, 0, 1]], dtype=np.float32)
    angle = -0.5
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]], dtype=np.float32)

    rigid = translation @ rotation
    cv_warp = cv2.warpPerspective(img_1, rigid, img_1.shape[::-1])
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('img2 original rigid')
    ax[0].imshow(cv_warp, cmap='gray')
    start = time.time()
    my_rigid = findRigidLK(img_1, cv_warp)
    end = time.time()
    print("Time: {:.2f}".format(start - end))
    print("difference in rigid")
    print((rigid - my_rigid).sum())
    my_warp = cv2.warpPerspective(img_1, my_rigid, img_1.shape[::-1])
    print("mse= ", np.square(my_warp, cv_warp).mean())
    ax[1].set_title('img2 my rigid')
    ax[1].imshow(my_warp, cmap='gray')
    plt.show()


def imageWarpingDemo(img_path):
    print("Image Warping demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # tran_img = cv2.cvtColor(cv2.imread('input/TransHome.jpg'), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -5],
                  [0, 1, 4],
                  [0, 0, 1]], dtype=np.float32)
    cv_warp = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    start = time.time()
    my_warp = warpImages(img_1, cv_warp, t)
    end = time.time()
    print("Time: {:.2f}".format(start - end))
    print("mse: ", np.square(cv_warp-my_warp).mean())
    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].imshow(img_1)
    ax[0].set_title('original')
    ax[1].imshow(cv_warp)
    ax[1].set_title('cv warping')
    ax[2].imshow(my_warp)
    ax[2].set_title('my back warping')
    plt.show()


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())
    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)
    translationlkdemo(img_path)
    rigidlkdemo(img_path)
    imageWarpingDemo(img_path)
    img_path = 'input/pyr_bit.jpg'
    pyrGaussianDemo(img_path)
    pyrLaplacianDemo(img_path)
    blendDemo()


if __name__ == '__main__':
    main()
