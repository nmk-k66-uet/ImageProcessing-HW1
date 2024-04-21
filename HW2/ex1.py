import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    # Need to implement here
    image_array = np.array(img)
    padded_img = np.pad(image_array, pad_width= filter_size // 2, mode = 'edge')
    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    # Need to implement here
    img_array = padding_img(img, filter_size)
    h, w = img.shape
    img_new = np.zeros([h, w])
    pad_width = filter_size // 2
    for i in range(pad_width, h + pad_width): 
        for j in range(pad_width, w + pad_width): 
            temp = img_array[i - pad_width : i + pad_width + 1, j - pad_width : j + pad_width + 1]
            mean = np.mean(temp)
            img_new[i - pad_width, j - pad_width] = mean
    img_new = img_new.astype(np.uint8)
    return img_new
            

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
  # Need to implement here
    img_array = padding_img(img, filter_size)
    h, w = img.shape
    img_new = np.zeros([h, w])
    pad_width = filter_size // 2
    for i in range(pad_width, h + pad_width): 
        for j in range(pad_width, w + pad_width): 
            temp = img_array[i - pad_width : i + pad_width + 1, j - pad_width : j + pad_width + 1]
            median = np.median(temp)
            img_new[i - pad_width, j - pad_width] = median
    img_new = img_new.astype(np.uint8)
    return img_new

def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here
    gt_img = np.array(gt_img, dtype='double')
    smooth_img = np.array(smooth_img, dtype='double')
    mse = np.mean((gt_img - smooth_img) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png" # <- need to specify the path to the noise image

    img_gt = "ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    org_img = read_img(img_gt)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(org_img, median_smoothed_img))

