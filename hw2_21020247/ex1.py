import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    height, width = img.shape[:2]
    pad_width = filter_size // 2
    padded_img = np.pad(img, pad_width, mode='edge')
    return padded_img

def mean_filter(img, filter_size=3):
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neighborhood = padded_img[i:i+filter_size, j:j+filter_size]
            mean_value = np.mean(neighborhood)
            smoothed_img[i, j] = mean_value

    return smoothed_img.astype(np.uint8) 


def median_filter(img, filter_size=3):
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neighborhood = padded_img[i:i+filter_size, j:j+filter_size]
            median_value = np.median(neighborhood)
            smoothed_img[i, j] = median_value

    return smoothed_img


def psnr(gt_img, smooth_img):
    assert gt_img.shape == smooth_img.shape
    assert gt_img.dtype == smooth_img.dtype
    mse = np.mean((gt_img - smooth_img) ** 2)
    max_pixel_value = np.iinfo(gt_img.dtype).max
    psnr_score = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr_score


def show_res(before_img, after_img):
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "./ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "./ex1_images/ori_img.png"  # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

