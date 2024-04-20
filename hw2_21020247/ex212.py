import numpy as np
from skimage import io as io_url
import matplotlib.pyplot as plt
import cv2


def DFT_slow(data):
    N = len(data) 
    DFT = np.zeros(N, dtype=np.complex64)

    for k in range(N):
        for n in range(N):
            DFT[k] += data[n] * np.exp(-2j * np.pi * k * n / N)

    return DFT


def show_img(origin, row_fft, row_col_fft):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    axs[0].imshow(origin, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(np.log(np.abs(np.fft.fftshift(row_fft))), cmap='gray')
    axs[1].set_title('Row-wise FFT')
    axs[1].axis('off')
    axs[2].imshow((np.log(np.abs(np.fft.fftshift(row_col_fft)))), cmap='gray')
    axs[2].set_title('Column-wise FFT')
    axs[2].axis('off')
    plt.show()


def DFT_2D(gray_img):
    row_fft = np.fft.fft2(gray_img)
    row_col_fft = np.fft.fftshift(row_fft)

    return row_fft, row_col_fft


if __name__ == '__main__':
    img = io_url.imread('https://img2.zergnet.com/2309662_300.jpg')
    gray_img = np.mean(img, -1)
    row_fft, row_col_fft = DFT_2D(gray_img)
    show_img(gray_img, row_fft, row_col_fft)
