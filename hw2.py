import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Tk, Label, Button

window = tk.Tk()
window.title('HW2_GUI')
window.geometry('600x500')
window.resizable(False, False)

def load_raw_img(file_path):
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype = np.uint8)
    return raw_data.reshape((512,512))

img_a = load_raw_img('pirate_a.raw')
img_b = load_raw_img('pirate_b.raw')
first_lbl = tk.Label(text='1. Enhancement Using Spatial Filters', bg='gray', fg='white' ,height=1)
first_lbl.pack()

def avg_filter():

    avg_kernel = np.ones((3,3), np.float32)/9
    img_a_avg = cv2.filter2D(img_a, -1, avg_kernel)
    img_b_avg = cv2.filter2D(img_b, -1, avg_kernel)

    #pirate_a original and averaging
    plt.subplot(2,2,1)
    plt.title('original image A')
    plt.imshow(img_a, cmap='gray')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.title('pirate A with 3x3 averaging')
    plt.imshow(img_a_avg, cmap='gray')
    plt.axis('off')

    #pirate_b oringinal and averaging
    plt.subplot(2,2,3)
    plt.imshow(img_b, cmap='gray')
    plt.title('original pirate B')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(img_b_avg, cmap='gray')
    plt.title('pirate B with 3x3 averaging')
    plt.axis('off')

    plt.show()

def laplacian():
    img_a_med = cv2.medianBlur(img_a, 3)
    best_image = img_a_med

    laplacian_filtered = cv2.Laplacian(best_image, cv2.CV_64F, ksize=3)
    laplacian_filtered = cv2.convertScaleAbs(laplacian_filtered)

    plt.subplot(1,2,1)
    plt.title('Best image')
    plt.imshow(best_image, cmap='gray')

    plt.subplot(1,2,2)
    plt.title('laplacian filtered image')
    plt.imshow(laplacian_filtered, cmap='gray')
    plt.axis('off')
    plt.show() 


avg_btn = tk.Button(text='average filter', command=avg_filter)
avg_btn.pack()


def med_filter():
    img_a_med = cv2.medianBlur(img_a, 3)
    img_b_med = cv2.medianBlur(img_b, 3)
    #pirate_a original and median 
    plt.subplot(2,2,1)
    plt.title('original image A')
    plt.imshow(img_a, cmap='gray')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.title('pirate A with 3x3 median')
    plt.imshow(img_a_med, cmap='gray')
    plt.axis('off')

    #pirate_b original and median 
    plt.subplot(2,2,3)
    plt.title('original image B')
    plt.imshow(img_b, cmap='gray')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.title('pirate B with 3x3 median')
    plt.imshow(img_b_med, cmap='gray')
    plt.axis('off')

    plt.show()

med_btn = tk.Button(text='median filter', command=med_filter)
med_btn.pack()
lap_btn = tk.Button(text='laplacian filter (pirate A with 3x3 median)', command=laplacian)
lap_btn.pack()

#7x7 averaging filter
img2 = cv2.imread('BarTest.tif', cv2.IMREAD_GRAYSCALE)
second_lbl = tk.Label(text='2. applying filters to BarTest.tif', bg='gray', fg='white' ,height=1)
second_lbl.pack()

def mean_7x7():
    img_mean7 = cv2.blur(img2, (7,7))
    
    plt.subplot(2,1,1)
    plt.title('original image')
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title('7x7 arithmetic image')
    plt.imshow(img_mean7, cmap='gray')
    plt.axis('off')

    plt.show()

mean7x7_btn = tk.Button(text='7x7 arithmetic mean filter', command=mean_7x7)
mean7x7_btn.pack()

def mean_3x3():
    img_mean3 = cv2.blur(img2, (3,3))
    
    plt.subplot(2,1,1)
    plt.title('original image')
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title('3x3 arithmetic image')
    plt.imshow(img_mean3, cmap='gray')
    plt.axis('off')

    plt.show()

mean3x3_btn = tk.Button(text='3x3 arithmetic mean filter', command=mean_3x3)
mean3x3_btn.pack()

def median_7x7():
    img_median7 = cv2.medianBlur(img2, 7)
    
    plt.subplot(2,1,1)
    plt.title('original image')
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title('7x7 median image')
    plt.imshow(img_median7, cmap='gray')
    plt.axis('off')

    plt.show()

median7x7_btn = tk.Button(text='7x7 median filter', command=median_7x7)
median7x7_btn.pack()

def median_3x3():
    img_median3 = cv2.medianBlur(img2, 3)
    
    plt.subplot(2,1,1)
    plt.title('original image')
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title('3x3 median image')
    plt.imshow(img_median3, cmap='gray')
    plt.axis('off')

    plt.show()

median3x3_btn = tk.Button(text='3x3 median filter', command=median_3x3)
median3x3_btn.pack()

#2D-FFT
img3 = cv2.imread('lenna_gray.tif', cv2.IMREAD_GRAYSCALE)
img3_f = np.fft.fft2(img3)
img3_f_shift = np.fft.fftshift(img3_f)
third_lbl = tk.Label(text='3. 2D-FFT to lenna_gray.tif and obtain magnitude and phase', bg='gray', fg='white' ,height=1)
third_lbl.pack()


def FFT():
    magnitude_spectrum = 20*np.log(np.abs(img3_f_shift))

    plt.subplot(1,2,1)
    plt.imshow(img3, cmap='gray')
    plt.title('original image')

    plt.subplot(1,2,2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('2D-FFT Spectrum')
    
    plt.show()

FFT_btn = tk.Button(text='2D-FFT',command=FFT)
FFT_btn.pack()

def iFFT():
    magnitude = np.abs(img3_f_shift)
    magnitude_only = np.fft.ifft2(np.fft.ifftshift(magnitude))
    magnitude_only_img = np.abs(magnitude_only)

    phase = np.angle(img3_f_shift)
    phase_only = np.fft.ifftshift(np.fft.ifftshift(np.exp(1j*phase)))
    phase_only_img = np.abs(phase_only)

    plt.subplot(1,2,1)
    plt.imshow(magnitude_only_img, cmap='gray')
    plt.title('magnitude only')

    plt.subplot(1,2,2)
    plt.imshow(phase_only_img, cmap='gray')
    plt.title('phase only')

    plt.show()

iFFT_btn = tk.Button(text='phase and amplitude only',command=iFFT)
iFFT_btn.pack()

#DIP_image
img4 = cv2.imread('DIP_image.tif', cv2.IMREAD_GRAYSCALE)
fourth_lbl = tk.Label(text='4. process DIP_image.tif by the giving steps', bg='gray', fg='white' ,height=1)
fourth_lbl.pack()

def fourier_trans():
    img4_shift = img4*np.fromfunction(lambda x,y:(-1)**(x+y), img4.shape)
    img4_shift_img = np.abs(img4_shift)

    img4_dft = np.fft.fft2(img4_shift)
    img4_dft_img = np.abs(img4_dft)

    img4_dft_conjugate = np.conj(img4_dft)
    img4_dft_conjugate_img = np.abs(img4_dft_conjugate)

    img4_dft_inverse = np.fft.ifft2(img4_dft_conjugate)
    img4_dft_inverse_img = np.abs(img4_dft_inverse)

    img4_real = np.real(img4_dft_inverse)
    img4_final = img4_real*np.fromfunction(lambda x,y:(-1)**(x+y),img4.shape)
    img4_final_img = np.abs(img4_final)

    plt.figure(figsize=(12,6))

    plt.subplot(2,3,1), plt.imshow(img4, cmap='gray'), plt.title('original image')
    plt.subplot(2,3,2), plt.imshow(img4_shift_img, cmap='gray'), plt.title('1. original image*(-1)^(x+y)')
    plt.subplot(2,3,3), plt.imshow(img4_dft_img, cmap='gray'), plt.title('2. discrete fourier transform')
    plt.subplot(2,3,4), plt.imshow(img4_dft_conjugate_img, cmap='gray'), plt.title('3. taking the complex conjugate of DFT')
    plt.subplot(2,3,5), plt.imshow(img4_dft_inverse_img, cmap='gray'), plt.title('4. inverse DFT')
    plt.subplot(2,3,6), plt.imshow(img4_final_img, cmap='gray'), plt.title('5. original image*real part')

    plt.tight_layout()
    plt.show()

DFT_btn = tk.Button(text='DFT and IDFT', command=fourier_trans)
DFT_btn.pack()


window.mainloop()
