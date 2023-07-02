import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

def go_get_them(file_path):

    df = pd.read_csv(file_path, header=None)

    range_values = df.iloc[0,1:].values

    real_images = []
    imag_images = []

    for index, row in df.iloc[1:].iterrows():
        parameter_value = row[0]
        complex_values = row[1:]

        real_values = np.real(complex_values.astype(complex))
        imag_values = np.imag(complex_values.astype(complex))

        real_images.append(real_values)
        imag_images.append(imag_values)

    real_images = np.array(real_images)
    imag_images = np.array(imag_images)

    imag_images = np.nan_to_num(imag_images)
    real_images = np.nan_to_num(real_images)

    real_data = (real_images - np.min(real_images)) / (np.max(real_images) - np.min(real_images))
    imag_data = (imag_images - np.min(imag_images)) / (np.max(imag_images) - np.min(imag_images))

    magnitude = np.sqrt(np.square(real_data) + np.square(imag_data))
    phase = np.arctan2(imag_data, real_data)

    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))

    phaseMreal = phase - real_data
    imagMreal = imag_data - real_data

    phaseMreal = (phaseMreal - np.min(phaseMreal)) / (np.max(phaseMreal) - np.min(phaseMreal))
    imagMreal = (imagMreal - np.min(imagMreal)) / (np.max(imagMreal) - np.min(imagMreal))

    # Create the subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Set the images and titles
    ax1.imshow(real_data, cmap='binary_r')
    ax1.set_title('Real Part')
    ax2.imshow(imag_data, cmap='binary_r')
    ax2.set_title('Imaginary Part')
    ax3.imshow(magnitude, cmap='binary_r')
    ax3.set_title('Magnitude')
    ax4.imshow(phase, cmap='binary_r')
    ax4.set_title('Phase')

    # Adjust the spacing between rows
    plt.subplots_adjust(hspace=0.5)  # Increase the value to increase spacing
    plt.savefig('1.png')
    plt.close()
    return real_data, imag_data, magnitude, phase, phaseMreal, imagMreal


def cut_me(data):
    """
    POC: correction using naive separation of areas with samples with lines cut
    """
    data_uint8 = (data * 255).astype(np.uint8)

    _, binary = cv2.threshold(data_uint8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    edges = cv2.Canny(binary,100,200,apertureSize = 3)

    edge_points_per_row = np.sum(edges != 0, axis=1)

    threshold = 0.10 * np.max(edge_points_per_row)

    start_candidates = np.where(np.diff(edge_points_per_row) > threshold)[0]
    start_of_object = start_candidates[0] if len(start_candidates) > 0 else None

    end_candidates = np.where(np.flip(np.diff(edge_points_per_row)) < -threshold)[0]
    end_of_object = len(edge_points_per_row) - end_candidates[0] if len(end_candidates) > 0 else None

    if start_of_object is None or end_of_object is None:
        return None
    else:
        # Now 'start_of_object' and 'end_of_object' should contain the y-coordinates of the start and end of your object

        # You can draw these lines onto your image as follows:
        cv2.line(data_uint8, (0, start_of_object), (data_uint8.shape[1] - 1, start_of_object), (0, 0, 255), 2)
        cv2.line(data_uint8, (0, end_of_object), (data_uint8.shape[1] - 1, end_of_object), (0, 0, 255), 2)



    # plt.imshow(edges, cmap='binary_r')
    # plt.colorbar()
    # plt.title("edg")
    # plt.show()
    # plt.imshow(data_uint8, cmap='binary_r')
    # plt.colorbar()
    # plt.title("edges")
    # plt.show()

    rows, _ = data.shape
    up = start_of_object
    dp = rows - end_of_object

    return up, dp



def thereshold_background_correction(phaseMreal):

    """
    POC: auto - correction using thereshold. just horrible approach for datasets like with huge noises overlaps
    the samples. Better than lines cut for datasets when we dont see the whole probe or dont have much background.
    still may be need for manually adjust parameters based of type of object or streight of the cable noises
    """
    # orig and naive - smoll noises, lots of objects
    # Apply Otsu's thresholding
    thresh = threshold_otsu(phaseMreal)
    binary = phaseMreal > thresh + 0.35 # -0.1 # TODO automate this. for now may be need to manully adjust thereshold
    '''
        thresh = threshold_otsu(phaseMreal)

        columns_left = phaseMreal.shape[1] // 2

        phaseMreal_left = phaseMreal[:, :columns_left]
        phaseMreal_right = phaseMreal[:, columns_left:]

        binary_left = phaseMreal_left > thresh - 0.3
        binary_right = phaseMreal_right > (thresh + 0.3)

        binary = np.hstack((binary_left, binary_right))
    '''

    ''' #good to separate objects/material changes, will completly remove teh noises.TODO automatic detection
    columns_left = 50 # adjust this value as per your requirement

    # Split the image into two parts
    phaseMreal_left = phaseMreal[:, :columns_left]
    phaseMreal_right = phaseMreal[:, columns_left:]

    # Apply Otsu's thresholding to each part
    thresh_left = threshold_otsu(phaseMreal_left)
    thresh_right = threshold_otsu(phaseMreal_right)

    # Threshold each part
    binary_left = phaseMreal_left > thresh_left
    binary_right = phaseMreal_right > thresh_right

    # Combine the parts back together
    binary = np.hstack((binary_left, binary_right))

    '''
    background_mask = binary

    phase_masked = phaseMreal.copy()
    phase_masked[~background_mask] = np.nan

    mean_value_bg = np.nanmean(phase_masked, axis=0)

    phase_bg_corr = phaseMreal - mean_value_bg

    phase_bg_corr = (phase_bg_corr - np.nanmin(phase_bg_corr)) / (np.nanmax(phase_bg_corr) - np.nanmin(phase_bg_corr))

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    #
    # ax1.imshow(phase_masked, cmap='binary_r')
    # ax1.set_title('masked')
    # ax2.imshow(phase_bg_corr, cmap='binary_r')
    # ax2.set_title('Corrected Phase')
    #
    # plt.show()

    # fig, (ax1) = plt.subplots(1, 1)
    #
    # # ax1.imshow(phase_masked, cmap='binary_r')
    # # ax1.set_title('masked')
    # ax1.imshow(phase_bg_corr, cmap='binary_r')
    # ax1.set_title('Corrected Phase')

    return phase_bg_corr


def background_correction(data, up, dp):
    """
    background correction based on cuts
    """

    mean_value_first = np.mean(data[:up])
    mean_value_last = np.mean(data[-1 * dp:])

    difference_first = data[:up] - mean_value_first
    difference_last = data[-1 * dp:] - mean_value_last

    mean_difference_per_column_first = np.mean(difference_first, axis=0)
    mean_difference_per_column_last = np.mean(difference_last, axis=0)

    average_mean_difference = (mean_difference_per_column_first + mean_difference_per_column_last) / 2

    if(dp>2):
        corrected_data = data - average_mean_difference
    else:
        corrected_data = data - mean_difference_per_column_first

    corrected_data = (corrected_data - np.min(corrected_data)) / (np.max(corrected_data) - np.min(corrected_data))

    return corrected_data


def visualize_data(data1, data2, title1=" ", title2=" ", filename1='1.png', filename2='2.png'):
    """
    3D and 2D visualisation
    """
    normalized_amplitude = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))

    normalized_phase = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))

    phase_color = plt.cm.hsv(normalized_phase)

    phase_color[..., 3] = normalized_amplitude

    plt.imshow(phase_color)
    plt.title(title1)
    plt.savefig(filename1)
    plt.close()

    x = np.arange(data1.shape[1])
    y = np.arange(data1.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, data2, color='b', alpha=0.5)

    ax.plot_surface(x, y, data1, color='r', alpha=0.5)

    plt.title(title2)
    plt.savefig(filename2)
    plt.close()




