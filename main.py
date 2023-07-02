#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import filter
import argparse

def whole(imag_data, phaseMreal, imagMreal,phase, magnitude):
    up, dp = filter.cut_me(imag_data)
    phaseMrealFiltr = filter.background_correction(phaseMreal, up, dp)
    plt.imshow(phaseMreal, cmap='binary_r');plt.colorbar();plt.title("Phase filter with real part");plt.savefig("2")
    plt.imshow(phaseMrealFiltr, cmap='binary_r');plt.title("Phase|real background correction");plt.savefig("3")
    imagMrealFiltr = filter.background_correction(imagMreal, up, dp)
    plt.imshow(imagMreal, cmap='binary_r');plt.title("Imag filter with real part");plt.savefig("4")
    plt.imshow(imagMrealFiltr, cmap='binary_r');plt.title("Imag|real background correction");plt.savefig("5")
    phaseFiltr = filter.background_correction(phase, up, dp)
    plt.imshow(phaseFiltr, cmap='binary_r'); plt.title("Phase background correction"); plt.savefig("6")
    magnitudeFiltr = filter.background_correction(magnitude, up, dp)
    plt.imshow(magnitudeFiltr, cmap='binary_r'); plt.title("Magnitude background correction");plt.savefig("7")
    return phaseFiltr, magnitudeFiltr

def part(phaseMreal, imagMreal,phase, magnitude):
    phaseMrealFiltr = filter.thereshold_background_correction(phaseMreal)
    plt.imshow(phaseMreal, cmap='binary_r');plt.colorbar();plt.title("Phase filter with real part");plt.savefig("2")
    plt.imshow(phaseMrealFiltr, cmap='binary_r');plt.title("Phase|real background correction");plt.savefig("3")
    imagMrealFiltr = filter.thereshold_background_correction(imagMreal)
    plt.imshow(imagMreal, cmap='binary_r');plt.title("Imag filter with real part");plt.savefig("4")
    plt.imshow(imagMrealFiltr, cmap='binary_r');plt.title("Imag|real background correction");plt.savefig("5")
    phaseFiltr = filter.thereshold_background_correction(phase)
    plt.imshow(phaseFiltr, cmap='binary_r'); plt.title("Phase background correction"); plt.savefig("6")
    # magnitudeFiltr = filter.thereshold_background_correction(magnitude)
    # plt.imshow(magnitudeFiltr, cmap='binary_r'); plt.title("Magnitude background correction");plt.savefig("7")

    return phaseFiltr, magnitude

def main(file_path, opt):

    real_data, imag_data, magnitude, phase, phaseMreal, imagMreal = filter.go_get_them(file_path)
    if opt == 1:
        phaseFiltr, magnitudeFiltr = part(phaseMreal, imagMreal, phase, magnitude)
    else:
        phaseFiltr, magnitudeFiltr = whole(imag_data, phaseMreal, imagMreal, phase, magnitude)
    filter.visualize_data(phase, magnitude, "phase - color, amplitude - brightness", "phase - red, amplitude - blue",
                          "8.png", "9.png")
    filter.visualize_data(phaseFiltr, magnitude, "phase - color, amplitude - brightness, pre-filtered",
                          "phase - red, amplitude - blue, pre-filtered", "10.png", "11.png")


if __name__ == "__main__":
        print("About to save image...")
        parser = argparse.ArgumentParser(description='Process some data.')
        parser.add_argument('-f', '--file', required=True, help='The file to process')
        parser.add_argument('-p', '--param', required=True, help='The parameter for processing')
        args = parser.parse_args()

        main(args.file, int(args.param))

2