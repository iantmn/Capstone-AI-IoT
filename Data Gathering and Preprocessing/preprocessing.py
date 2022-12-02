import json
import numpy as np
import pprint as p
import matplotlib.pyplot as plt

from data_processing import Preprocessing

from collections.abc import Collection

def main() -> None:
    # try:
    #     with open(r'test-imu-export/testing/testing.3ia0sgrv.ingestion-77d7f974d5-xtfq4.json') as f:
    #         json_data = json.load(f)['payload']
    #         data = json_data['values']
    #         sample_frequency = 1 / (json_data['interval_ms'] / 1000)

    #         data_array = np.array(data)

    #         # TODO making the windows and make the ffts

    #         # Not necessary anymore?
    #         time = np.arange(0, len(data) / sample_frequency, 1 / sample_frequency)

    #         ffts, pwr_peak = fourier(data_array, sample_frequency)
    #         print(pwr_peak)

    #         ffts_shifted = np.fft.fftshift(ffts)
    #         # Not necessary anymore
    #         frequency = np.arange(-sample_frequency/2 + sample_frequency/(2*len(data)), #start
    #                               sample_frequency/2 + sample_frequency/(2*len(data)), #stop
    #                               sample_frequency/(len(data))) #interval

    #         fig, axes = plt.subplots(2, 1)
    #         axes[0].plot(time, data_array, label=['X', 'Y', 'Z'])
    #         axes[0].legend(loc='upper left')
    #         axes[0].set_title("Accelerometer data")
    #         axes[1].plot(frequency, abs(ffts_shifted), label=['X', 'Y', 'Z'])
    #         axes[1].legend(loc='upper left')
    #         axes[1].set_title("Fourier Transform")
    #         plt.show()
            
    # except FileNotFoundError:
    #     print("File not found!")
    # windowing(r'test-imu-export\testing\testing.3ia0sgrv.ingestion-77d7f974d5-xtfq4.json')
    # windowing(r'Accelerometer Data 0000.txt')

    # mean, std, min_inter, max_inter = get_time_stats(r'Data files/Accelerometer Data X-axis.txt')
    # mean, std, min_inter, max_inter = Preprocessing.get_time_stats(r'data-lopen\Data Timo\Accelerometer Data 2022-11-29 11-11-24.txt')
    # print(mean, std, min_inter, max_inter, std / mean * 1000) # Mean, std and the promillage of the std to the mean
    # plot_accelerometer(r'Data files/Accelerometer Data X-axis.txt', 0.5)
    # plot_accelerometer(r'data-lopen\Data Timo\Accelerometer Data 2022-11-29 11-11-24.txt', 0.5, 3)
    # windowing(r'Data files/Accelerometer Data X-axis.txt', action_ID='Walking', label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=1, offset=0.1, epsilon=0.03, do_plot=False)
    # windowing(r'data-lopen\Data Timo\Accelerometer Data 2022-11-29 11-11-24.txt', action_ID='Walking', label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=1, offset=0.1, epsilon=0.03, do_plot=False)
    pre = Preprocessing('Walking')
    pre.windowing(r'data-lopen\Data Timo\Accelerometer Data 2022-11-29 11-54-03.txt', label='stairs_down',
                start_offset=0.5, stop_offset=3, size=40, offset=0.1, epsilon=0.03, do_plot=False)


if __name__ == "__main__":
    main()
