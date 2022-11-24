import json
import numpy as np
import pprint as p
import matplotlib.pyplot as plt

from collections.abc import Collection

def main() -> None:
    try:
        with open(r'test-imu-export/testing/testing.3ia0sgrv.ingestion-77d7f974d5-xtfq4.json') as f:
            json_data = json.load(f)['payload']
            data = json_data['values']
            interval = json_data['interval_ms'] / 1000

            data_array = np.array(data)
            time = np.arange(0, interval * len(data), interval)

            ffts = np.fft.fft2(data_array, axes=(-2, -2, -2))
            ffts_shifted = np.fft.fftshift(ffts)
            frequency = np.arange(-1/(2*interval) + 1/(2*interval*len(data)),
                                  1/(2*interval) + 1/(2*interval*len(data)), 1/(interval*len(data)))
            # (S, f) = plt.psd(data_array[:, 0], 1/interval)

            print(cntrd_pwr(ffts[:ffts.shape[0]//2, :], 0.0001))
            print(peak_value_frequency(ffts, interval))

            fig, axes = plt.subplots(2, 1)
            axes[0].plot(time, data_array, label=['X', 'Y', 'Z'])
            axes[0].legend()
            axes[0].set_title("Accelerometer data")
            axes[1].plot(frequency, ffts_shifted, label=['X', 'Y', 'Z'])
            axes[1].legend(loc='upper left')
            axes[1].set_title("Fourier Transform")
            # axes[2].plot(f, S, label=['X', 'Y', 'Z'])
            # axes[2].legend(loc='upper left')
            # axes[2].set_title("Fourier Transform")
            plt.show()
            
    except FileNotFoundError:
        print("File not found!")

def cntrd_pwr(dataset: Collection, epsilon: float) -> tuple[list[float], list[float]]:
    """epsilon should be between 0 and 1"""
    print(np.array(dataset).shape)
    total_power = [0., 0., 0.]
    centroid = [0., 0., 0.]
    for i in range(3):  # 0 to 3
        maxm = max(dataset[:, i])
        length = len(dataset[:, i])
        print(f'maximum: {maxm}, length {length}')
        for j in range(length):
            total_power[i] += abs(dataset[j][i])
            if abs(dataset[j][i]) > epsilon*maxm:
                centroid[i] += i*abs(dataset[j][i])
        centroid[i] = centroid[i]/length
    return centroid, total_power

def peak_value_frequency(dataset: Collection, interval: float) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    data = np.array(dataset)
    mid = data.shape[0] // 2
    for i in range(data.shape[1]):
        loc = np.where(max(data[1:mid, i]))[0] + 1
        print(loc)
        peaks.append((loc, data[loc] / (interval * data.shape[0])))
    return peaks

if __name__ == "__main__":
    main()
