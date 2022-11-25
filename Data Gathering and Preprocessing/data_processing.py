import json
import numpy as np
from collections.abc import Collection


def fourier(data: Collection, sample_frequency: float, epsilon: float = 0.1) -> tuple[np.ndarray[float], list[float]]:
    """In the function: zeropadding, fft time data columns, spectral analysis

    Args:
        data (Collection): numpy array of multiple sensors. Rows are data, columns are different sensors

    Returns:
        tuple[np.array[float]]: tuple of features for this specific windowed sample.
    """    
    zero_padding = 0  # amount of zeros used for zero_padding
    data = np.array(data)
    if not zero_padding == 0:
        data = np.append(data, np.zeros(shape=(zero_padding, 3))) # zero pad
    ffts = np.fft.fft2(data, axes=(-2, -2, -2))
    peaks = peak_value_frequency(ffts, sample_frequency)
    pwrs = cntrd_pwr(ffts, sample_frequency, epsilon)
    # print(peaks)
    # print(pwrs)
    return ffts, [peak + pwr for peak, pwr in zip(peaks, pwrs)]

def cntrd_pwr(dataset: Collection, sampling_frequency: float, epsilon: float = 0.1) -> list[tuple[float, float]]:
    """Finds the maximum power of 3 sensors and the frequency for which the energy at the left is the same as the energy on the right
    takes a threshold of epsilon*max_value

    Args:
        dataset (Collection): frequency domain dataset
        epsilon (float): parameter to determine what samples play a part in the centroid calcultations. range 0-1
        sampling_frequency (float): sampling frequency

    Returns:
        list[tuple[int, int, int], tuple[float, float, float]]: list of a tuple of the centroid frequency and 
        the maximum power of all sensors
    """
    # print(np.array(dataset).shape)
    
    total_power = [0., 0., 0.]  # 3 sensors total power
    centroid = [0., 0., 0.]    # 3 sensors centroid, not returned
    index = [0, 0, 0]  # index of the centroid
    # compute for all three sensors
    for i in range(3):
        maxm = max(dataset[:, i])
        length = len(dataset[:, i])
        # print(f'maximum: {maxm}, length {length}')
        # sum power and sum all values above the threshold
        for j in range(length):
            total_power[i] += abs(dataset[j][i])
            if abs(dataset[j][i]) > epsilon*maxm:
                centroid[i] += abs(dataset[j][i])
        goal = centroid[i]/2
        centroid[i] = 0.
        # reset j, go through the dataset again and stop when you surpass centroid/2
        j = 0
        while centroid[i] < goal:
            if abs(dataset[j][i]) > epsilon*maxm:
                centroid[i] += abs(dataset[j][i])
            j += 1
            index[i] += 1
    return [(e/sampling_frequency, pwr) for e, pwr in zip(index, total_power)]
    # return [tuple([e/sampling_frequency for e in index]), tuple(total_power)]


def peak_value_frequency(dataset: Collection, sampling_frequency: float) -> list[tuple[int, float]]:
    """find the frequency that is the most present in a PSD. Do not include DC component

    Args:
        dataset (Collection): frequency domain dataset
        sampling_frequncy (float): sampling_frequency

    Returns:
        list[tuple[int, float]]: list with a tuple containing the index and the corresponding frequency
    """
    
    # List to save the peaks and there frequencies in
    peaks: list[tuple[int, float]] = []
    # Casting the dataset collection into a numpy array
    data = np.array(dataset)
    
    # Finding the sample in the middle of the collection (second half is a mirrored copy of the first half)
    mid = data.shape[0] // 2
    # Finding the maximum value and its location and its frequency for each column of the array
    for i in range(data.shape[1]):
        max_value = max(abs(data[1:mid, i]))
        loc = np.where(abs(data[1:mid, i]) == max_value)[0][0] + 1
        # print(loc, abs(data[loc, i] / (interval * data.shape[0])), abs(data[loc, i] / (interval * data.shape[0])))
        peaks.append((loc, loc * sampling_frequency / (data.shape[0])))
    return peaks

def windowing(filename: str, size: float = 2, offset: float = 0.2, sampling_frequency: float = 0) -> np.ndarray:
    """_summary_

    Args:
        filename (str): location 
        size (float, optional): _description_. Defaults to 2.
        offset (float, optional): _description_. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """    
    try:
        with open(filename) as f:
            if filename.split('.')[-1].lower() == 'json':
                json_data = json.load(f)['payload']
                data = json_data['values']
                sample_frequency = 1 / (json_data['interval_ms'] / 1000)
            elif filename.split('.')[-1].lower() == 'csv':
                pass
            elif filename.txt('.')[-1].lower() == 'txt':
                pass
            else:
                raise NotImplementedError(f"Filetype {filename.split('.')[-1]} is not implemented")

            data_array = np.array(data)

            if sampling_frequency == 0:
                raise ValueError('No value was given for sampling frequency')
            else:
                samples_amount = size * sampling_frequency
                shift_amount = offset * sampling_frequency
            starting_index = 0
            while next(f):
                pass
    except FileNotFoundError:
        print(f"File {filename} at the relative path not found!")

        
def window(dataset, size, offset, sample_frequency):
    pass
