import json
import numpy as np
import math as m
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

def windowing(filename: str, size: float = 2, offset: float = 0.2) -> np.ndarray:
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
            file_extension = filename.split('.')[-1].lower()
            if file_extension == 'json':
                json_data = json.load(f)['payload']
                # data = json_data['values']
                data_array = np.array(json_data['values'])
                sample_frequency = 1 / (json_data['interval_ms'] / 1000)
            elif file_extension == 'csv':
                raise NotImplementedError(f"Filetype {filename.split('.')[-1]} is not implemented")
                # sample rate seems to not be constant in the 'test wisser' file...
            elif file_extension == 'txt':
                # Skip the header lines
                for _ in range(5): f.readline() # TODO change the range back to 4!

                # Find the sample frequency by dividing 1 by the difference in time of two samples
                t0 = float(f.readline().strip().split(',')[0])
                t1 = float(f.readline().strip().split(',')[0])
                sample_frequency = round(1 / (t1 - t0), 2)

                # Finding the last timestamp of the data
                last_point = 0.0
                for line in f:
                    last_point = float(line.strip().split(',')[0])
            else:
                raise NotImplementedError(f"Filetype {filename.split('.')[-1]} is not implemented")

        print(sample_frequency)

        # Variable for the previous window and the current window
        prev_window: list[list[float]] = []
        current_window: list[list[float]] = []
        
        # Amount of samples per window
        samples_window = int(size * sample_frequency)
        # print(offset * sample_frequency)
        # Amount of samples in the offset
        samples_offset = m.ceil(offset * sample_frequency)
        
        # print(size, sample_frequency, samples_window)
        # print(offset, sample_frequency, samples_offset)

        # Open the file to write the features to
        with open('intervals.txt', 'w') as g:
            # Amount of samples to skip. Amount increases with offset / sample_frequency for every next window
            total_offset = 0 # TODO change offset for working file!!

            # As long as the total offset + the window size is smaller than the total duration of the recording,
            # the input file is read every time with the new offset. The lines of code that can be reused from the
            # previous time, are added to the new window and the new lines are also added.

            k = 0
            # while k < 4:
            # print(total_offset + samples_window, )
            while (total_offset + samples_window < int(last_point * sample_frequency)):
                with open(filename) as f:
                    # Skipping the header lines
                    if file_extension == 'txt' or file_extension == 'csv':
                        if file_extension == 'txt':
                            for _ in range(5 + total_offset): f.readline() # TODO change the range back to 4 + offset!
                        else:
                            for _ in range(1 + total_offset): f.readline()

                    # print(len(current_window))
                    if len(current_window) == 0:
                        for i in range(samples_window):
                            line = f.readline().strip().split(',')
                            current_window.append([])
                            for j in range(3):
                                current_window[i].append(float(line[j]))
                    else:
                        # try:
                            for i in range(samples_window):
                                if i < size * sample_frequency - samples_offset - 1:
                                    current_window[i] = prev_window[i + samples_offset]
                                    f.readline()
                                else:
                                    line = f.readline().strip().split(',')
                                    # print(line)
                                    for j in range(3):
                                        current_window[i][j] = float(line[j])
                        # except IndexError:
                        #     raise IndexError(f'Index: {i} from range {samples_window} '
                        #                      f'len: {len(current_window)}\n'
                        #                      f'Index: {i + samples_offset} from range {samples_window} '
                        #                      f'len: {len(prev_window)}')

                    if len(prev_window) > 0:
                        print('prev', prev_window[0], prev_window[-1], len(prev_window))
                    print('curr', current_window[0], current_window[-1], len(current_window))

                    if len(prev_window) == 0:
                        for i in range(samples_window):
                            prev_window.append(current_window[i])
                    else:
                        for i in range(samples_window):
                            prev_window[i] = current_window[i]
                k += 1
                total_offset += samples_offset

    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} at the relative path not found!")

        
def window(dataset, size, offset, sample_frequency):
    pass
