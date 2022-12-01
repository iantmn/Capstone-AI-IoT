import json
import numpy as np
import math as m
import matplotlib.pyplot as plt
from collections.abc import Collection


def fourier(data: Collection, sample_frequency: float, epsilon: float = 0.1, zero_padding: int = 0) -> tuple[np.ndarray[float], list[float]]:
    """In the function: zeropadding, fft time data columns, spectral analysis

    Args:
        data (Collection): numpy array of multiple sensors. Rows are data, columns are different sensors
        sample_frequency (float): floating point number with the sample frequency used to measure the data
        epsilon (float): relative boundary for what values that are higher than the epsilon * maximum value is used
        to add to the total power
        zero_padding: total amount of zero's are added to the data collection. This will increase the amount of
        frequencies in the fourier transform 

    Returns:
        tuple[np.array[float]]: tuple of features for this specific windowed sample.
    """
    # Making sure that the data collection is a numpy array
    data = np.array(data)
    # Adding the zero's if this should be done 
    if zero_padding > 0:
        data = np.append(data, np.zeros([zero_padding, 3]), axis=0)
        
    # Calculating the fft's
    ffts = np.fft.fft2(data, axes=(-2, -2, -2))
    
    # Extracting the features
    peaks = peak_value_frequency(ffts[:np.shape(data)[0]//2], sample_frequency)
    pwrs = cntrd_pwr(ffts[:np.shape(data)[0]//2], sample_frequency, epsilon)
    
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
    
    # Making sure that the dataset is a numpy array
    data = np.array(dataset)
    
    sensor_range = data.shape[1]
    total_power = [0.]*sensor_range  # 3 sensors total power
    centroid = [0.]*sensor_range    # 3 sensors centroid, not returned
    index = [0]*sensor_range  # index of the centroid
    # compute for all three sensors
    for i in range(sensor_range):
        maxm = max(data[1:, i])
        length = len(data[:, i])
        # print(f'maximum: {maxm}, length {length}')
        # Sum power and sum all values above the threshold
        for j in range(1, length):
            total_power[i] += abs(data[j][i])
            if abs(data[j][i]) > epsilon*maxm:
                centroid[i] += abs(data[j][i])
        goal = centroid[i]/2
        centroid[i] = 0.
        # reset j, go through the dataset again and stop when you surpass centroid/2
        j = 1
        while centroid[i] < goal:
            if abs(data[j][i]) > epsilon*maxm:
                centroid[i] += abs(data[j][i])
            j += 1
            index[i] += 1
            
    return [(e*sampling_frequency/(2*len(data)), pwr) for e, pwr in zip(index, total_power)]


def peak_value_frequency(dataset: Collection, sampling_frequency: float) -> list[tuple[int, float]]:
    """find the frequency that is the most present in a PSD. Do not include DC component

    Args:
        dataset (Collection): frequency domain dataset
        sampling_frequncy (float): the sampling frequency used to measure the data

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
        index = np.where(abs(data[1:mid, i]) == max_value)[0][0] + 1
        # print(loc, abs(data[loc, i] / (interval * data.shape[0])), abs(data[loc, i] / (interval * data.shape[0])))
        peaks.append((index, index * sampling_frequency / (2 * data.shape[0])))
    return peaks

def windowing(filename: str, action_ID: float, label: float,
        start_offset: float = 0, stop_offset: float = 0,
        size: float = 2, offset: float = 0.2, start: int = 1, stop: int = 3,
        epsilon: float = 0.1, do_plot: bool = False) -> str:
    """
    Function for making windows of a certain size, with an offset. A window is made and the features of the window
    are axtracted. The the window is slided the offset amount of seconds to the right and new window is made and
    its features are extracted. This is done until the end of the file is reached. The extracted features are saved
    in a file

    Args:
        filename (str): The relative path of the file with the data, seen from the main file.
        action_ID (str): The name of the recorder activity.
        label (str): The name of the label of the activity.
        size (float, optional): Size of the window in seconds. Defaults to 2.
        start_offset (float, optional): Skip the first r seconds of the data. Defaults to 0.
        stop_offset (float, optional): Skipt the last r seconds of the data. Defaults to 0.
        offset (float, optional): Size of the offset in seconds. Defaults to 0.2.
        start (int, optional): Start column from the data file. Defaults to 1.
        stop (int, optional): Last column (including) of the data file. Defaults to 3.
        epsilon (float, optional): Variable for the fourier function. Defaults to 0.1.
        do_plot (bool, optional): Set to true when a plot of every window is wanted. Defaults to False.

    Raises:
        NotImplementedError: This error is raised if the file extension is not supported
        FileNotFoundError: This error is raised if the file-parameter cannot be found
        ValueError: This error is raised when a value in the file cannot be converted to a float

    Returns:
        str: The relative path of the output file. This file contains the extracted features.
    """

    output_file = f'features_{action_ID}.txt'

    # This section lets you input y/n if you want to write the features to the file. Prevent adding the same data twice
    done = False
    while not done:
        write_to_file = input("Do you want to save the extracted features? y/n\n")
        # Check if the input is valid
        if write_to_file == 'y' or write_to_file == 'n':
            done = True
        else:
            print("Input not valid! Try again")

    try:
        with open(filename) as f:
            # Check the file extension
            file_extension = filename.split('.')[-1]
            if file_extension == 'txt':
                # Skip the header lines
                for _ in range(5): f.readline() # TODO change the range back to 4!
            elif file_extension == 'csv':
                # Skip the header lines, and the first line
                for _ in range(2): f.readline()
            # If the file extension is not supported
            else:
                raise NotImplementedError(f"Filetype {filename.split('.')[-1]} is not implemented")

            # Find the sample frequency by dividing 1 by the difference in time of two samples
            t0 = float(f.readline().strip().split(',')[0])
            t1 = float(f.readline().strip().split(',')[0])
            sample_frequency = round(1 / (t1 - t0), 2)

            # Finding the last timestamp of the data
            last_point = 0.0
            for line in f:
                last_point = float(line.strip().split(',')[0])

            if size + start_offset + stop_offset > last_point - t0:
                size = last_point - t0 - start_offset - stop_offset

            print(size, t0, last_point)

        # Variable for the previous window and the current window
        prev_window: list[list[float]] = []
        current_window: list[list[float]] = []
        
        # Amount of samples per window
        samples_window = int(size * sample_frequency) + 1 # This should be the same as m.ceil (now the math library is not needed anymore)
        # Amount of samples in the offset
        samples_offset = int(offset * sample_frequency) + 1 # Same as previous statement
        
        # print(size, sample_frequency, samples_window)
        # print(offset, sample_frequency, samples_offset)

        # When the end of a datafile is reached, this value is set to False and the loop is exited
        not_finished = True
        
        # Opening the data file again and skipping the header lines.
        k = 0
        with open(filename) as f:
            for _ in range(4 + int(start_offset * sample_frequency)): f.readline()
            # Opening the output file; the extracted features will be put in the file
            with open(output_file, 'a') as g:
                # While the end of the file is not yet reached
                while not_finished:
                # while k < 1:
                    # If there is no window made yet
                    if len(current_window) == 0:
                        for _ in range(samples_window):
                            # Store a list of the sensordata of the line that is read
                            line = f.readline().strip().split(',')
                            # Initialise the current window, make sure the added value is an empty list
                            current_window.append([])
                            for i in range(start, stop + 1):
                                current_window[-1].append(float(line[i]))
                    else:
                        # If this is the first time that previous window is called, initialise it as a copy of current window
                        # Use .copy() to prevent aliasing
                        if len(prev_window) == 0:
                            for i in range(samples_window):
                                prev_window.append(current_window[i].copy())
                        # Else we make it a direct copy of the current window as well
                        else:
                            for i in range(samples_window):
                                prev_window[i] = current_window[i].copy()
                        
                        # Overwrite the current window values with it's previous values
                        # The current window is slided the samples_offset amount of samples into the future. Thus
                        # The samples [samples_offset:] of prev_window are the first samples for the current window.
                        # The rest of the samples are new and read from the data file
                        for i in range(samples_window - samples_offset):
                            current_window[i] = prev_window[i + samples_offset].copy()
                            
                        # Read new lines from the file and add these to the end of the current file
                        try:
                            for i in range(samples_offset):
                                line = f.readline().strip().split(',')
                                # The last line of the file is an empty string. When detected we exit the while loop
                                if line[0] == '':
                                    not_finished = False
                                    break
                                elif float(line[0]) > last_point - stop_offset:
                                    not_finished = False
                                    break
                                # Read samples_offset amount of samples and add these to the current window
                                for j in range(start, stop + 1):
                                    # print(i, i + samples_window - samples_offset, j, line, len(current_window), len(current_window[0]))
                                    current_window[i + samples_window - samples_offset][j - start] = float(line[j])
                        except EOFError:
                            not_finished = False
                    # if len(prev_window) > 0:
                    #     print('prev', prev_window[0], prev_window[len(prev_window)//2], prev_window[-1], len(prev_window))
                    # print('curr', current_window[0], current_window[len(current_window)//2], current_window[-1], len(current_window))
                    
                    if not_finished:
                        # choose the length of the zero padding of the window. Increased definition
                        padding = int(len(current_window) * 6 / 5)
                        # get the features from the window and the fourier signal for plotting
                        ffts, features = fourier(current_window, sample_frequency, zero_padding=padding-len(current_window), epsilon=epsilon)
                        
                        # print(features)
                            
                        # build a string of the feature data. The first element of the string is the label of the action
                        features_list = [label]
                        for tup in features:
                            for i, data in enumerate(tup):
                                # We don't take the first value since it is an index representation 
                                if i > 0:
                                    features_list.append(str(data))
                        # print(features_list)
                        # Add the features to the file if write_to_file is 'y'
                        if write_to_file == 'y':
                            g.write(','.join(features_list) + '\n')

                        # Time axis for the plots
                        time = np.arange(start_offset + offset * k, start_offset + offset * k + len(current_window) / sample_frequency,
                                        1 / sample_frequency)

                        # If we want to plot
                        if do_plot:
                            # mirrored fft for better readability
                            ffts_shifted = np.fft.fftshift(ffts)
                            # frequency axis for non-mirrored fft, account for zero_padding
                            frequency = np.arange(0, sample_frequency/2 - 1/padding, sample_frequency/padding)
                            # frequency axis for mirrored fft, account for zero_padding
                            frequency_shift = np.arange(-sample_frequency/2 + sample_frequency/(2*padding), #start
                                                sample_frequency/2 + sample_frequency/(2*padding), #stop
                                                sample_frequency/padding) #interval

                            # Plotting specs
                            fig, axes = plt.subplots(2, 1)
                            axes[0].plot(time, current_window, label=['X', 'Y', 'Z'])
                            axes[0].legend(loc='upper left')
                            axes[0].set_title("Accelerometer data")
                            # axes[1].plot(frequency, abs(ffts), label=['X', 'Y', 'Z'])
                            axes[1].plot(frequency, abs(ffts[0:round(len(ffts)/2)]), label=['X', 'Y', 'Z'])
                            axes[1].legend(loc='upper left')
                            axes[1].set_title("Fourier Transform")

                        k += 1
        # print(k)
        plt.show()
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} at the relative path not found!")
    except ValueError:
        raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")
    
    return output_file
    
def plot_accelerometer(filename: str, start_offset: float = 0, stop_offset: float = 0, start: int = 1, stop: int = 3) -> None:
    try:
        with open(filename) as f:
            # Check the file extension
            file_extension = filename.split('.')[-1]
            if file_extension == 'txt':
                # Skip the header lines
                for _ in range(5): f.readline() # TODO change the range back to 4!

                # Find the sample frequency by dividing 1 by the difference in time of two samples
                t0 = float(f.readline().strip().split(',')[0])
                t1 = float(f.readline().strip().split(',')[0])
                sample_frequency = round(1 / (t1 - t0), 2)

                last_point = 0.0
                for line in f:
                    last_point = float(line.strip().split(',')[0])
            # If the file extension is not supported
            else:
                raise NotImplementedError(f"Filetype .{filename.split('.')[-1]} is not implemented")
            
        with open(filename) as f:
            data: list[list[float]] = []
            for _ in range(5 + int(start_offset * sample_frequency)): f.readline()

            not_finished = True
            while not_finished:
                line = f.readline().strip().split(',')
                # The last line of the file is an empty string. When detected we exit the while loop
                if line[0] == '':
                    not_finished = False
                    break
                elif float(line[0]) > last_point - stop_offset:
                    not_finished = False
                    break
                # Read samples_offset amount of samples and add these to the current window
                data.append([])
                for j in range(start, stop + 1):
                    # print(i, i + samples_window - samples_offset, j, line, len(current_window), len(current_window[0]))
                    data[-1].append(float(line[j]))


        # with open(filename) as f:
        #     data2: list[list[float]] = []
        #     for _ in range(5): f.readline()

        #     not_finished = True
        #     while not_finished:
        #         line = f.readline().strip().split(',')
        #         # The last line of the file is an empty string. When detected we exit the while loop
        #         if line[0] == '':
        #             not_finished = False
        #             break
        #         # Read samples_offset amount of samples and add these to the current window
        #         data2.append([])
        #         for j in range(start, stop + 1):
        #             # print(i, i + samples_window - samples_offset, j, line, len(current_window), len(current_window[0]))
        #             data2[-1].append(float(line[j]))
            
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} at the relative path not found!")
    except ValueError:
        raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")
    
    time = np.arange(0, len(data) / sample_frequency, 1 / sample_frequency)
    # time2 = np.arange(0, len(data2) / sample_frequency, 1 / sample_frequency)
    
    fig, axes = plt.subplots(1, 1)
    axes.plot(time, data)
    # axes[0].plot(time2, data2)
    plt.show()

def get_time_data(filename: str) -> tuple[float, float, float, float]:
    try:
        with open(filename) as f:
            for _ in range(4): f.readline()

            intervals: np.ndarray[float] = np.array([])
            t0 = float(f.readline().strip().split(',')[0])
            for line in f:
                if line != '':
                    t1 = float(line.strip().split(',')[0])
                    intervals = np.append(intervals, t1 - t0)
                    t0 = t1

        # print(intervals)
        mean = np.mean(intervals)
        std = np.std(intervals)
        min_inter = min(intervals)
        max_inter = max(intervals)
        return mean, std, min_inter, max_inter

    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} at the relative path not found!")
    except ValueError:
        raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")
