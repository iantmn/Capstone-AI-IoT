import json
import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt

from collections.abc import Collection, Iterable


class Preprocessing():
    def __init__(self, action_ID: str = "") -> None:
        self.action_ID = action_ID
        self.output_file = f'features_{action_ID}.txt'

    def time(self, data: Collection, samples_window: float) -> tuple[float, float, float, float, float]:
        """Function to extract feature from the time-domain data

        Args:
            data (Collection): array of multiple sensors. Rows are data, columns are different sensors.

        Returns:
            tuple[float, float, float, float, float]: tuple of features for this specific windowed sample.
        """   
        # Making sure that the data collection is a numpy array     
        data = np.array(data)

        # List to save the extracted features in. It has as much items (lists) as sensors present
        features: list[list[float]] = [[] for i in range(data.shape[1])]
        for i in range(data.shape[1]):
            # Extract the features: minimum, maximum, average, standard deviation and area under the curve
            features[i].append(min(data[:, i]))
            features[i].append(max(data[:, i]))
            features[i].append(np.mean(data[:, i]))
            features[i].append(np.std(data[:, i]))
            features[i].append(sum(data[:, i]) / samples_window)

        return features

        
    def fourier(self, data: Collection, sampling_frequency: float, epsilon: float = 0.1, zero_padding: int = 0):
        """In the function: zeropadding, fft time data columns, spectral analysis

        Args:
            data (Collection): array of multiple sensors. Rows are data, columns are different sensors.
            sampling_frequency (float): floating point number with the sample frequency used to measure the data.
            epsilon (float): relative boundary for what values that are higher than the epsilon * maximum value is used
            to add to the total power.
            zero_padding: total amount of zero's are added to the data collection. This will increase the amount of
            frequencies in the fourier transform .

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
        peaks = Preprocessing.peak_value_frequency(ffts[:np.shape(data)[0]//2], sampling_frequency)
        pwrs = Preprocessing.cntrd_pwr(ffts[:np.shape(data)[0]//2], sampling_frequency, epsilon)
        
        return ffts, [peak + pwr for peak, pwr in zip(peaks, pwrs)]

    @staticmethod
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

    @staticmethod
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
        
    @staticmethod
    def get_sampling_frequency(input_file: str, start_offset: float = 0, stop_offset: float = 0, size: float = 2) -> tuple[float, float, float]:
        with open(input_file) as f:
            # Check the file extension
            file_extension = input_file.split('.')[-1]
            if file_extension == 'txt':
                # Skip the header lines
                for _ in range(5): f.readline() # TODO change the range back to 4!
            elif file_extension == 'csv':
                # Skip the header lines, and the first line
                for _ in range(2): f.readline()
            # If the file extension is not supported
            else:
                raise NotImplementedError(f"Filetype {input_file.split('.')[-1]} is not implemented")

            # Find the sample frequency by dividing 1 by the difference in time of two samples
            t0 = float(f.readline().strip().split(',')[0])
            t1 = float(f.readline().strip().split(',')[0])
            sampling_frequency = round(1 / (t1 - t0), 2)

            # Finding the last timestamp of the data
            last_point = 0.0
            for line in f:
                last_point = float(line.strip().split(',')[0])

            if size + start_offset + stop_offset > last_point - t0:
                size = last_point - t0 - start_offset - stop_offset

            # print(size, t0, last_point)
            return sampling_frequency, last_point, size

    def windowing(self, input_file: str, video_file: str = '', label: float = '',
            start_offset: float = 0, stop_offset: float = 0,
            size: float = 2, offset: float = 0.2, start: int = 1, stop: int = 3,
            epsilon: float = 0.1, do_plot: bool = False, do_scale = False) -> None:
        """
        Function for making windows of a certain size, with an offset. A window is made and the features of the window
        are axtracted. The the window is slided the offset amount of seconds to the right and new window is made and
        its features are extracted. This is done until the end of the file is reached. The extracted features are saved
        in a file with the name of the output_file attribute.

        Args:
            input_file (str): The relative path of the file with the data, seen from the main file.
            video_file (str, optional): The relative path of the file with the corresponding video the captures the process of
            capturing the data, seen from the main file. The path is only printed to an output file. Defaults to ''.
            label (str): The name of the label of the activity. Defaults to ''.
            size (float, optional): Size of the window in seconds. Defaults to 2.
            start_offset (float, optional): Skip the first r seconds of the data. Defaults to 0.
            stop_offset (float, optional): Skipt the last r seconds of the data. Defaults to 0.
            offset (float, optional): Size of the offset in seconds. Defaults to 0.2.
            start (int, optional): Start column from the data file. Defaults to 1.
            stop (int, optional): Last column (including) of the data file. Defaults to 3.
            epsilon (float, optional): Variable for the fourier function. Defaults to 0.1.
            do_plot (bool, optional): Set to true when a plot of every window is wanted. Defaults to False.

        Raises:
            NotImplementedError: This error is raised if the file extension is not supported.
            FileNotFoundError: This error is raised if the file-parameter cannot be found.
            ValueError: This error is raised when a value in the file cannot be converted to a float.
        """

        # ID of the datapoint, necessary for active learning
        current_ID = 0
        last_index = 0

        # Counter for keeping the timestamps comparable with the timestamps list.
        # This list is used when writing to the file to know when a window starts.
        timestamp_counter = 0
        timestamp_list: list[float] = []

        # check if the file that we want to extract the data from has already been used in this action_ID
        already_processed = False
        try:
            with open('processed_data_files.txt') as f:
                for line in f:
                    if line.strip() == input_file:
                        already_processed = True
                        break
        except FileNotFoundError:
            pass

        # This section lets you input y/n if you want to write the features to the file. Prevent adding the same data twice
        while True:
            # Check if the file was already processed, if it is, ask if the file should be processed again.
            if already_processed:
                print('The file is already processed at least once.\n')
            write_to_file = input(f"Do you want to save the extracted features '{label}' for '{self.action_ID}'? y/n\n")
            # Check if the input is valid
            if write_to_file == 'y':
                with open('processed_data_files.txt', 'a') as f:
                    f.write(f"{input_file}")
                break
            elif write_to_file == 'n':
                break
            else:
                print("Input not valid! Try again")

        # Check if the file that we will write to exist and if it does, if it contains a header already.
        # Add header if it is not yet in the file.
        # We also check what the last number is in the file. This is used for the datapoint ID in the write-file
        try:
            with open(self.output_file, 'r') as checkfile:
                if checkfile.readline().strip() == '':
                    make_header = True
                else:
                    make_header = False
                    for line in checkfile:
                        last_index = int(line.strip().split(',')[0]) + 1
        except FileNotFoundError:
            make_header = True
        
        # If we want to write and there is no header yet, build the header
        if write_to_file == 'y' and make_header:
            with open(self.output_file, 'a') as g:
                # Build list of possible labels
                sensor_names = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
                feature_names = ['min', 'max', 'avg', 'std', 'AUC', 'pk', 'cn', 'pw']

                full_header_features = []
                for i in range(len(sensor_names)):
                    full_header_features.append('')
                    for j in feature_names:
                        full_header_features[i] += f',{sensor_names[i]}_{j}'
                # full_header_features = [',acc_x_pk,acc_x_cn,acc_x_pw',
                #                         ',acc_y_pk,acc_y_cn,acc_y_pw',
                #                         ',acc_z_pk,acc_z_cn,acc_z_pw',
                #                         ',gyr_x_pk,gyr_x_cn,gyr_x_pw',
                #                         ',gyr_y_pk,gyr_y_cn,gyr_y_pw',
                #                         ',gyr_z_pk,gyr_z_cn,gyr_z_pw']

                # Build the header
                # print(full_header_features)
                specified_header = 'ID,label,time'
                for i in range(stop):
                    specified_header += full_header_features[i]
                
                # Write the header to file
                g.write(specified_header + '\n')

        try:
            # get sampling frequency and the last point
            sampling_frequency, last_point, size = self.get_sampling_frequency(input_file, start_offset, stop_offset, size)

            # Variable for the previous window and the current window
            prev_window: list[list[float]] = []
            current_window: list[list[float]] = []
            
            # Amount of samples per window
            samples_window = int(size * sampling_frequency) + 1 # This should be the same as m.ceil (now the math library is not needed anymore)
            # Amount of samples in the offset
            samples_offset = int(offset * sampling_frequency) + 1 # Same as previous statement
            
            # print(size, sampling_frequency, samples_window)
            # print(offset, sampling_frequency, samples_offset)

            # When the end of a datafile is reached, this value is set to False and the loop is exited
            not_finished = True
            
            # Opening the data file again and skipping the header lines.
            k = 0
            with open(input_file) as f:
                # print(start_offset, stop_offset, size)
                for _ in range(4 + int(start_offset * sampling_frequency)): f.readline()
                # Opening the output file; the extracted features will be put in the file
                with open(self.output_file, 'a') as g:
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
                                # check if the sample is the first of a window
                                if timestamp_counter % samples_offset == 0:
                                    timestamp_list.append(line[0])
                                # We read a line, keep count
                                timestamp_counter += 1
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

                                    # check if the sample is the first of a window
                                    if timestamp_counter % samples_offset == 0:
                                        timestamp_list.append(line[0])
                                    # We read a line, keep count
                                    timestamp_counter += 1
                            except EOFError:
                                not_finished = False
                        # if len(prev_window) > 0:
                        #     print('prev', prev_window[0], prev_window[len(prev_window)//2], prev_window[-1], len(prev_window))
                        # print('curr', current_window[0], current_window[len(current_window)//2], current_window[-1], len(current_window))
                        
                        if not_finished:
                            # choose the length of the zero padding of the window. Increased definition
                            padding = int(len(current_window) * 6 / 5)
                            # get the features from the window and the fourier signal for plotting
                            features_time = self.time(current_window, samples_window)
                            ffts, features_fourier = self.fourier(current_window, sampling_frequency, zero_padding=padding-len(current_window), epsilon=epsilon)
                            
                            # Make a list with all the features in the right order per sensor. Right order is first time features and then frequency
                            features: list[list[float]] = []
                            for i in range(len(features_time)):
                                features.append(features_time[i] + list(features_fourier[i])[1:])
                                
                            # build a string of the feature data. The first element of the string is the timestamp, pop this timestamp
                            features_list = [timestamp_list.pop(0)]
                            for tup in features:
                                for i, data in enumerate(tup):
                                    features_list.append(str(data))
                            # print(features_list)
                            # Add the features to the file if write_to_file is 'y'
                            if write_to_file == 'y':
                                g.write(str(current_ID + last_index) + ',' + label + ',' + ','.join(features_list) + '\n')

                            # If we want to plot
                            if do_plot:
                                # mirrored fft for better readability
                                ffts_shifted = np.fft.fftshift(ffts)
                                # frequency axis for non-mirrored fft, account for zero_padding
                                frequency = np.arange(0, sampling_frequency/2 - 1/padding, sampling_frequency/padding)
                                # frequency axis for mirrored fft, account for zero_padding
                                frequency_shift = np.arange(-sampling_frequency/2 + sampling_frequency/(2*padding), #start
                                                    sampling_frequency/2 + sampling_frequency/(2*padding), #stop
                                                    sampling_frequency/padding) #interval
                                # Time axis for the plots
                                time = np.arange(start_offset + offset * k, start_offset + offset * k + len(current_window)//sampling_frequency,
                                                1 / sampling_frequency)

                                # print(time)

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
                            current_ID += 1
                            
                    # Writing the first and the last index and the relative path to the video to the output
                    # file with the files that are used.
                    with open(r'processed_data_files.txt', 'a') as h:
                        h.write(f",{last_index},{last_index + current_ID - 1},{video_file}\n")
            # print(k)
            plt.show()

            # If do_scale is set to True make a scaled version of the file as well
            if do_scale is True:
                self.SuperStandardScaler(self.output_file)
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"File {input_file} at the relative path not found!")
        except ValueError:
            raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")
        
    @staticmethod
    def plot_accelerometer(input_file: str, start_offset: float = 0, stop_offset: float = 0, start: int = 1, stop: int = 3) -> None:
        """Function to plot the time-domain curves of data from an input-file

        Args:
            input_file (str): The relative path of the file with the data, seen from the main file.
            start_offset (float, optional): Skip the first r seconds of the data. Defaults to 0.
            stop_offset (float, optional): Skipt the last r seconds of the data. Defaults to 0.
            start (int, optional): Start column from the data file. Defaults to 1.
            stop (int, optional): Last column (including) of the data file. Defaults to 3.

        Raises:
            FileNotFoundError: Error raised if the given input-file cannot be found.
            ValueError: Error raised if a data point cannot be converted to a float.
        """        
        try:
            # Calculate the sampling frequency and the last timestamp (the third parameter, size, is not used)
            sampling_frequency, last_point = Preprocessing.get_sampling_frequency(input_file, start_offset, stop_offset)[0:2]
                
            # Try opening the input-file
            with open(input_file) as f:
                # List with the datapoints, each row will have the data of one sensor
                data: list[list[float]] = []
                # Skip the first 4 lines (contains no data) + the amount of samples in the start_offset
                for _ in range(4 + int(start_offset * sampling_frequency)): f.readline()

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
                
        except FileNotFoundError:
            raise FileNotFoundError(f"File {input_file} at the relative path not found!")
        except ValueError:
            raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")
        
        # Define the time axis
        time = np.arange(0, len(data) / sampling_frequency, 1 / sampling_frequency)
        
        # Plot
        fig, axes = plt.subplots(1, 1)
        axes.plot(time, data)
        plt.show()


    @staticmethod
    def get_time_stats(input_file: str) -> tuple[float, float, float, float]:
        """Function to get some stats about the sampling period (average period, standard deviation, minimal and maximal period).

        Args:
            input_file (str): The relative path of the file with the data, seen from the main file.

        Raises:
            FileNotFoundError: Error raised if the given input-file cannot be found.
            ValueError: Error raised if a data point cannot be converted to a float.

        Returns:
            tuple[float, float, float, float]: The results: average period, standard deviation, minimal period and maximal period
        """        
        try:
            # Try opening the file
            with open(input_file) as f:
                # Read the first lines (does not contain data)
                for _ in range(4): f.readline()

                # Array with the periods
                intervals: np.ndarray[float] = np.array([])
                # First time sample
                t0 = float(f.readline().strip().split(',')[0])
                for line in f:
                    # Checking if the line is not empty (last line is empty)
                    if line != '':
                        # Second time sample
                        t1 = float(line.strip().split(',')[0])
                        # Saving the difference
                        intervals = np.append(intervals, t1 - t0)
                        # Setting the second time sample as the first
                        t0 = t1

            # Calculating the features
            mean = np.mean(intervals)
            std = np.std(intervals)
            min_inter = min(intervals)
            max_inter = max(intervals)
            return mean, std, min_inter, max_inter

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{input_file}' at the relative path not found!")
        except ValueError:
            raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")


    def SuperStandardScaler(self, input_file: str) -> None:
        """Scale the features with their respecitive scale. All centroid, peak and total power are put on the same scale.
        By setting do_scale in windowing to True this function is called automatically, else build a object of Preprocessing in main
        and execute Preprocessing.SuperStandardScaler(path)
        TODO: implement possibility to also use gyroscope data

        Args:
            input_file (str): input file where the unscaled features are stored
        """        
        # First, manually build the data_array, this is because importing with a header and starting columns is a pain
        with open(input_file) as f:
            # Split the set in header, starting columns and actual data
            header = np.array([f.readline().strip().split(',')], dtype='unicode')
            columns = np.array([[0]*3], dtype='unicode')
            data_array = np.array([[0]*(header.shape[1]-3)], dtype=float)
            # Fill the data_array
            while True:
                line = f.readline().strip().split(',')
                # Check if the last line was reached
                if line[0] == '':
                    break
                else:
                    # Split the line and add to the two np_arrays
                    columns = np.append(columns, np.array([line[:3]], dtype='unicode'), axis=0)
                    data_array = np.append(data_array, np.array([line[3:]], dtype=float), axis=0)

        # Yeah append only works when the first element has the same size so i filled one with zeros, sue me
        data_array = data_array[1:]
        columns = columns[1:]

        fa = 0
        if 'acc_x_min' in header:
            fa += 5
        if 'acc_x_pk' in header:
            fa += 3

        # Get amount of features and datapoints
        sensors_amount = (data_array.shape[1])//fa
        datapoints_amount = data_array.shape[0] - 1

        print(f'fa: {fa}, sensors_amount: {sensors_amount}')

        if sensors_amount > 6:
            raise ValueError('You have used more than 6 sensors, we have not yet implemented ')
        # If there are more then 3 sensors used, use two different sets.
        # Case < 4 sensors used, max 9 features. Sum 
        for i in range(fa):
            sum_feature = 0
            # Go trough every column with the same feature of different sensors and add them
            for j in range(0, min(3, sensors_amount)*fa, fa):
                sum_feature += sum(data_array[:, i+j])
            # Devide to get the mean
            sum_feature = sum_feature / (min(3, sensors_amount)*datapoints_amount)

            # Determine standard deviation of the feature
            std_feature = 0
            for j in range(0, min(3, sensors_amount)*fa, fa):
                for k in range(0, datapoints_amount):
                    std_feature += (data_array[k, i+j] - sum_feature)**2
            # Devide by n-1 and take root
            std_feature = (std_feature/(min(3, sensors_amount)*datapoints_amount-1))**0.5
            # Rescale the columns with their respective feature mean and std
            for j in range(0, min(3, sensors_amount)*fa, fa):
                data_array[:, i+j] = (data_array[:, i+j] - sum_feature)/std_feature
        # if there are more then 3 sensors used, we have gyroscope sensors as well.
        if sensors_amount > 3:
            for i in range(0, fa-3):
                sum_feature = 0
                # Go trough every column with the same feature of different sensors and add them
                for j in range(3*fa, sensors_amount*fa, fa):
                    sum_feature += sum(data_array[:, i+j])
                # Devide to get the mean
                sum_feature = sum_feature / ((sensors_amount-3)*datapoints_amount)

                # Determine standard deviation of the feature
                std_feature = 0
                for j in range(3*fa, sensors_amount*fa, fa):
                    for k in range(0, datapoints_amount):
                        std_feature += (data_array[k, i+j] - sum_feature)**2
                # Devide by n-1 and take root
                std_feature = (std_feature/((sensors_amount-3)*datapoints_amount-1))**0.5
                # Rescale the columns with their respective feature mean and std
                for j in range(3*fa, sensors_amount*fa, fa):
                    data_array[:, i+j] = (data_array[:, i+j] - sum_feature)/std_feature

        # Merge the shitshow
        data_array = data_array.astype('unicode')
        data_array = np.append(columns, data_array, axis=1)
        data_array = np.append(header, data_array, axis=0)

        # Save as csv file
        np.savetxt(f'features_{self.action_ID}_scaled.csv', data_array, fmt='%s', delimiter=',')


def empty_files(files: Iterable[str]) -> None:
    for file in files:
        with open(file, 'w') as f:
            f.write('')
