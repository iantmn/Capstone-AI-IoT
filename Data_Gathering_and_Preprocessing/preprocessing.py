import json
import numpy as np
import pprint as p
import matplotlib.pyplot as plt

from data_processing import Preprocessing

from collections.abc import Collection, Iterable

def main() -> None:
    # mean, std, min_inter, max_inter = get_time_stats(r'Data files/Accelerometer Data X-axis.txt')
    # mean, std, min_inter, max_inter = Preprocessing.get_time_stats(r'data-lopen\Data Timo\Accelerometer Data 2022-11-29 11-11-24.txt')
    # print(mean, std, min_inter, max_inter, std / mean * 1000) # Mean, std and the promillage of the std to the mean
    # Preprocessing.plot_accelerometer(r'data-lopen\Data Timo\Timo_hardlopen_1.txt', 2, 2)
    # Preprocessing.plot_accelerometer(r'../Active_labeling_walking_running_stairs.txt')
    empty_files(['features_Walking.txt', 'features_Walking_scaled.csv', 'processed_data_files.txt'])
    pre = Preprocessing('Walking')
    # pre.windowing(r"data-lopen/Alan data/Stairs-20221201T092013Z-001/Stairs/Accelerometer_Data_1_stairs_upwards[1].txt", label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_op_1.txt", label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_op_2.txt", label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_op_3.txt", label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Alan data/Stairs-20221201T092013Z-001/Stairs/Accelerometer_Data_2_stairs_upwards[2].txt", label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Alan data/Stairs-20221201T092013Z-001/Stairs/Accelerometer_Data_3_stairs_upwards[1].txt", label='stairs_up',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_af_1.txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_af_2.txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_af_3.txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_trap_af_4.txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Alan data/Stairs-20221201T092013Z-001/Stairs/Accelerometer_Data_1_stairs_downwards[1].txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Alan data/Stairs-20221201T092013Z-001/Stairs/Accelerometer_Data_2_stairs_downwards[1].txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Alan data/Stairs-20221201T092013Z-001/Stairs/Accelerometer_Data_3_stairs_downwards[1].txt", label='stairs_down',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    
    # pre.windowing(r"data-lopen/Data Timo/Timo_hardlopen_1.txt", label='running',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_hardlopen_2.txt", label='running',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_hardlopen_3.txt", label='running',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_hardlopen_4.txt", label='running',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    # pre.windowing(r"data-lopen/Data Timo/Timo_hardlopen_5.txt", label='running',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = False)
    
    # pre.windowing(r"data-lopen/Alan data/Walking-20221201T092023Z-001/Walking/Accelerometer_Data_walking_28-11[2].txt", label='walking',
    #             start_offset=0.5, stop_offset=3, size=2, offset=0.2, epsilon=0.03, do_plot=False, do_scale = True)
    pre.windowing([r"./csvfile_test1.csv", r"./csvfile_test1_gyro.csv"], r"./ynotebook dingen/Walking_part_2.mp4", # csvfile_test1.csv or csvfile_test_gyro_2
              start_offset=0, stop_offset=0, size=1, offset=0.2, epsilon=0.05, do_plot=False, do_scale=False)
    pre.windowing([r"./csvfile_test2.csv", r"./csvfile_test2_gyro.csv"], r"./notebook dingen/Walking_part_1.mp4", # csvfile_test1.csv or csvfile_test_gyro_1
              start_offset=0, stop_offset=0, size=1, offset=0.2, epsilon=0.05, do_plot=False, do_scale=True)
    # pre.SuperStandardScaler()


def empty_files(files: Iterable[str]) -> None:
    for file in files:
        with open(file, 'w') as f:
            f.write('')
    
if __name__ == "__main__":
    main()
