from Data_Gathering_and_Preprocessing.data_processing import Preprocessing, empty_files
from Model_Active_Learning.active_learning import ActiveLearning


# empty_files(['Preprocessed-data/Walking2/features_Walking2.txt',
#              'Preprocessed-data/Walking2/features_Walking2_scaled.csv',
#              'Preprocessed-data/Walking2/processed_data_files.txt'])
#
# pre = Preprocessing('Walking2')
# pre.windowing(r"Data/data-lopen/Walking_part_1.csv", r"Data/data-lopen/Walking_part_1.mp4", # csvfile_test1.csv or csvfile_test_gyro_2
#               start_offset=2.5, stop_offset=5, size=1, offset=0.2, epsilon=0.01, do_plot=False, do_scale=False)
# pre.windowing(r"Data/data-lopen/Walking_part_2.csv", r"Data/data-lopen/Walking_part_2.mp4", # csvfile_test1.csv or csvfile_test_gyro_1
#               start_offset=2.5, stop_offset=5, size=1, offset=0.2, epsilon=0.01, do_plot=False, do_scale=True)

labels = ['stairs_up', 'stairs_down', 'walking', 'running']
al = ActiveLearning(r'Preprocessed-data/Walking/features_Walking_scaled.csv', 'Walking', labels)
al.training(150)
print(f'number of errors: {al.testing(30)}')
al.plotting()
