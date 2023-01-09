from active_learning import ActiveLearning
import pandas as pd


def main():
    labels = ['stairs_up', 'stairs_down', 'walking', 'running']
    al = ActiveLearning(r'Preprocessed-data/Walking/features_Walking_scaled.csv', 'Walking', labels)
    al.training(150)
    print(f'number of errors: {al.testing(30)}')
    al.plotting()


if __name__ == '__main__':
    main()
