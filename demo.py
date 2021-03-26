# Produce a heatmap parameter scan on the model predictions
# Note: server must be running
import requests
import numpy as np
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt

def generate_heatmap():

    # Generate data ranges for heatmap
    min_sleep = 0
    max_sleep = 8
    min_study = 0
    max_study = 8
    n_sleep = 50
    n_study = 50

    sleep_range = np.arange(min_sleep, max_sleep, max_sleep/n_sleep)
    study_range = np.arange(min_study, max_study, max_study/n_study)

    p = [*product(sleep_range, study_range)]

    # Package inputs and generate predictions
    inputs = {'sleep': [val[0] for val in p], 'study': [val[1] for val in p]}
    preds = requests.post('http://localhost:5000/predict/linear', json=inputs).json()

    # Reshape for heatmap
    mat = np.array(preds).reshape(n_sleep, n_study)

    # Plot heatmap
    ax = sns.heatmap(mat)
    plt.xlabel('Study', fontweight='bold', fontsize=14)
    plt.ylabel('Sleep', fontweight='bold', fontsize=14)
    plt.title('Exam scores', fontweight='bold', fontsize=16)

    plt.xticks([*range(n_study)][::5], study_range.round(2)[::5])
    plt.yticks([*range(n_sleep)][::5], sleep_range.round(2)[::5])

    ax.invert_yaxis()
    plt.savefig('heatmap_demo.png', dpi=200)

if __name__ == '__main__':
    generate_heatmap()
