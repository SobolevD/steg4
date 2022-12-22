import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.svm
import sklearn.tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from utils.consts import M, SIGMA
from utils.in_out import read_image
from utils.nzb import svi_1_encode, get_plane
from utils.watermark import generate_watermark


def train_and_test(X, y, clf):
    # Отделяем 20% выборки для обучения и 80% для оценки
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, train_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return sklearn.metrics.f1_score(y_pred, y_test)


def get_series_array(bit_plate, max_i):

    bit_plate_flatten = bit_plate.copy().flatten()

    result = np.zeros(max_i)

    current_bit = 0
    series_length = 0
    while current_bit < bit_plate_flatten.size:
        for series_bit in range (current_bit, bit_plate_flatten.size):
            if bit_plate_flatten[current_bit] == bit_plate_flatten[series_bit] and series_length < max_i:
                series_length += 1
            else:
                result[series_length - 1] += 1
                current_bit += series_length
                series_length = 0
                break
            if series_bit == bit_plate_flatten.size - 1:
                current_bit = bit_plate_flatten.size

    return result


def get_feature_vector(series):
    tmp = np.array(np.split(series, 5)) # Чтоб было по 4 штуки
    result = np.zeros(5)
    for i in range(tmp.shape[0]):
        result[i] = np.sum(np.array(tmp[i]))
    return result


if __name__ == '__main__':
    images_file_names = os.listdir('resources/images')

    dt = []
    rf = []
    mlp = []
    ab = []

    p = 2
    K = np.array(images_file_names).size
    y = np.zeros(K)

    q_array = [1, 0.7, 0.5, 0.3, 0.1]
    for q in q_array:

        train_dataset = []
        print(f'q={q}')

        image_num = 0

        for image_file_name in images_file_names:

            if image_num % 10 == 0:
                print(f'Processing {image_num*2}%')

            full_image_path = 'resources/images/' + image_file_name
            image = read_image(full_image_path)

            image_to_process = ''
            if image_num < K / 2:
                watermark = generate_watermark(image.size, M, SIGMA).reshape(image.shape[0], image.shape[1])
                image_to_process = svi_1_encode(image, watermark, p, q)
            else:
                image_to_process = image
                y[image_num] = 1

            bit_plate = get_plane(image_to_process, p)
            series_array = get_series_array(bit_plate, 20)  # / image.size
            feature_vector = get_feature_vector(series_array)
            train_dataset.append(feature_vector)

            image_num += 1

        decision_tree_result = train_and_test(train_dataset, y, sklearn.tree.DecisionTreeClassifier(max_depth=5))
        random_forest_result = train_and_test(train_dataset, y, sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=1000))
        mlp_result = train_and_test(train_dataset, y, MLPClassifier(alpha=1, max_iter=1000))
        ada_boost_result = train_and_test(train_dataset, y, AdaBoostClassifier(n_estimators=50))
        print("====================================================")
        print(f'Decision tree classifier accuracy: {decision_tree_result}')
        print(f'Random forest classifier accuracy: {random_forest_result}')
        print(f'MLP classifier accuracy: {mlp_result}')
        print(f'AdaBoost classifier accuracy: {ada_boost_result}')
        print("====================================================")
        dt.append(decision_tree_result)
        rf.append(random_forest_result)
        mlp.append(mlp_result)
        ab.append(ada_boost_result)

    plt.plot(dt, q_array)
    plt.plot(rf, q_array)
    plt.plot(mlp, q_array)
    plt.plot(ab, q_array)
    plt.show()

    plt.plot(q_array, dt, label='Decision Tree Classifier')
    plt.plot(q_array, rf, label='Random Forest Classifier')
    plt.plot(q_array, mlp, label='MLP Classifier')
    plt.plot(q_array, ab, label='AdaBoost Classifier')
    plt.xlabel('q')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



