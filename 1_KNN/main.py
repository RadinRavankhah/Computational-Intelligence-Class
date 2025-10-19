import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

def read_data_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            values = line.strip().split(',')
            if len(values) == 5:
                data.append(values)
    df = pd.DataFrame(data, columns=["sepal_length","sepal_width","petal_length","petal_width","flower_label"])
    X = df.iloc[:, :-1].values[:-1].astype(float)
    y = df.iloc[:, -1].values[:-1]
    return X, y

def compare_feature_difference(feature_value_list_1, feature_value_list_2):
    if len(feature_value_list_1) != len(feature_value_list_2):
        raise Exception('The two feature value lists should be of the same length')
    
    sum = 0
    for i in range(len(feature_value_list_1)):
        temp_diff = (feature_value_list_1[i] - feature_value_list_2[i]) ** 2
        sum += temp_diff
    
    return math.sqrt(sum)


def knn_predict(X_train: np.ndarray, y_train, X_test, k: int):
    result_list = []
    for test_item in X_test:
        temp_distances_for_each_test_item = []
        for train_item_index in range(len(X_train)):
            temp_distance = compare_feature_difference(test_item, X_train[train_item_index])
            temp_distances_for_each_test_item.append([temp_distance, y_train[train_item_index]])

        sorted_data = sorted(temp_distances_for_each_test_item, key=lambda x: x[0])
        top_k = sorted_data[:k]
        
        tag_counts = {}
        for _, tag in top_k:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if not tag_counts:
            result = "N/A"
        
        max_count = max(tag_counts.values())
        most_common_tags = [tag for tag, count in tag_counts.items() if count == max_count]
        
        if len(most_common_tags) > 1:
            result = "N/A"
        else:
            result = most_common_tags[0]
        
        result_list.append(result)
    return result_list

def kfold_split(X, y, k=10):
    n = len(X)
    indices = np.arange(n)
    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))
        folds.append((train_idx, val_idx))
    return folds

def kfold_cross_validation(X, y, k_for_kfold: int, k_for_knn: int):
    folds = kfold_split(X, y, k_for_kfold)
    accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        y_pred = knn_predict(X_train, y_train, X_val, k_for_knn)

        correct_predictions_count = 0
        for i in range(len(y_pred)): # y_pred and y_val should be the same length
            if y_pred[i] == y_val[i]:
                correct_predictions_count += 1
        
        acc = correct_predictions_count/len(y_pred)
        accuracies.append(acc)
        # print(f"Fold {fold_idx + 1} accuracy: {acc:.3f}")

    # print(np.mean(accuracies))
    return np.mean(accuracies)

def main():
    X, y = read_data_file('1_KNN/iris.data')
    # print(X, y)

    X = MinMaxScaler().fit_transform(X)
    # print(X)

    k_list = [3, 5]
    k_accuracies = []
    for k in k_list:
        k_accuracy = kfold_cross_validation(X, y, 10, k)
        k_accuracies.append(k_accuracy)
        print(f"Accuracy for k={k}: {k_accuracy:.4f}")
    print(f"Best K value: {k_list[k_accuracies.index(max(k_accuracies))]}")


main()