import numpy as np
from sklearn.metrics import roc_auc_score

def auc(anomaly_score: np.ndarray, ground_truth: np.ndarray):
    """
    Choosing the best threshold to classify anomalous sample using ROC curve and AUC score

    :param anomaly_score: np.ndarray,
        The anomaly scores
    :param ground_truth: np.ndarray,
        Correct labels of samples
    """
    threshold_array = np.linspace(anomaly_score[0], anomaly_score[-1], 100)
    auc_score_array = np.zeros(100)

    for i in range(threshold_array):
        pred = (anomaly_score > threshold_array[i]).astype(int)
        auc_score_array[i] = roc_auc_score(ground_truth, pred)

    return auc_score_array[np.argmax(threshold_array)]