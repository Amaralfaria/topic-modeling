import pandas as pd
from sklearn import metrics
import numpy as np
import re


def metric_calc(data_file, ground_truth_col, output_col):
    """
    Calculate alignment metrics between predicted topics and ground-truth topics.

    Parameters:
    - data_file (str): Path to data file (containing both ground-truth and predicted topics)
    - ground_truth_col (str): Column name for ground-truth topics
    - output_col (str): Column name for predicted topics
    """
    # Load data
    data = pd.read_json(data_file, lines=True)
    output_topics = data[output_col]

    # Only retain the first topic in the list of topics
    output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    output_topics = [re.findall(output_pattern, topic)[0] for topic in output_topics]

    data["parsed_output"] = output_topics

    harmonic_purity, ari, mis = calculate_metrics(
        ground_truth_col, "parsed_output", data
    )

    print("--------------------")
    print("Alignment between predicted topics and ground truth:")
    print("Harmonic Purity: ", harmonic_purity)
    print("ARI: ", ari)
    print("MIS: ", mis)
    print("--------------------")

    return calculate_metrics(ground_truth_col, "parsed_output", data)

def calculate_metrics(true_col, pred_col, df):
    """
    Calculate topic alignment between df1 and df2 (harmonic purity, ARI, NMI)

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - harmonic_purity: Harmonic purity score
    - ari: Adjusted Rand Index
    - mis: Normalized Mutual Information
    """
    _, _, harmonic_purity = calculate_purity(true_col, pred_col, df)
    ari = metrics.adjusted_rand_score(df[true_col], df[pred_col])
    mis = metrics.normalized_mutual_info_score(df[true_col], df[pred_col])
    return (harmonic_purity, ari, mis)

def calculate_purity(true_col, pred_col, df):
    """
    Calculate harmonic purity between two set of clusterings

    Parameters:
    - true_col: Column containing a ground-truth label for each document
    - pred_col: Column containing a predicted label for each document (containing parsed topics)
    - df: Pandas data frame containing two columns (true_col and pred_col)

    Returns:
    - purity: Purity score
    - inverse_purity: Inverse purity score
    - harmonic_purity: Harmonic purity score
    """
    contingency_matrix = metrics.cluster.contingency_matrix(df[true_col], df[pred_col])
    precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
    recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    purity = (
        np.amax(precision, axis=0) * contingency_matrix.sum(axis=0)
    ).sum() / contingency_matrix.sum()
    inverse_purity = (
        np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    harmonic_purity = (
        np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    return (purity, inverse_purity, harmonic_purity)