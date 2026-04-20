import pandas as pd
from sklearn import metrics
import numpy as np
import re
import json
from utils.file import txt_to_octis_format
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity


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
    # output_topics = data[output_col]

    # Only retain the first topic in the list of topics
    # output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    # output_topics = [re.findall(output_pattern, topic)[0] for topic in output_topics]

    # data["parsed_output"] = output_topics
    data["parsed_output"] = data[output_col].astype(str)

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


def get_npmi(caminho_jsonl, topics_path, coluna_tokens):
    documentos = []
    with open(caminho_jsonl, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            documentos.append(dados[coluna_tokens].split())

    id2word = Dictionary(documentos)

    dados_topicos = txt_to_octis_format(topics_path)["topics"] 

    cm = CoherenceModel(
        topics=dados_topicos, 
        texts=documentos, 
        dictionary=id2word, 
        coherence='c_npmi'
    )

    npmi_medio = cm.get_coherence()

    return npmi_medio

def get_topic_diversity(topics_path, topk=10):
    """
    Calcula a diversidade de tópicos (Topic Diversity).
    Não exige o corpus original, apenas o arquivo de tópicos.
    """
    
    # 1. Carrega os tópicos do arquivo TXT usando a função que criamos anteriormente
    # Certifique-se de que a função txt_to_octis_format esteja definida no seu código
    dados_topicos = txt_to_octis_format(topics_path) 
    
    # 2. Instancia a métrica de Diversidade
    # 'topk' define quantas palavras do topo de cada tópico serão comparadas
    diversity_metric = TopicDiversity(topk=topk)

    # 3. Calcule o score
    # O OCTIS espera o dicionário {"topics": [[...], [...]]}
    score = diversity_metric.score(dados_topicos)

    return score