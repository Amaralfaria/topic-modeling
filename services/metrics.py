import pandas as pd
from sklearn import metrics
import numpy as np
from utils.file import txt_to_octis_format
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from utils.file import get_data_from_column

class MetricsService:
    def __init__(self, result_file, output_column, ground_truth_column = None):
        self.result_file = result_file
        self.output_column = output_column
        self.ground_truth_column = ground_truth_column
        self.data = None

    def get_harmonic_purity(self):
        if not self.ground_truth_column:
            return 0

        _, _, hp = self._calculate_purity()
        return hp

    def get_adjusted_rand_index(self):
        if not self.ground_truth_column:
            return 0

        return metrics.adjusted_rand_score(self._get_data()[self.ground_truth_column], self._get_data()[self.output_column])

    def get_normalized_mutual_information(self):
        if not self.ground_truth_column:
            return 0

        return metrics.normalized_mutual_info_score(self._get_data()[self.ground_truth_column], self._get_data()[self.output_column])

    def get_npmi(self, topics, coluna_tokens):
        documentos = self._get_tokens(coluna_tokens)

        id2word = Dictionary(documentos)

        cm = CoherenceModel(
            topics=self._get_topics_data(topics), 
            texts=documentos, 
            dictionary=id2word, 
            coherence='c_npmi'
        )

        npmi_medio = cm.get_coherence()

        return npmi_medio

    def get_topic_diversity(self, topics, topk=10):
        diversity_metric = TopicDiversity(topk=topk)
        return diversity_metric.score(
            {
                "topics": self._get_topics_data(topics)
            }
        )

    def _calculate_purity(self):
        contingency_matrix = metrics.cluster.contingency_matrix(self._get_data()[self.ground_truth_column], self._get_data()[self.output_column])
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

    def _get_data(self):
        if self.data is None:
            self.data = pd.read_json(self.result_file, lines=True)
            self.data[self.ground_truth_column] = self.data[self.ground_truth_column].astype(str)

        return self.data

    def _get_tokens(self, tokens_column):
        return [doc.split() for doc in get_data_from_column(self.result_file, tokens_column)]

    def _get_topics_data(self, topics):
        return [words for id_t, words in topics if int(id_t) >= 0]