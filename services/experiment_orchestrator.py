from models.lda import MalletLDA
from models.bertopic import Bertopic
from models.ctm import CTM
from utils.file import load_json, add_column_to_jsonl, save_json
from sentence_transformers import SentenceTransformer
from services.metrics import MetricsService
from datetime import datetime

class ExperimentOrchestrator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None
        self.model = None

    def run(self):
        for experiment in self._get_config()["experiments"]:
            self._fit(experiment)
            self.save_topic_assignment(experiment)
            save_json(self._get_experiment_output(experiment), self.get_experiment_info(experiment))

    def get_experiment_info(self, experiment):
        return {
            "metadata": {
                "model": experiment["model"],
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": experiment["params"]
            },
            "metrics": self.get_metrics(experiment, self._get_result_output(experiment)),
            "topics": {str(id_t): words for id_t, words in self.model.get_topics()}
        }

    def save_topic_assignment(self, experiment):
        add_column_to_jsonl(experiment["dataset"], self._get_result_output(experiment), self.model.get_document_topics(), experiment["output_col"])

    def get_metrics(self, experiment, result_path):
        metrics_service = MetricsService(result_path, experiment["output_col"], experiment.get("ground_truth_col"))
        return {
            "harmonic_purity": metrics_service.get_harmonic_purity(),
            "adjusted_rand_index": metrics_service.get_adjusted_rand_index(),
            "normalized_mutual_information": metrics_service.get_normalized_mutual_information(),
            "npmi": metrics_service.get_npmi(self.model.get_topics(), experiment["tokens_col"]),
            "topic_diversity": metrics_service.get_topic_diversity(self.model.get_topics())
        }

    def _fit(self, experiment):
        if experiment["model"] == "LDA":
            self.model = MalletLDA(experiment["dataset"], experiment["tokens_col"])
            self.model.fit(experiment["params"]["n_topics"])
        elif experiment["model"] == "BERTopic":
            self.model = Bertopic(experiment["dataset"], experiment["raw_col"], experiment["tokens_col"])
            self.model.fit(experiment["params"]["n_topics"], SentenceTransformer(experiment["params"]["embedding_model"]))
        elif experiment["model"] == "CTM":
            self.model = CTM(experiment["dataset"], experiment["raw_col"], experiment["tokens_col"])
            self.model.fit(experiment["params"]["n_topics"], experiment["params"]["embedding_model"], experiment["params"]["context_size"])

    def _get_config(self):
        if self.config == None:
            self.config = load_json(self.config_file)
        return self.config
    
    def _get_result_output(self, experiment):
        return self._get_output_dir(experiment) + "/result.jsonl"

    def _get_experiment_output(self, experiment):
        return self._get_output_dir(experiment) + "/experiment-data.json"

    def _get_output_dir(self, experiment):
        return experiment["result_path"] + "/" + experiment["name"] + "/k-" + str(experiment["params"]["n_topics"])