from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from utils.file import get_data_from_column

class CTM:
    def __init__(self, caminho_entrada, coluna_texto, coluna_tokens):
        self.caminho_entrada = caminho_entrada
        self.coluna_texto = coluna_texto
        self.coluna_tokens = coluna_tokens
        self.model = None
        self.training_dataset = None

    def fit(self, num_topicos, embedding_model, context_size):
        qt = TopicModelDataPreparation(embedding_model)
        self.training_dataset = qt.fit(text_for_contextual=self._get_text(), text_for_bow=self._get_tokens())
        self.model = CombinedTM(bow_size=len(qt.vocab), contextual_size=context_size, n_components=num_topicos, num_epochs=30)
        self.model.fit(self.training_dataset)

        return self.model

    def get_document_topics(self):
        topic_predictions = self.model.get_thetas(self.training_dataset)
        return [prediction.argmax() for prediction in topic_predictions]

    def get_topics(self):
        topic_lists = self.model.get_topic_lists(10)
        return [(i, words) for i, words in enumerate(topic_lists)]

    def _get_tokens(self):
        return list(get_data_from_column(self.caminho_entrada, self.coluna_tokens))

    def _get_text(self):
        return list(get_data_from_column(self.caminho_entrada, self.coluna_texto))