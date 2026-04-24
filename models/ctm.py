from contextualized_topic_models.models.ctm import CombinedTM
from utils.file import get_data_from_column
from contextualized_topic_models.datasets.dataset import CTMDataset
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

class CTM:
    def __init__(self, caminho_entrada, coluna_texto, coluna_tokens):
        self.caminho_entrada = caminho_entrada
        self.coluna_texto = coluna_texto
        self.coluna_tokens = coluna_tokens
        self.model = None
        self.training_dataset = None

    def fit(self, num_topicos, embedding_model, context_size):
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x)
        train_bow_embeddings = vectorizer.fit_transform(self._get_tokens())
        
        vocab = vectorizer.get_feature_names_out()
        id2token = dict(enumerate(vocab))

        modelo_de_embedding = SentenceTransformer(embedding_model)
        train_contextualized_embeddings = modelo_de_embedding.encode(self._get_text())

        self.training_dataset = CTMDataset(
            train_contextualized_embeddings, 
            train_bow_embeddings, 
            id2token
        )

        self.model = CombinedTM(bow_size=len(vocab), contextual_size=context_size, n_components=num_topicos, num_epochs=30)
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