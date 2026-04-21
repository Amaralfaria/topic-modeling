from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from utils.file import get_data_from_column

class Bertopic:
    def __init__(self, documents_path, text_column):
        self.documents_path = documents_path
        self.text_column = text_column
        self.model = None
        
    def fit(self, n_topics, embedding_model, stop_words):
        vectorizer_model = CountVectorizer(stop_words=stop_words)
        self.model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            verbose=True,
            vectorizer_model=vectorizer_model
        )

        self.model.fit_transform(self._get_text())

        return self.model

    def get_document_topics(self):
        return self.model.topics_

    def get_topics(self):
        return [
            (id, [word for word, _ in words_score])
            for id, words_score in self.model.get_topics().items()
        ]

    def _get_text(self):
        return list(get_data_from_column(self.documents_path, self.text_column))