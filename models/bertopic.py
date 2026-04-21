import json
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


def get_bertopic(caminho_entrada, coluna_texto, num_topicos, embedding_model, stop_words):
    documentos = []
    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            documentos.append(dados[coluna_texto])

    print("Iniciando treinamento do BERTopic...")
    vectorizer_model = CountVectorizer(stop_words=stop_words)
    modelo_bertopic = BERTopic(
        embedding_model=embedding_model,
        nr_topics=num_topicos,
        verbose=True,
        vectorizer_model=vectorizer_model
    )

    modelo_bertopic.fit_transform(documentos)

    return modelo_bertopic


def salvar_topicos_a_documentos(topic_model, caminho_entrada, output_jsonl):
    documentos = []

    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            documentos.append(dados)

    topics = topic_model.topics_

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, obj in enumerate(documentos):
            obj['topic_id'] = int(topics[idx])
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def salvar_topicos(topic_model, output_txt):
    with open(output_txt, 'w', encoding='utf-8') as f:
        info_topicos = topic_model.get_topics()
        
        for topic_id, words_score in info_topicos.items():
            header = f"Tópico {topic_id}:" if topic_id != -1 else "Tópico -1 (Ruído/Outliers):"
            
            lista_formatada = [f"{word} ({score:.4f})" for word, score in words_score[:10]]
            f.write(f"{header}\n")
            f.write(f"{', '.join(lista_formatada)}\n\n")