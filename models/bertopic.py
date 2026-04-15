import json
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

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