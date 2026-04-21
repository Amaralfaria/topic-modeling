from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
import json
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

        

def get_ctm(caminho_entrada, coluna_texto, coluna_tokens, num_topicos, embedding_model, context_size):
    qt = TopicModelDataPreparation(embedding_model)

    documentos_nao_processados = []
    documentos_processados = []
    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            documentos_nao_processados.append(dados[coluna_texto])
            documentos_processados.append(dados[coluna_tokens])

    training_dataset = qt.fit(text_for_contextual=documentos_nao_processados, text_for_bow=documentos_processados)

    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=context_size, n_components=num_topicos, num_epochs=30) # 50 topics

    ctm.fit(training_dataset) # run the model

    return ctm, training_dataset


def atribui_topicos(ctm, caminho_entrada, training_dataset, output_file):
    documentos = [  ]

    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            documentos.append(dados)

    topic_predictions = ctm.get_thetas(training_dataset)

    # 6. Salvar novo JSONL com os resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(documentos):
            # Pegamos o índice do tópico com maior probabilidade
            predicted_topic_id = int(topic_predictions[i].argmax())
            
            # Adicionamos a informação ao dicionário original
            doc['topic_id'] = predicted_topic_id
            
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"Processamento concluído. Arquivo salvo em: {output_file}")
    

def save_topic_definitions(ctm, output_txt, n_words=10):
    """
    Salva as palavras-chave de cada tópico em um arquivo de texto.
    """
    topic_lists = ctm.get_topic_lists(n_words)
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=== Definição dos Tópicos Gerados ===\n\n")
        for i, words in enumerate(topic_lists):
            line = f"Tópico {i}: {', '.join(words)}\n"
            f.write(line)
            
    print(f"Lista de tópicos salva em: {output_txt}")