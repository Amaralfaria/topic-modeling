from gensim.models.wrappers import LdaMallet
import json
import gensim.corpora as corpora
from utils.file import get_data_from_column

PATH_TO_MALLET = "library/mallet-2.1.0/bin/mallet"

class MalletLDA:
    def __init__(self, caminho_documentos, coluna_tokens):
        self.caminho_documentos = caminho_documentos
        self.coluna_tokens = coluna_tokens
        self.modelo = None
        self.id2word = None
        self.corpus = None
        

    def fit(self, num_topicos):
        self.id2word = corpora.Dictionary(self._get_tokens())
        
        self.id2word.filter_extremes(
            no_below=5, 
            no_above=0.5, 
            keep_n=15000
        )

        self.corpus = [self.id2word.doc2bow(text) for text in self._get_tokens()]

        self.modelo = LdaMallet(
            mallet_path=PATH_TO_MALLET,
            corpus=self.corpus,
            num_topics=num_topicos, 
            id2word=self.id2word,
            iterations=2000,         
            optimize_interval=10,    
            alpha=1.0,               
        )

        return self.modelo
    
    def get_document_topics(self):
        return [self._get_most_likely_topic(dist) for dist in self._get_topic_distributions()]

    def get_topics(self):
        return [
                (id_topico, [palavra for palavra, _ in lista_palavras])
                for id_topico, lista_palavras in self.modelo.show_topics(
                    num_topics=-1, num_words=10, formatted=False
                )
            ]

    def _get_most_likely_topic(self, distribution):
        return max(distribution, key=lambda x: x[1])[0]

    def _get_topic_distributions(self):
        return self.modelo[self.corpus]

    def _get_tokens(self):
        for data in get_data_from_column(self.caminho_documentos, self.coluna_tokens):
            yield data.split()



def atribuir_topicos_e_salvar(caminho_entrada, caminho_saida, modelo_lda, corpus, nome_coluna="topico_lda"):
    """
    Atribui o tópico mais provável a cada documento e salva em um novo arquivo JSONL,
    seguindo a lógica de avaliação do artigo.
    
    Args:
        caminho_entrada (str): Caminho do arquivo .jsonl original.
        caminho_saida (str): Caminho onde o novo .jsonl será salvo.
        modelo_lda: O objeto do modelo LdaMallet treinado.
        corpus: O corpus (Bag-of-Words) utilizado no treino.
        nome_coluna (str): Nome da nova coluna para o tópico atribuído.
    """
    
    # 1. Obter a distribuição de tópicos para todos os documentos
    # O Mallet retorna uma lista de listas com as probabilidades de cada tópico
    distribuicoes = modelo_lda[corpus]
    
    # 2. Ler o arquivo original e atualizar as linhas
    novas_linhas = []
    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        for i, linha in enumerate(f):
            dados = json.loads(linha)
            
            # Pegar a distribuição do documento atual
            doc_topicos = distribuicoes[i]
            
            # Identificar o tópico mais provável (maior probabilidade)
            # doc_topicos é uma lista de tuplas [(id_topico, prob), ...]
            # Seguindo o critério do artigo: "assign each document to its most probable topic"
            topico_dominante = max(doc_topicos, key=lambda x: x[1])[0]
            
            # Adicionar a nova coluna com o ID do tópico
            dados[nome_coluna] = int(topico_dominante)
            novas_linhas.append(dados)
            
    # 3. Salvar o resultado no novo arquivo JSONL
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        for item in novas_linhas:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Processamento concluído! Resultados salvos em: {caminho_saida}")


def salvar_legenda_topicos(modelo_lda, caminho_txt, num_palavras=10):
    """
    Gera um arquivo de texto com a lista de tópicos e suas palavras principais.
    """
    # O método show_topics retorna uma lista de (id_do_topico, "0.01*palavra1 + 0.02*palavra2...")
    topicos = modelo_lda.show_topics(num_topics=-1, num_words=num_palavras, formatted=False)

    with open(caminho_txt, 'w', encoding='utf-8') as f:
        for id_topico, lista_palavras in topicos:
            # lista_palavras é algo como: [('trump', 0.031), ('eua', 0.027)...]
            # Extraímos apenas o primeiro elemento de cada tupla (a palavra)
            palavras_apenas = [palavra for palavra, peso in lista_palavras]
            
            # Criamos a linha: "ID palavra1 palavra2 palavra3..."
            # Se preferir com vírgulas, use: ", ".join(palavras_apenas)
            linha = f"{id_topico} {' '.join(palavras_apenas)}\n"
            
            f.write(linha)

    print(f"Legenda de tópicos salva em: {caminho_txt}")
