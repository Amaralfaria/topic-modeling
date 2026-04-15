from gensim.models.wrappers import LdaMallet
import json
import gensim.corpora as corpora

path_to_mallet_binary = "library/mallet-2.1.0/bin/mallet"

def get_lda(caminho_jsonl, coluna_texto, num_topicos):
    """
    Replica a configuração de LDA Mallet do artigo TopicGPT.
    
    Args:
        caminho_jsonl (str): Caminho para o arquivo .jsonl
        coluna_texto (str): Nome da chave no JSON que contém o texto pré-processado (lista de tokens    )
        num_topicos (int): O 'k' desejado (Ex: 31 para Wiki ou 79 para Bills) [cite: 197]
        path_mallet (str): Caminho para o binário do Mallet no sistema
    """
    
    # 1. Carregamento dos dados
    documentos = []
    with open(caminho_jsonl, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            # O artigo assume que o texto já passou por lematização/limpeza [cite: 139]
            documentos.append(dados[coluna_texto].split())

    # 2. Preparação do Vocabulário (Dicionário)
    id2word = corpora.Dictionary(documentos)
    
    # Restrição rigorosa do artigo: Vocabulário |V| de 15.000 termos 
    id2word.filter_extremes(
        no_below=5, 
        no_above=0.5, 
        keep_n=15000
    )
    for i in range(1,5):
        print(documentos[i])

    # 3. Criação do Corpus (Bag-of-Words)
    corpus = [id2word.doc2bow(text) for text in documentos]

    # 4. Inicialização do Modelo com hiperparâmetros da Seção 4.2 do PDF
    modelo = LdaMallet(
        mallet_path=path_to_mallet_binary,
        corpus=corpus,
        num_topics=num_topicos, # k controlado para comparação justa [cite: 136]
        id2word=id2word,
        iterations=2000,        # Definido no artigo 
        optimize_interval=10,   # Otimização a cada 10 intervalos 
        alpha=1.0,              # Valor fixo de alfa 
        # Nota: O Mallet usa um parâmetro interno para beta, o artigo fixa em 0.1 
    )

    return modelo, corpus, id2word

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
    topicos = modelo_lda.show_topics(num_topics=-1, num_words=num_palavras, formatted=True)
    
    with open(caminho_txt, 'w', encoding='utf-8') as f:
        f.write("=== LEGENDA DOS TÓPICOS GENERADOS ===\n\n")
        for id_topico, palavras in topicos:
            linha = f"Tópico {id_topico}: {palavras}\n"
            f.write(linha)
            
    print(f"Legenda de tópicos salva em: {caminho_txt}")
