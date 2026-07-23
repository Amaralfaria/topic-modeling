from services.experiment_orchestrator import ExperimentOrchestrator
from services.metrics import MetricsService


import json
from services.experiment_orchestrator import ExperimentOrchestrator
from services.metrics import MetricsService

def get_metrics(result_path, output_col, ground_truth_col, tokens_col, topics):
    metrics_service = MetricsService(result_path, output_col, ground_truth_col)
    # Correção da ordem dos parâmetros para alinhar com a assinatura do método criado
    return metrics_service.get_npmi_per_topic(topics=topics, coluna_tokens=tokens_col)


def calcular_npmi_dos_topicos_salvos(topics_json_path, dataset_path, tokens_col):
    """
    Lê os tópicos salvos em JSON, calcula o NPMI individual e imprime o ranking
    para a análise qualitativa (Top-2 e Bottom-2).
    """
    # 1. Ler o arquivo JSON que contém os resultados (o que você enviou)
    with open(topics_json_path, 'r', encoding='utf-8') as f:
        resultados_json = json.load(f)
        
    # 2. Extrair os tópicos e converter do formato dict para lista de tuplas
    # O JSON armazena as chaves como strings ("0", "1"), então convertemos para int
    topicos_dit = resultados_json.get("topics", {})
    topicos_formatados = [(int(id_t), palavras) for id_t, palavras in topicos_dit.items()]
    
    # 3. Instanciar o MetricsService
    # O result_file deve ser o caminho do seu dataset (o arquivo que contém os documentos)
    metrics_service = MetricsService(result_file=dataset_path, output_column=None, ground_truth_column=None)
    
    # 4. Calcular o NPMI por tópico
    npmi_por_topico = metrics_service.get_npmi_per_topic(topics=topicos_formatados, coluna_tokens=tokens_col)
    
    # 5. Imprimir os resultados (Ordenados por ID)
    print("=== NPMI por Tópico (Ordenado por ID) ===")
    for id_t, score in npmi_por_topico:
        print(f"Tópico {id_t}: {score:.4f}")
        
    # 6. Identificar e imprimir o Top-2 e o Bottom-2 para a nossa Análise Qualitativa (Seção 6.3.2)
    print("\n=== Ranking para Análise Qualitativa ===")
    
    # Ordena a lista com base no segundo elemento da tupla (o escore NPMI)
    npmi_ordenado = sorted(npmi_por_topico, key=lambda x: x[1])
    
    print("Bottom-2 (Menor Coesão - Piores):")
    print(f"1. Tópico {npmi_ordenado[0][0]} (Score: {npmi_ordenado[0][1]:.4f}) -> {topicos_dit[str(npmi_ordenado[0][0])]}")
    print(f"2. Tópico {npmi_ordenado[1][0]} (Score: {npmi_ordenado[1][1]:.4f}) -> {topicos_dit[str(npmi_ordenado[1][0])]}")
    
    print("\nTop-2 (Maior Coesão - Melhores):")
    # Pega os dois últimos elementos da lista ordenada
    print(f"1. Tópico {npmi_ordenado[-1][0]} (Score: {npmi_ordenado[-1][1]:.4f}) -> {topicos_dit[str(npmi_ordenado[-1][0])]}")
    print(f"2. Tópico {npmi_ordenado[-2][0]} (Score: {npmi_ordenado[-2][1]:.4f}) -> {topicos_dit[str(npmi_ordenado[-2][0])]}")


if __name__ == "__main__":
    #  calcular_npmi_dos_topicos_salvos(
    #      topics_json_path='caminho/para/seu/lda_output.json', 
    #      dataset_path='caminho/para/seu/dataset_instagram.jsonl', # Ou .json
    #      tokens_col='nome_da_coluna_de_tokens'
    # )

    ExperimentOrchestrator("config/instagram-no-jornal.json").run(3)