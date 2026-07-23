import os
import json
import argparse
import matplotlib.pyplot as plt

def generate_k_graphs(sub_dir):
    base_dir = "./data/output"
    target_dir = os.path.join(base_dir, sub_dir)
    
    models = ["bertopic", "ctm", "lda", "prodlda"]
    k_values_str = ["k-10", "k-20", "k-30", "k-40", "k-50"]
    k_values_int = [10, 20, 30, 40, 50]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    if not os.path.exists(target_dir):
        print(f"Erro: O diretório base '{target_dir}' não existe.")
        return

    # Dicionários para armazenar as métricas de cada modelo
    npmi_data = {model: [] for model in models}
    td_data = {model: [] for model in models}

    # Extrai os dados dos arquivos de média
    for model in models:
        for k_str in k_values_str:
            file_path = os.path.join(target_dir, model, f"experiment-mean-{k_str}.json")
            
            # Se o arquivo existir, lê os dados. Se não, preenche com None.
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        metrics = data.get("metrics", {})
                        npmi_data[model].append(metrics.get("npmi", None))
                        td_data[model].append(metrics.get("topic_diversity", None))
                    except json.JSONDecodeError:
                        print(f"Erro: Arquivo JSON corrompido em {file_path}")
                        npmi_data[model].append(None)
                        td_data[model].append(None)
            else:
                print(f"Aviso: Arquivo de média não encontrado: {file_path}")
                npmi_data[model].append(None)
                td_data[model].append(None)

    # --- FUNÇÃO AUXILIAR PARA PLOTAR ---
    def plot_metric(metric_dict, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        has_data = False
        
        for i, model in enumerate(models):
            # Filtra os valores None (caso algum arquivo tenha faltado) para a plotagem
            valid_k = []
            valid_metric = []
            for j, val in enumerate(metric_dict[model]):
                if val is not None:
                    valid_k.append(k_values_int[j])
                    valid_metric.append(val)
            
            if valid_metric:
                plt.plot(valid_k, valid_metric, label=model.upper(), 
                         color=colors[i], marker=markers[i], markersize=8, linewidth=2, alpha=0.8)
                has_data = True
                
        if has_data:
            plt.title(f"{title} (Base: {sub_dir})", fontsize=14)
            plt.xlabel("Número de Tópicos (k)", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            # Garante que o eixo X mostre apenas os valores discretos exatos (10, 20, 30, 40, 50)
            plt.xticks(k_values_int)
            plt.legend(title="Modelos", loc="best")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            output_file = os.path.join(target_dir, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Sucesso: Gráfico salvo em {output_file}")
        else:
            print(f"Aviso: Nenhum dado válido para gerar o gráfico de {ylabel}.")

    # Gera o gráfico de NPMI
    plot_metric(npmi_data, "Evolução do NPMI por Número de Tópicos", "NPMI", "grafico-media-npmi.png")
    
    # Gera o gráfico de Topic Diversity
    plot_metric(td_data, "Evolução da Topic Diversity por Número de Tópicos", "Topic Diversity", "grafico-media-topic-diversity.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera gráficos discretos (k vs NPMI/Topic Diversity) a partir das médias.")
    parser.add_argument("sub_dir", type=str, help="O resto do endereço a partir de ./data/output (ex: bills)")
    
    args = parser.parse_args()
    generate_k_graphs(args.sub_dir)