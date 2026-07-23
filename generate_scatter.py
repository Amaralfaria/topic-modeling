import os
import json
import argparse
import matplotlib.pyplot as plt

def generate_scatter_plot(sub_dir):
    base_dir = "./data/output"
    target_dir = os.path.join(base_dir, sub_dir)
    
    models = ["bertopic", "ctm", "lda", "prodlda"]
    experiments = ["EXP01a/k-21", "EXP01b/k-21", "EXP01c/k-21"]
    
    # Cores associadas a cada modelo para o gráfico
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    if not os.path.exists(target_dir):
        print(f"Erro: O diretório base '{target_dir}' não existe.")
        return

    plt.figure(figsize=(10, 6))
    has_data = False

    # Itera sobre os modelos para plotar os pontos agrupados por cor/legenda
    for i, model in enumerate(models):
        model_path = os.path.join(target_dir, model)
        
        if not os.path.exists(model_path):
            continue

        npmi_vals = []
        ari_vals = []

        # Percorre as pastas dos experimentos originais
        for exp in experiments:
            file_path = os.path.join(model_path, exp, "experiment-data.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        metrics = data.get("metrics", {})
                        
                        # Extrai as duas métricas desejadas
                        if "npmi" in metrics and "adjusted_rand_index" in metrics:
                            npmi_vals.append(metrics["npmi"])
                            ari_vals.append(metrics["adjusted_rand_index"])
                            has_data = True
                    except json.JSONDecodeError:
                        print(f"Erro: Arquivo JSON corrompido em {file_path}")

        # Se encontrou dados para este modelo, adiciona ao gráfico
        if npmi_vals:
            plt.scatter(ari_vals, npmi_vals, label=model.upper(), color=colors[i], s=100, alpha=0.8, edgecolors='w')

    # Configura e salva o gráfico se houver dados
    if has_data:
        plt.title(f"Dispersão: NPMI vs Adjusted Rand Index (Base: {sub_dir})", fontsize=14)
        plt.xlabel("Adjusted Rand Index (ARI)", fontsize=12)
        plt.ylabel("NPMI", fontsize=12)
        plt.legend(title="Modelos", loc="best")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Salva o arquivo na raiz da base passada (ex: ./data/output/bills/grafico-dispersao.png)
        output_file = os.path.join(target_dir, "grafico-dispersao.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sucesso: Gráfico de dispersão salvo em {output_file}")
    else:
        print("Aviso: Nenhum dado válido encontrado para gerar o gráfico.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um gráfico de dispersão NPMI vs ARI para os experimentos.")
    parser.add_argument("sub_dir", type=str, help="O resto do endereço a partir de ./data/output (ex: bills)")
    
    args = parser.parse_args()
    generate_scatter_plot(args.sub_dir)