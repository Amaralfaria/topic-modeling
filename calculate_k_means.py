import os
import json
import argparse

def calculate_k_metrics_mean(sub_dir):
    base_dir = "./data/output"
    target_dir = os.path.join(base_dir, sub_dir)
    
    models = ["bertopic", "ctm", "lda", "prodlda"]
    experiments = ["EXP06a", "EXP06b", "EXP06c"]
    k_values = ["k-10", "k-20", "k-30", "k-40", "k-50"]

    if not os.path.exists(target_dir):
        print(f"Erro: O diretório base '{target_dir}' não existe.")
        return

    # Itera sobre os modelos
    for model in models:
        model_path = os.path.join(target_dir, model)
        
        if not os.path.exists(model_path):
            print(f"Aviso: Pasta do modelo não encontrada, ignorando: {model_path}")
            continue

        # Itera sobre cada valor de k
        for k_val in k_values:
            metrics_sum = {}
            valid_files_count = 0

            # Percorre as pastas dos experimentos (EXP01a, EXP01b, EXP01c)
            # para procurar a subpasta do k atual
            for exp in experiments:
                # O caminho agora inclui o exp e o k_val
                # Ex: ./data/output/bills/bertopic/EXP01a/k-10/experiment-data.json
                file_path = os.path.join(model_path, exp, k_val, "experiment-data.json")
                
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            metrics = data.get("metrics", {})
                            
                            # Inicializa as chaves do dicionário de soma na primeira leitura
                            if not metrics_sum:
                                for key in metrics:
                                    metrics_sum[key] = 0.0
                            
                            # Soma os valores de cada métrica
                            for key, value in metrics.items():
                                if key in metrics_sum:
                                    metrics_sum[key] += value
                                    
                            valid_files_count += 1
                        except json.JSONDecodeError:
                            print(f"Erro: Arquivo JSON corrompido em {file_path}")
                else:
                    pass # Ocultamos o aviso por arquivo não encontrado para não poluir o terminal, 
                         # mas você pode adicionar um print aqui se quiser debuggar.

            # Se encontrou arquivos válidos para este "k", calcula a média e salva
            if valid_files_count > 0:
                metrics_mean = {key: value / valid_files_count for key, value in metrics_sum.items()}
                
                output_data = {
                    "metrics": metrics_mean
                }
                
                # O arquivo é salvo na raiz do modelo com o sufixo k correspondente
                # Ex: ./data/output/bills/bertopic/experiment-mean-k-10.json
                output_file = os.path.join(model_path, f"experiment-mean-{k_val}.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
                    
                print(f"Sucesso: Média para {model} ({k_val}) salva em {output_file}")
            else:
                print(f"Aviso: Nenhum dado encontrado para o modelo '{model}' com '{k_val}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula a média das métricas de experimentos agrupados por k.")
    parser.add_argument("sub_dir", type=str, help="O resto do endereço a partir de ./data/output (ex: bills)")
    
    args = parser.parse_args()
    calculate_k_metrics_mean(args.sub_dir)