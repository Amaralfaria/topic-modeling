import os
import json
import argparse

def calculate_metrics_mean(sub_dir):
    base_dir = "./data/output"
    target_dir = os.path.join(base_dir, sub_dir)
    
    models = ["bertopic", "ctm", "lda", "prodlda"]
    experiments = ["EXP01a/k-21", "EXP01b/k-21", "EXP01c/k-21"]

    if not os.path.exists(target_dir):
        print(f"Erro: O diretório base '{target_dir}' não existe.")
        return

    for model in models:
        model_path = os.path.join(target_dir, model)
        
        if not os.path.exists(model_path):
            print(f"Aviso: Pasta do modelo não encontrada, ignorando: {model_path}")
            continue

        metrics_sum = {}
        valid_files_count = 0

        # Percorre as pastas dos experimentos (EXP01a, EXP01b, EXP01c)
        for exp in experiments:
            file_path = os.path.join(model_path, exp, "experiment-data.json")
            
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
                print(f"Aviso: Arquivo não encontrado {file_path}")

        # Se encontrou arquivos válidos, calcula a média e salva
        if valid_files_count > 0:
            metrics_mean = {key: value / valid_files_count for key, value in metrics_sum.items()}
            
            output_data = {
                "metrics": metrics_mean
            }
            
            output_file = os.path.join(model_path, "experiment-mean.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
                
            print(f"Sucesso: Médias calculadas e salvas em {output_file}")
        else:
            print(f"Aviso: Nenhum dado de experimento válido encontrado para o modelo '{model}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula a média das métricas de experimentos.")
    parser.add_argument("sub_dir", type=str, help="O resto do endereço a partir de ./data/output (ex: bills)")
    
    args = parser.parse_args()
    calculate_metrics_mean(args.sub_dir)