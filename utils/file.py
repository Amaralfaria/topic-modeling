from pathlib import Path
from datetime import datetime
import json


def get_data_from_column(document_path, desired_column):
    with open(document_path, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            yield dados[desired_column]


def add_column_to_jsonl(original_data_path, output_path, new_column_values, column_name):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(original_data_path, 'r', encoding='utf-8') as f_in, \
        open(output_path, 'w', encoding='utf-8') as f_out:
        
        for i, linha in enumerate(f_in):
            dados = json.loads(linha)
            dados[column_name] = int(new_column_values[i])
            f_out.write(json.dumps(dados, ensure_ascii=False) + '\n')

def save_json(path, object):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(object, f, indent=4, ensure_ascii=False)

def save_metrics(output_path, model_name, params, metrics, topics):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "model": model_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params
        },
        "metrics": metrics,
        "topics": {str(id_t): words for id_t, words in topics}
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def save_topics(topics_list, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for id_topico, words_list in topics_list:
            f.write(f"{id_topico} {' '.join(words_list)} \n")

def load_json(path):
    with open(path) as f:
        return json.load(f)


def txt_to_octis_format(filepath):
    meus_topicos = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for linha in f:
            # Remove espaços em branco no início/fim e pula linhas vazias
            linha = linha.strip()
            if not linha:
                continue
            
            # Divide a linha por espaços ou tabs
            partes = linha.split()
            
            if int(partes[0]) < 0:
                continue
            # O primeiro item é o ID do tópico (ex: 0, 1, 2...), 
            # então pegamos do segundo item em diante
            palavras = partes[1:]
            
            meus_topicos.append(palavras)
            
    return {"topics": meus_topicos}
