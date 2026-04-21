from pathlib import Path
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

def load_json(path):
    with open(path) as f:
        return json.load(f)