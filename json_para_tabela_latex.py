import os
import json
import argparse
import sys

# Definindo as constantes de acordo com a sua estrutura
ALGORITMOS = ['bertopic', 'ctm', 'lda', 'prodlda']
LETRAS = ['a', 'b', 'c']  # Letras esperadas nas pastas de experimento

def escape_latex(text):
    """Escapa caracteres especiais para o LaTeX."""
    return text.replace('_', '\\_')

def generate_latex_table(json_data, num_topicos):
    """Gera o código LaTeX da tabela usando os dados do JSON e os metadados."""
    # Extrai metadados
    metadata = json_data.get("metadata", {})
    model_name = metadata.get("model", "Modelagem de Tópicos")
    topics = json_data.get("topics", {})
    
    latex_lines = [
        "\\begin{table}[htpb]",
        "    \\centering",
        "    \\scriptsize",
        "    \\begin{tabularx}{\\textwidth}{@{}l X@{}}",
        "        \\toprule",
        "        \\textbf{Tópico} & \\textbf{Palavras-chave} \\\\",
        "        \\midrule"
    ]
    
    # Ordena as chaves dos tópicos numericamente
    try:
        sorted_topic_keys = sorted(topics.keys(), key=int)
    except ValueError:
        print("Aviso: Chaves dos tópicos não são inteiros, ordenando alfabeticamente.")
        sorted_topic_keys = sorted(topics.keys())
        
    tem_ruido = False
    for topic_id in sorted_topic_keys:
        words = topics[topic_id]
        escaped_words = [escape_latex(w) for w in words]
        words_str = ", ".join(escaped_words)
        
        # Trata o tópico de ruído (-1)
        if str(topic_id) == "-1":
            label = "-1 (Ruído)"
            tem_ruido = True
        else:
            label = str(topic_id)
            
        latex_lines.append(f"        \\textbf{{{label}}} & {words_str} \\\\")
        
    caption_sufix = " O Tópico -1 representa o ruído (outliers)." if tem_ruido else ""
    
    latex_lines.extend([
        "        \\bottomrule",
        "    \\end{tabularx}",
        f"    \\caption{{Tópicos extraídos pelo modelo {model_name} com $K={num_topicos}$ tópicos.{caption_sufix}}}",
        f"    \\label{{tab:topicos_{model_name.lower()}_{num_topicos}}}",
        "\\end{table}\n"
    ])
    
    return "\n".join(latex_lines)

def processar_experimentos(base_dir, base, exp_nums, ks):
    arquivos_gerados = 0
    
    for algoritmo in ALGORITMOS:
        for exp_num in exp_nums:
            # Formata o número do experimento. Ex: 1 -> "01"
            exp_str = str(exp_num).zfill(2)
            
            for letra in LETRAS:
                pasta_exp = f"EXP{exp_str}{letra}"
                
                for k in ks:
                    pasta_k = f"k-{k}"
                    
                    # Constrói o caminho completo até a pasta final
                    caminho_pasta = os.path.join(base_dir, base, algoritmo, pasta_exp, pasta_k)
                    caminho_json = os.path.join(caminho_pasta, "experiment-data.json")
                    
                    if os.path.exists(caminho_json):
                        try:
                            with open(caminho_json, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            latex_output = generate_latex_table(data, k)
                            
                            caminho_tex = os.path.join(caminho_pasta, "tabela_topicos.tex")
                            with open(caminho_tex, 'w', encoding='utf-8') as f_out:
                                f_out.write(latex_output)
                                
                            print(f"[SUCESSO] Tabela gerada em: {caminho_tex}")
                            arquivos_gerados += 1
                        except Exception as e:
                            print(f"[ERRO] Falha ao processar {caminho_json}: {e}")
                    else:
                        # Opcional: imprimir quando o arquivo não existe para fins de debug
                        # print(f"[IGNORADO] Arquivo não encontrado: {caminho_json}")
                        pass
                        
    print(f"\nConcluído! Foram gerados {arquivos_gerados} arquivos .tex.")

def main():
    parser = argparse.ArgumentParser(description="Script para gerar tabelas LaTeX a partir de json de modelagem de tópicos.")
    parser.add_argument("-b", "--base", required=True, help="Nome da base de dados (ex: bills)")
    parser.add_argument("-e", "--exp_nums", required=True, nargs='+', type=int, help="Números dos experimentos. Ex: 1 2 3")
    parser.add_argument("-k", "--ks", required=True, nargs='+', type=int, help="Valores de K (número de tópicos). Ex: 10 20 30")
    parser.add_argument("-d", "--base_dir", default="data/output", help="Diretório raiz onde os outputs estão salvos (default: data/output)")
    
    args = parser.parse_args()
    
    print(f"Iniciando processamento...\nBase: {args.base}\nAlgoritmos: {ALGORITMOS}\nExperimentos: {args.exp_nums}\nLetras: {LETRAS}\nKs: {args.ks}\n")
    
    processar_experimentos(args.base_dir, args.base, args.exp_nums, args.ks)

if __name__ == "__main__":
    main()