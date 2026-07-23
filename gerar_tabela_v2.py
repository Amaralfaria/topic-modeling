import os
import json
import argparse
import sys

# Definindo as constantes de acordo com a sua estrutura
ALGORITMOS = ['bertopic', 'ctm', 'lda', 'prodlda']
LETRAS = ['a', 'b', 'c']

def escape_latex(text):
    """Escapa caracteres especiais para o LaTeX."""
    return text.replace('_', '\\_')

def obter_melhor_experimento(base_dir, base, algoritmo, exp_num_str, k, top_words):
    """Lê os arquivos a, b, c de um algoritmo/exp/k e retorna os tópicos do que teve maior NPMI."""
    max_npmi = -float('inf')
    best_topics = None
    best_letra = None

    for letra in LETRAS:
        caminho_json = os.path.join(base_dir, base, algoritmo, f"EXP{exp_num_str}{letra}", f"k-{k}", "experiment-data.json")
        
        if os.path.exists(caminho_json):
            try:
                with open(caminho_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extrai a métrica NPMI
                metrics = data.get("metrics", {})
                npmi = metrics.get("npmi", -float('inf'))
                
                # Se for o melhor até agora, atualiza
                if npmi > max_npmi:
                    max_npmi = npmi
                    topics_raw = data.get("topics", {})
                    # Limita a quantidade de top words
                    best_topics = {tid: words[:top_words] for tid, words in topics_raw.items()}
                    best_letra = letra
            except Exception as e:
                print(f"[ERRO] Falha ao processar {caminho_json}: {e}")
                
    return best_topics, max_npmi, best_letra

def generate_comparative_latex_table(best_results, k, top_words, base, exp_num, npmi_info):
    """Gera o código LaTeX da tabela comparativa em xltabular.

    A tabela gerada pode quebrar entre páginas, repete o cabeçalho nas páginas
    seguintes e mantém colunas flexíveis do tipo X para textos longos.
    """
    # Descobre todas as chaves de tópicos únicas (incluindo o -1 do BERTopic)
    all_keys = set()
    for topics in best_results.values():
        if topics:
            all_keys.update(topics.keys())

    if not all_keys:
        return None  # Nenhum dado encontrado para gerar tabela

    # Ordena as chaves numericamente
    sorted_keys = sorted(list(all_keys), key=int)

    # Construção da legenda detalhada informando qual letra foi escolhida para cada algoritmo
    tem_ruido = "-1" in all_keys
    caption_sufix = " O Tópico -1 do BERTopic representa o ruído (outliers)." if tem_ruido else ""
    info_str = ", ".join([
        f"{algo.upper()} (Exp {npmi_info[algo]['letra']}, NPMI: {npmi_info[algo]['npmi']:.4f})"
        for algo in ALGORITMOS if npmi_info.get(algo)
    ])

    caption = (
        f"Comparativo dos tópicos extraídos para a base {base} ($K={k}$, Top {top_words} palavras). "
        f"Para cada algoritmo, foi selecionada a execução com o maior NPMI: {info_str}.{caption_sufix}"
    )
    continuation_caption = (
        f"Comparativo dos tópicos extraídos para a base {base} "
        f"($K={k}$, Top {top_words} palavras) -- continuação."
    )
    label = f"tab:comparativo_{base}_exp{exp_num}_k{k}"

    # xltabular combina quebra de página do longtable com colunas X do tabularx.
    # As colunas dos algoritmos usam \RaggedRight para evitar espaçamentos ruins.
    cols_format = " ".join([r">{\RaggedRight\arraybackslash}X"] * len(ALGORITMOS))
    total_cols = 1 + len(ALGORITMOS)
    headers = ["\\textbf{Tópico}"] + [f"\\textbf{{{algo.upper()}}}" for algo in ALGORITMOS]
    header_line = " & ".join(headers) + " \\\\"

    latex_lines = [
        "% Requer no preâmbulo:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{xltabular}",
        "% \\usepackage{array}",
        "% \\usepackage{ragged2e}",
        "\\begingroup",
        "\\tiny % Fonte pequena recomendada para tabelas com muitas colunas",
        "\\setlength{\\tabcolsep}{3pt} % Reduz o espaçamento horizontal entre colunas",
        "\\renewcommand{\\arraystretch}{1.15} % Aumenta levemente o espaçamento vertical",
        f"\\begin{{xltabular}}{{\\textwidth}}{{@{{}}l {cols_format}@{{}}}}",
        f"    \\caption{{{caption}}}\\label{{{label}}}\\\\",
        "    \\toprule",
        f"    {header_line}",
        "    \\midrule",
        "    \\endfirsthead",
        "",
        f"    \\caption[]{{{continuation_caption}}}\\\\",
        "    \\toprule",
        f"    {header_line}",
        "    \\midrule",
        "    \\endhead",
        "",
        "    \\midrule",
        f"    \\multicolumn{{{total_cols}}}{{r}}{{Continua na próxima página}}\\\\",
        "    \\endfoot",
        "",
        "    \\bottomrule",
        "    \\endlastfoot",
        ""
    ]

    # Preenche as linhas
    for tid in sorted_keys:
        label_topic = str(tid)
        if label_topic == "-1":
            label_topic = "-1 (Ruído)"

        row_items = [f"\\textbf{{{label_topic}}}"]

        # Para cada algoritmo, pega as palavras do tópico
        # ou coloca um traço se não existir, ex.: tópico -1 no LDA.
        for algo in ALGORITMOS:
            topics_dict = best_results.get(algo)
            if topics_dict and tid in topics_dict:
                words_str = ", ".join([escape_latex(w) for w in topics_dict[tid]])
                row_items.append(words_str)
            else:
                row_items.append("-")

        latex_lines.append("    " + " & ".join(row_items) + " \\\\")

    latex_lines.extend([
        "\\end{xltabular}",
        "\\endgroup\n"
    ])

    return "\n".join(latex_lines)

def processar_lote_comparativo(base_dir, base, exp_nums, ks, top_words):
    # Cria uma pasta para salvar os comparativos
    output_dir = os.path.join(base_dir, base, "comparativos")
    os.makedirs(output_dir, exist_ok=True)
    
    arquivos_gerados = 0
    
    for exp_num in exp_nums:
        exp_str = str(exp_num).zfill(2)
        
        for k in ks:
            best_results = {}
            npmi_info = {}
            
            # 1. Pega os melhores resultados para cada algoritmo
            for algoritmo in ALGORITMOS:
                best_topics, max_npmi, best_letra = obter_melhor_experimento(
                    base_dir, base, algoritmo, exp_str, k, top_words
                )
                
                if best_topics is not None:
                    best_results[algoritmo] = best_topics
                    npmi_info[algoritmo] = {"letra": best_letra, "npmi": max_npmi}
                    
            # 2. Gera a tabela LaTeX comparativa se tivermos encontrado dados
            if best_results:
                latex_output = generate_comparative_latex_table(best_results, k, top_words, base, exp_str, npmi_info)
                
                if latex_output:
                    nome_arquivo = f"tabela_comparativa_EXP{exp_str}_k{k}.tex"
                    caminho_tex = os.path.join(output_dir, nome_arquivo)
                    
                    with open(caminho_tex, 'w', encoding='utf-8') as f_out:
                        f_out.write(latex_output)
                        
                    print(f"[SUCESSO] Gerado: {caminho_tex}")
                    arquivos_gerados += 1
            else:
                print(f"[AVISO] Nenhum dado encontrado para Experimento {exp_num}, K={k}")

    print(f"\nConcluído! Foram gerados {arquivos_gerados} arquivos .tex comparativos na pasta '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Script para gerar tabela comparativa LaTeX pegando o melhor NPMI.")
    parser.add_argument("-b", "--base", required=True, help="Nome da base de dados (ex: bills)")
    parser.add_argument("-e", "--exp_nums", required=True, nargs='+', type=int, help="Números dos experimentos. Ex: 1 2")
    parser.add_argument("-k", "--ks", required=True, nargs='+', type=int, help="Valores de K. Ex: 10 20 30")
    parser.add_argument("-t", "--top_words", type=int, default=10, help="Número de top words a exibir (default: 10)")
    parser.add_argument("-d", "--base_dir", default="data/output", help="Diretório raiz (default: data/output)")
    
    args = parser.parse_args()
    
    print("Iniciando processamento comparativo...")
    print(f"Base: {args.base} | Exp Nums: {args.exp_nums} | Ks: {args.ks} | Top Words: {args.top_words}\n")
    
    processar_lote_comparativo(args.base_dir, args.base, args.exp_nums, args.ks, args.top_words)

if __name__ == "__main__":
    main()