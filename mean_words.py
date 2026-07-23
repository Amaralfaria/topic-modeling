import json
import argparse


def count_words(text):
    if not isinstance(text, str):
        return 0

    return len(text.split())


def main():
    parser = argparse.ArgumentParser(
        description="Calcula estatísticas de quantidade de palavras em uma propriedade de um arquivo JSONL."
    )

    parser.add_argument(
        "jsonl_file",
        help="Caminho para o arquivo JSONL"
    )

    parser.add_argument(
        "property_name",
        help="Nome da propriedade que contém o texto"
    )

    args = parser.parse_args()

    total_words = 0
    document_count = 0
    min_words = None
    max_words = None

    with open(args.jsonl_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Linha {line_number}: JSON inválido.")
                continue

            text = obj.get(args.property_name)

            if text is None:
                print(
                    f"Linha {line_number}: propriedade '{args.property_name}' não encontrada."
                )
                continue

            word_count = count_words(text)

            total_words += word_count
            document_count += 1

            if min_words is None or word_count < min_words:
                min_words = word_count

            if max_words is None or word_count > max_words:
                max_words = word_count

    if document_count == 0:
        print("Nenhum documento válido encontrado.")
        return

    average_words = total_words / document_count

    print(f"Documentos analisados: {document_count}")
    print(f"Total de palavras: {total_words}")
    print(f"Média de palavras por documento: {average_words:.2f}")
    print(f"Menor documento: {min_words} palavras")
    print(f"Maior documento: {max_words} palavras")


if __name__ == "__main__":
    main()