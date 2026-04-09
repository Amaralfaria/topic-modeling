from models.lda import get_lda, atribuir_topicos_e_salvar

if __name__ == "__main__":
    lda_modelo, corpus_treino, _ = get_lda("data/input/wiki-text/pre-processed/train.metadata.jsonl", "tokenized_text", 31)
    atribuir_topicos_e_salvar("data/input/wiki-text/pre-processed/train.metadata.jsonl", "data/output/wiki-text/result.jsonl", lda_modelo, corpus_treino)    