from services.experiment_orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    ExperimentOrchestrator("config/templates/config.json").run()
    # lda = MalletLDA("data/input/instagram/pre-processed/train.metadata.jsonl", "tokenized_text")
    # lda.fit(10)
    # save_document_topics("data/input/instagram/pre-processed/train.metadata.jsonl", "data/output/instagram/lda/k-teste/result.jsonl", lda.get_document_topics())
    # save_topics(lda.get_topics(), "data/output/instagram/lda/k-teste/topicos.txt")


    # ctm = CTM("data/input/instagram/pre-processed-complete/train.metadata.jsonl", "text", "tokenized_text")
    # ctm.fit(10, "paraphrase-multilingual-MiniLM-L12-v2", 384)
    # save_document_topics("data/input/instagram/pre-processed-complete/train.metadata.jsonl", "data/output/instagram/ctm/k-teste/result.jsonl", ctm.get_document_topics())
    # save_topics(ctm.get_topics(), "data/output/instagram/ctm/k-teste/topicos.txt")

    # bertopic = Bertopic("data/input/instagram/pre-processed-complete/train.metadata.jsonl", "text", "tokenized_text")
    # bertopic.fit(10, "paraphrase-multilingual-MiniLM-L12-v2")
    # save_document_topics("data/input/instagram/pre-processed-complete/train.metadata.jsonl", "data/output/instagram/bertopic/k-teste/result.jsonl", bertopic.get_document_topics())
    # save_topics(bertopic.get_topics(), "data/output/instagram/bertopic/k-teste/topicos.txt")

    # bertopic_model = get_bertopic("data/input/instagram/instagram.jsonl", "text", i, SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"), list(PT_STOP_WORDS))
    # salvar_topicos_a_documentos(bertopic_model, "data/input/instagram/instagram.jsonl", "data/output/instagram/bertopic/k-" + str(i) + "/result.jsonl")
    # salvar_topicos(bertopic_model, "data/output/instagram/bertopic/k-" + str(i) + "/topics.txt")


    # lda_modelo, corpus_treino, _ = get_lda("data/input/bills/pre-processed/train.metadata.jsonl", "tokenized_text", 21)
    # atribuir_topicos_e_salvar("data/input/bills/pre-processed/train.metadata.jsonl", "data/output/bills/lda/k-21/result.jsonl", lda_modelo, corpus_treino)    

    # lda_modelo, corpus_treino, _ = get_lda("data/input/bills/pre-processed/train.metadata.jsonl", "tokenized_text", 21)
    # atribuir_topicos_e_salvar("data/input/bills/pre-processed/train.metadata.jsonl", "data/output/bills/lda/k-21/result.jsonl", lda_modelo, corpus_treino)    

    # for i in range(10, 60, 10):
    #     lda_modelo, corpus_treino, _ = get_lda("data/input/instagram/pre-processed/train.metadata.jsonl", "tokenized_text", i)
    #     atribuir_topicos_e_salvar("data/input/instagram/pre-processed/train.metadata.jsonl", "data/output/instagram/lda/k-" + str(i) + "/result.jsonl", lda_modelo, corpus_treino)    
    #     salvar_legenda_topicos(lda_modelo, "data/output/instagram/lda/k-" + str(i) + "/topicos.txt")

    # bertopic_model = get_bertopic("data/input/wiki-text/pre-processed/train.metadata.jsonl", "text", 45, SentenceTransformer("all-MiniLM-L6-v2"), list(STOP_WORDS))
    # salvar_topicos_a_documentos(bertopic_model, "data/input/wiki-text/pre-processed/train.metadata.jsonl", "data/output/wiki-text/bertopic/k-45/result.jsonl")
    # salvar_topicos(bertopic_model, "data/output/wiki-text/bertopic/k-45/topics.txt")

    # bertopic_model = get_bertopic("data/input/bills/pre-processed/train.metadata.jsonl", "summary", 21, SentenceTransformer("all-MiniLM-L6-v2"), list(STOP_WORDS))
    # salvar_topicos_a_documentos(bertopic_model, "data/input/bills/pre-processed/train.metadata.jsonl", "data/output/bills/bertopic/k-21/result.jsonl")
    # salvar_topicos(bertopic_model, "data/output/bills/bertopic/k-21/topics.txt")

    # for i in range(10, 60, 10):
    #     bertopic_model = get_bertopic("data/input/instagram/instagram.jsonl", "text", i, SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"), list(PT_STOP_WORDS))
    #     salvar_topicos_a_documentos(bertopic_model, "data/input/instagram/instagram.jsonl", "data/output/instagram/bertopic/k-" + str(i) + "/result.jsonl")
    #     salvar_topicos(bertopic_model, "data/output/instagram/bertopic/k-" + str(i) + "/topics.txt")

    # ctm, traning_dataset = get_ctm("data/input/wiki-text/pre-processed/train.metadata.jsonl", "text", "tokenized_text", 45, "all-MiniLM-L6-v2", 384)
    # atribui_topicos(ctm, "data/input/wiki-text/pre-processed/train.metadata.jsonl", training_dataset=traning_dataset, output_file="data/output/wiki-text/ctm/k-45/result.jsonl")
    # save_topic_definitions(ctm, "data/output/wiki-text/ctm/k-45/topics.txt")

    # ctm, traning_dataset = get_ctm("data/input/bills/pre-processed/train.metadata.jsonl", "summary", "tokenized_text", 21, "all-MiniLM-L6-v2", 384)
    # atribui_topicos(ctm, "data/input/bills/pre-processed/train.metadata.jsonl", training_dataset=traning_dataset, output_file="data/output/bills/ctm/k-21/result.jsonl")
    # save_topic_definitions(ctm, "data/output/bills/ctm/k-21/topics.txt")


    # for i in range(10, 60, 10):
    #     ctm, traning_dataset = get_ctm("data/input/instagram/pre-processed-complete/train.metadata.jsonl", "text", "tokenized_text", i, "paraphrase-multilingual-MiniLM-L12-v2", 384)
    #     atribui_topicos(ctm, "data/input/instagram/pre-processed-complete/train.metadata.jsonl", training_dataset=traning_dataset, output_file="data/output/instagram/ctm/k-" + str(i) + "/result.jsonl")
    #     save_topic_definitions(ctm, "data/output/instagram/ctm/k-" + str(i) + "/topics.txt")

    # metricsService =  MetricsService("data/output/bills/lda/k-21/result.jsonl", "topico_lda", "topic")
    # print(metricsService.get_harmonic_purity())
    # print(metricsService.get_adjusted_rand_index())
    # print(metricsService.get_normalized_mutual_information())
    # print(metricsService.get_npmi("data/output/bills/lda/k-21/topics.txt", "tokenized_text"))
    # print(metricsService.get_topic_diversity("data/output/bills/lda/k-21/topics.txt"))

    # harmonic_purity, ari, mis = metric_calc("data/output/bills/lda/k-21/result.jsonl", "topic", "topico_lda")
    # npmi = get_npmi("data/output/bills/lda/k-21/result.jsonl", "data/output/bills/lda/k-21/topics.txt", "tokenized_text")
    # topic_diversity = get_topic_diversity("data/output/bills/lda/k-21/topics.txt", 10)
    # save_metrics(harmonic_purity, ari, mis, npmi, topic_diversity, "data/output/bills/lda/k-21/metrics.txt")

    # harmonic_purity, ari, mis = metric_calc("data/output/bills/ctm/k-21/result.jsonl", "topic", "topic_id")
    # npmi = get_npmi("data/output/bills/ctm/k-21/result.jsonl", "data/output/bills/ctm/k-21/topics.txt", "tokenized_text")
    # topic_diversity = get_topic_diversity("data/output/bills/ctm/k-21/topics.txt", 10)
    # save_metrics(harmonic_purity, ari, mis, npmi, topic_diversity, "data/output/bills/ctm/k-21/metrics.txt")

    # harmonic_purity, ari, mis = metric_calc("data/output/bills/bertopic/k-21/result.jsonl", "topic", "topic_id")
    # npmi = get_npmi("data/output/bills/bertopic/k-21/result.jsonl", "data/output/bills/bertopic/k-21/topics.txt", "tokenized_text")
    # topic_diversity = get_topic_diversity("data/output/bills/bertopic/k-21/topics.txt", 10)
    # save_metrics(harmonic_purity, ari, mis, npmi, topic_diversity, "data/output/bills/bertopic/k-21/metrics.txt")

    # harmonic_purity, ari, mis = metric_calc("data/output/wiki-text/lda/k-45/result.jsonl", "category", "topico_lda")
    # npmi = get_npmi("data/output/wiki-text/lda/k-45/result.jsonl", "data/output/wiki-text/lda/k-45/topicos.txt", "tokenized_text")
    # topic_diversity = get_topic_diversity("data/output/wiki-text/lda/k-45/topicos.txt", 10)
    # save_metrics(harmonic_purity, ari, mis, npmi, topic_diversity, "data/output/wiki-text/lda/k-45/metrics.txt")

    # harmonic_purity, ari, mis = metric_calc("data/output/wiki-text/ctm/k-45/result.jsonl", "category", "topic_id")
    # npmi = get_npmi("data/output/wiki-text/ctm/k-45/result.jsonl", "data/output/wiki-text/ctm/k-45/topics.txt", "tokenized_text")
    # topic_diversity = get_topic_diversity("data/output/wiki-text/ctm/k-45/topics.txt", 10)
    # save_metrics(harmonic_purity, ari, mis, npmi, topic_diversity, "data/output/wiki-text/ctm/k-45/metrics.txt")

    # for i in range(10, 60, 10):
    #     npmi = get_npmi("data/output/instagram/lda/k-" + str(i) + "/result.jsonl", "data/output/instagram/lda/k-" + str(i) + "/topicos.txt", "tokenized_text")
    #     topic_diversity = get_topic_diversity("data/output/instagram/lda/k-" + str(i) + "/topicos.txt", 10)
    #     save_metrics(0, 0, 0, npmi, topic_diversity, "data/output/instagram/lda/k-" + str(i) + "/metrics.txt")

        # npmi = get_npmi("data/output/instagram/ctm/k-" + str(i) + "/result.jsonl", "data/output/instagram/ctm/k-" + str(i) + "/topics.txt", "tokenized_text")
        # topic_diversity = get_topic_diversity("data/output/instagram/ctm/k-" + str(i) + "/topics.txt", 10)
        # save_metrics(0, 0, 0, npmi, topic_diversity, "data/output/instagram/ctm/k-" + str(i) + "/metrics.txt")
