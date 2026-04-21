from gensim.models.wrappers import LdaMallet
import gensim.corpora as corpora
from utils.file import get_data_from_column

PATH_TO_MALLET = "library/mallet-2.1.0/bin/mallet"

class MalletLDA:
    def __init__(self, caminho_documentos, coluna_tokens):
        self.caminho_documentos = caminho_documentos
        self.coluna_tokens = coluna_tokens
        self.modelo = None
        self.id2word = None
        self.corpus = None
        

    def fit(self, num_topicos):
        self.id2word = corpora.Dictionary(self._get_tokens())
        
        self.id2word.filter_extremes(
            no_below=5, 
            no_above=0.5, 
            keep_n=15000
        )

        self.corpus = [self.id2word.doc2bow(text) for text in self._get_tokens()]

        self.modelo = LdaMallet(
            mallet_path=PATH_TO_MALLET,
            corpus=self.corpus,
            num_topics=num_topicos, 
            id2word=self.id2word,
            iterations=2000,         
            optimize_interval=10,    
            alpha=1.0,               
        )

        return self.modelo
    
    def get_document_topics(self):
        return [self._get_most_likely_topic(dist) for dist in self._get_topic_distributions()]

    def get_topics(self):
        return [
                (id_topico, [palavra for palavra, _ in lista_palavras])
                for id_topico, lista_palavras in self.modelo.show_topics(
                    num_topics=-1, num_words=10, formatted=False
                )
            ]

    def _get_most_likely_topic(self, distribution):
        return max(distribution, key=lambda x: x[1])[0]

    def _get_topic_distributions(self):
        return self.modelo[self.corpus]

    def _get_tokens(self):
        for data in get_data_from_column(self.caminho_documentos, self.coluna_tokens):
            yield data.split()