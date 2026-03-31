from spacy.language import Language
from typing import Optional, List, Union, Iterable, Tuple, Iterator, Dict
import spacy
from spacy.tokens import Token
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse

class Preprocessor:
    def __init__(
        self,
        docs: Iterable[Union[Tuple[str, str], str]],
        model_name: str = "en_core_web_sm",
        stopwords: Optional[Iterable[str]] = None,
        detect_entities: bool = True,
        detect_noun_chunks: bool = True,
        filter_entities: Optional[List[str]] = ['ORG', 'PERSON', 'FACILITY', 'GPE', 'LOC'],
        min_doc_freq: float = 1.0,
        max_doc_freq: float = 1.0,
        max_vocab_size: Optional[int] = None,
        vocabulary: Optional[Union[Iterable[str], Dict[str, int]]] = None,
        min_doc_size: Optional[int] = 1,
    ):
        self.model_name = model_name
        self.detect_entities = detect_entities
        self.detect_noun_chunks = detect_noun_chunks
        self.filter_entities = filter_entities
        self.docs = docs
        self.stopwords = stopwords
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq = max_doc_freq
        self.max_vocab_size = max_vocab_size
        self.vocabulary = vocabulary
        self.min_doc_size = min_doc_size

    def create_document_term_matrix(self, retain_text: bool) -> Tuple[sparse.csr_matrix, Dict[str, int], List[Dict[str, str]]]:
        doc_tokens = self.tokenize()
        # for doc in doc_tokens:
        #     print(doc)

        cv = self.get_count_vectorizer()
        dtm = cv.fit_transform(doc[0] for doc in doc_tokens) #type: ignore
        vocab = self.get_vocabulary(cv)
        metadata = self.get_metadata(doc_tokens, cv, retain_text)

        doc_counts = np.array(dtm.sum(1)).squeeze() # type: ignore
        if doc_counts.min() < self.min_doc_size: # type: ignore
            docs_to_keep = doc_counts  >= min_doc_size # type: ignore
            dtm = dtm[docs_to_keep] # type: ignore
            metadata = [md for idx, md in enumerate(metadata) if docs_to_keep[idx]]

        return dtm, vocab, metadata # type: ignore


    def get_count_vectorizer(self):
        return CountVectorizer(
            analyzer=lambda x: x, #type: ignore
            min_df=float(self.min_doc_freq) if self.min_doc_freq < 1 else int(self.min_doc_freq),
            max_df=float(self.max_doc_freq) if self.max_doc_freq <=1 else int(self.max_doc_freq),
            max_features=self.max_vocab_size,
            vocabulary=self.vocabulary,
        )

    def get_metadata(self, doc_tokens: Iterator[Union[Tuple[str, str], str]], cv: CountVectorizer, retain_text: bool) -> List[Dict[str, str]]:
        metadata = [data for _, data in self.docs]

        if retain_text:
            vocab = self.get_vocabulary(cv)
            
            metadata = [ #type: ignore
                {**data, "tokenized_text": " ".join(w for w in doc if w in vocab)} #type: ignore
                for data, (doc, _) in zip(metadata, doc_tokens) #type: ignore
            ]
            
        return metadata #type: ignore


    def get_vocabulary(self, cv: CountVectorizer) -> Dict[str, int]:
        return {k: int(v) for k, v in sorted(cv.vocabulary_.items(), key=lambda kv: kv[1])} #type: ignore

    def tokenize(self) -> Iterator[Union[Tuple[str, str], str]]:
        spacy_model = self.create_pipeline()

        for doc in spacy_model.pipe(self.docs, as_tuples=True, n_process=1): #type: ignore
            doc, id = doc #type: ignore

            tokens = [text for tok in doc for text in self.to_string(tok) if self.keep(text)] #type: ignore
            if not tokens:
                continue

            tokens = tokens, id # type: ignore
            yield tokens # type: ignore

    def create_pipeline(self) -> Language:
        nlp = spacy.load(
            self.model_name,
            disable=['lemmatizer', 'ner', 'tagger', 'parser', 'tok2vec', 'attribute_ruler'],
        )

        if self.detect_entities:
            nlp.enable_pipe("ner")
        if self.detect_noun_chunks:
            nlp.enable_pipe("tok2vec")
            nlp.enable_pipe("tagger")
            nlp.enable_pipe("attribute_ruler")
            nlp.enable_pipe("parser")

        any_phrases = self.detect_entities or self.detect_noun_chunks
        if any_phrases:
            nlp.add_pipe(
                "merge_phrases",
                config={
                    "filter_entities": self.filter_entities,
                },
            )

        return nlp

    def keep(self, token: str) -> bool:
        if self.stopwords and token in self.stopwords:
            return False
        return True

    
    def to_string(self, token: Token) -> Tuple[str]:
        text = token.lower_
        text = text.strip()

        return (text.replace(" ", "_"), )
