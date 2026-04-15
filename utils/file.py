from typing import Union, Any, Optional, Dict, List
from pathlib import Path
import json
from scipy import sparse

def save_json(obj: Any, fpath: Union[str, Path], indent: Optional[int] = None):
    with open(fpath, "w") as outfile:
        json.dump(obj, outfile, indent=indent)

def save_jsonl(
    metadata: List[Dict[str, Any]],
    outpath: Union[str, Path],
    dtm: Optional[sparse.csr_matrix] = None,
    vocab: Optional[Dict[str, int]] = None,
):
    """
    Save a list of dictionaries to a jsonl file. If `dtm` and `vocab` are provided,
    save document-term matrix as a dictionary in the following format, where each
    row is a document:
    {
        "id": <doc_1>,
        <other metadata>: ...
        "tokenized_counts": {
            <word_2>: <count_of_word_2_in_doc_1>,
            <word_6>: <count_of_word_6_in_doc_1>,
            ...
        },
    }
    """

    if dtm is not None and vocab is None:
        raise ValueError("`vocab` must be provided if `dtm` is provided")

    if vocab is not None : 
        inv_vocab = dict(zip(vocab.values(), vocab.keys()))
    
    with open(outpath, mode="w") as outfile:
        for i, data in enumerate(metadata):
            if dtm is not None and vocab is not None:
                row = dtm[i] #type: ignore
                words_in_doc = [inv_vocab[idx] for idx in row.indices] #type: ignore
                counts = [int(v) for v in row.data]   #type: ignore
                word_counts = Dict(zip(words_in_doc, counts)) #type: ignore
                data.update({"tokenized_counts": word_counts}) #type: ignore
            row_json = json.dumps(data)
            if i == 0:
                outfile.write(row_json)
            else:
                outfile.write(f"\n{row_json}")


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
            
            # O primeiro item é o ID do tópico (ex: 0, 1, 2...), 
            # então pegamos do segundo item em diante
            palavras = partes[1:]
            
            meus_topicos.append(palavras)
            
    return {"topics": meus_topicos}
