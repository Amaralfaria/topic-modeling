import json
from typing import Iterator
from domain.bills import Bill

class BillsRepository:
    def __init__(self) -> None:
        self._path = "data/bills_subset.jsonl"

    def find_all(self) -> Iterator[Bill]:
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                bill = Bill(json_line["id"], json_line["summary"], json_line["topic"], json_line["subtopic"], json_line["subjects_top_term"], json_line["tokenized_text"])
                yield bill
