from domain.post import Post
from typing import Any
from pymongo.database import Database
from infra.db.collections import POST_COLLECTIONS

class PostRepository:
    def __init__(self, db: Database[Any]):
        self.db = db

    def find_all(self) -> list[Post]:
        result: list[Post] = []

        for collection in POST_COLLECTIONS:
            result.extend([
                Post(doc["text"]) #type: ignore
                for doc in self.find_all_from_collection(collection)
            ])

        return result

    def find_all_from_collection(self, collection_name: str):
        return self.db[collection_name].find(
            {},
            {"text": 1}
        )