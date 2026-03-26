
class Bill:
    def __init__(self, id: str, summary: str, topic: str, subtopic: str, subjects_top_term: str, tokenized_text: str) -> None:
        self.id = id
        self.summary = summary
        self.topic = topic
        self.subtopic = subtopic
        self.subjects_top_term = subjects_top_term
        self.tokenized_text = tokenized_text