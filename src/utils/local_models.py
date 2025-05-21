import torch
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinBert:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="ProsusAI/finbert",
            trust_remote_code=True
        )
        self.labels = ["positive", "negative", "neutral"]

    def estimate_sentiment(self, news: List[str]) -> Tuple[float, str]:
        if news:
            tokens = self.tokenizer(news, return_tensors="pt", padding=True).to(self.device)
            result = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
            result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
            probability = float(result[torch.argmax(result)])
            sentiment = self.labels[torch.argmax(result)]
            return probability, sentiment
        else:
            return 0, self.labels[-1]

# if __name__ == "__main__":
#     finbert_model = FinBert("cpu")
#     tensor, sentiment = finbert_model.estimate_sentiment(
#         ["markets responded negatively to the news!", "traders were displeased!"]
#     )
#     print(tensor, sentiment)
