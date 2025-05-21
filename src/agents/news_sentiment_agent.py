import torch.cuda
import yfinance as yf
from datetime import datetime
from typing import List

from utils.local_models import FinBert


class NewsSentimentAgent:
    def __init__(self, ticker: str, news_count: int = 10):
        self.ticker = yf.Ticker(ticker)
        self.news_count = news_count
        self.news_list = None

    def get_news(self) -> List[str]:
        processed_news_list = []

        for news in self.ticker.get_news(count=self.news_count):
            title = f"Title: {news.get("content").get("title")}"
            summary = f"Summary: {news.get("content").get("summary")}"
            # datetime_ = f"Datetime: {datetime.fromisoformat(news.get("content").get("pubDate").replace("Z", ""))}"
            processed_news_list.append(
                "\n".join([title, summary])
            )
        return processed_news_list

    def get_sentiment(self):
        fin_bert = FinBert(device="cuda" if torch.cuda.is_available() else "cpu")
        if self.news_list is None:
            self.news_list = self.get_news()
        probability, sentiment = fin_bert.estimate_sentiment(self.news_list)
        return probability, sentiment


# if __name__ == '__main__':
#     sentiment_agent = SentimentAgent(ticker="AAPL", news_count=10)
#     print(sentiment_agent.get_sentiment())
