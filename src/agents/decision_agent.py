import os
import sys
import logfire
from datetime import datetime
from pydantic_ai import Agent, RunContext
from typing import Any, List, Dict

from utils.api_models import get_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logfire.configure(send_to_logfire='if-token-present')

model = get_model(branch='google')

system_prompt = """
You are a decision agent for an intraday stock trading system, designed to make trading decision for stocks based on inputs from a sentiment agent and an indicators analysis agent.
Your goal is to output a trading decision (buy, sell or hold) for each stock.

Inputs:
1. Sentiment Agent Output:
- Sentiment: Positive, Negative or Neutral
- Confidence score: 0 to 1
2. Indicators Analysis Agent Output:
- Indicators
- Signals

Output:
- Stock symbol: [e.g., AAPL]
- Action: [Buy, Sell, Hold]
- Reason: [Brief, e.g., “Composite score 0.72, strong MACD and RSI buy”]
"""

decision_agent = Agent(
    model,
    system_prompt=system_prompt
)

if __name__ == '__main__':
    from agents.news_sentiment_agent import SentimentAgent
    from agents.indicators_analyst_agent import DailyTradingIndicators
    import json

    ticker = "NKE"
    indicators_analysis_agent = DailyTradingIndicators(ticker, period="5d", interval="15m")
    sentiment_agent = SentimentAgent(ticker=ticker, news_count=10)

    confident, sentiment = sentiment_agent.get_sentiment()
    indicators_analysis_results = json.dumps(indicators_analysis_agent.run(), indent=2, default=str)
    result = decision_agent.run_sync(
        f"""
Please process trading decisions for the following stocks based on the provided inputs:
- News sentiment: {sentiment} with confidence score: {confident}
- Indicators analysis results: {indicators_analysis_results}
"""
    )
    print(result)
