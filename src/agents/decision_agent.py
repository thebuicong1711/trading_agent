import os
import sys
import logfire
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import Any, List, Dict

from utils.api_models import get_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logfire.configure(send_to_logfire='if-token-present')

model = get_model(branch='google')


class Decision(BaseModel):
    symbol: str = Field(description="The stock ticker symbol (e.g., AAPL for Apple Inc.)")
    action: str = Field(description="The recommended action: BUY, SELL, or HOLD")
    strength: str = Field(description="The confidence level of the decision: WEAK, NEUTRAL, or STRONG")
    reason: str = Field(description="The rationale or justification for the recommended action")


system_prompt = """
You are a decision agent for an intraday stock trading system, responsible for generating trading decisions (BUY, SELL, or HOLD) for stocks based on inputs from a sentiment agent and an indicators analysis agent.

**Objective**: For each stock, output a trading decision in the following format:
- `symbol`: The stock ticker (e.g., AAPL).
- `action`: The recommended action (BUY, SELL, or HOLD).
- `strength`: The confidence level (WEAK, NEUTRAL, or STRONG).
- `reason`: A concise explanation for the decision, integrating sentiment and technical signals.

**Inputs**:
1. **Sentiment Agent Output**:
   - `sentiment`: Positive, Negative, or Neutral.
   - `confidence_score`: A value between 0 and 1 indicating sentiment strength.
   - Example: `(0.9999995231628418, 'neutral')`

2. **Indicators Analysis Agent Output**:
   - `info`: Contains stock details (e.g., ticker, timestamp, price data, moving averages, technical indicators, pivot points, price position).
   - `signals`: Technical indicator signals (e.g., RSI, MACD, Bollinger Bands, Stochastic, Moving Averages, Volume) with:
     - `signal`: BUY, SELL, or HOLD.
     - `strength`: WEAK, NEUTRAL, or STRONG.
     - `explanation`: Reasoning for the signal.
     - `overall`: A summary signal with an explanation.
   - Example: Includes ticker (e.g., AAPL), price data (e.g., current price: 202.06), technical indicators (e.g., RSI_14: 28.37), and signals (e.g., RSI: BUY, strength: STRONG, explanation: "RSI(14) is oversold below 30").

**Instructions**:
- Analyze the sentiment and technical signals to make an informed decision.
- Prioritize strong signals and high-confidence sentiment scores.
- Provide a clear, concise reason for the decision, referencing both sentiment and technical indicators.
- Ensure the output aligns with the provided Decision model format.
"""

decision_agent = Agent(
    model,
    system_prompt=system_prompt,
    output_type=Decision,
)

# if __name__ == '__main__':
#     from agents.news_sentiment_agent import NewsSentimentAgent
#     from agents.indicators_analyst_agent import DailyTradingIndicators
#     import json
#
#     ticker = "AAPL"
#     indicators_analysis_agent = DailyTradingIndicators(ticker, period="5d", interval="15m")
#     sentiment_agent = NewsSentimentAgent(ticker=ticker, news_count=10)
#
#     confident, sentiment = sentiment_agent.get_sentiment()
#     indicators_analysis_results = json.dumps(indicators_analysis_agent.run(), indent=2, default=str)
#     result = decision_agent.run_sync(
#         f"""
# Please process trading decisions for the following stocks based on the provided inputs:
# - News sentiment: {sentiment} with confidence score: {confident}
# - Indicators analysis results: {indicators_analysis_results}
# """
#     )
#     print(result.output)
