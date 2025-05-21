import os
import json
from dotenv import load_dotenv

from agents.news_sentiment_agent import NewsSentimentAgent
from agents.indicators_analyst_agent import DailyTradingIndicators
from agents.decision_agent import decision_agent
from agents.portfolio_agent import PortfolioAgent

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if __name__ == '__main__':
    trading_list = ["NVDA", "AMD", "INTC", "NKE", "TSLA", "AAPL"]

    decisions = []
    for ticker in trading_list:
        indicators_analysis_agent = DailyTradingIndicators(ticker, period="30d", interval="1h")
        sentiment_agent = NewsSentimentAgent(ticker=ticker, news_count=10)

        confident, sentiment = sentiment_agent.get_sentiment()
        indicators_analysis_results = json.dumps(indicators_analysis_agent.run(), indent=2, default=str)
        result = decision_agent.run_sync(
            f"""
        Please process trading decisions for the following stocks based on the provided inputs:
        - News sentiment: {sentiment} with confidence score: {confident}
        - Indicators analysis results: {indicators_analysis_results}
        """
        )
        decision = result.output
        decisions.append(decision)
        print(decision)

    portfolio_agent = PortfolioAgent(API_KEY, SECRET_KEY)
    target_allocations = portfolio_agent.process_decisions(decisions)
    portfolio_agent.execute_rebalance(target_allocations)
    # portfolio_agent.cancel_all_pending_orders()