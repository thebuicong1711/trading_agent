import os
import time
from datetime import datetime

from dotenv import load_dotenv
from typing import Dict, List

from alpaca.trading import GetOrdersRequest, OrderStatus, QueryOrderStatus, GetPortfolioHistoryRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class PortfolioAgent:
    def __init__(self,
                 api_key: str,
                 secret_key: str,
                 paper: bool = True,
                 max_position_size: float = 0.2,  # Maximum allocation to a single position (20%)
                 cash_reserve: float = 0.1,  # Minimum cash reserve (10%)
                 rebalance_threshold: float = 0.05,  # Rebalance when allocations drift by 5%
                 ):
        # Initialize API client
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        # Portfolio parameters
        self.max_position_size = max_position_size
        self.cash_reserve = cash_reserve
        self.rebalance_threshold = rebalance_threshold
        self.paper = paper

        # Current portfolio state
        self.portfolio = None
        self.cash = 0
        self.equity = 0

        # Update initial portfolio state
        self.update_portfolio_state()

    def update_portfolio_state(self):
        account = self.trading_client.get_account()
        self.cash = float(account.cash)
        self.equity = float(account.equity)

        positions = self.trading_client.get_all_positions()
        self.portfolio = {
            p.symbol: {
                'qty': int(p.qty),
                'market_value': float(p.market_value),
                'current_price': float(p.current_price),
                'allocation': round(float(p.market_value) / self.equity, 4)
            } for p in positions
        }

        print(f"Portfolio updated - Equity: ${self.equity:.2f}, Cash: ${self.cash:.2f}")
        return True

    def get_pending_orders(self) -> Dict[str, List[Dict]]:
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
            )
            orders = self.trading_client.get_orders(req)
            pending_orders = {}
            for order in orders:
                symbol = order.symbol

                if symbol not in pending_orders:
                    pending_orders[symbol] = []

                order_info = {
                    'id': order.id,
                    'side': order.side.value,
                    'qty': float(order.qty) if order.qty else None,
                    'notional': float(order.notional) if order.notional else None,
                    'status': order.status.value,
                    'created_at': order.created_at.isoformat(),
                    'type': order.type.value
                }
                pending_orders[symbol].append(order_info)

            print(f"Found {len(orders)} pending orders for {len(pending_orders)} symbols")
            return pending_orders
        except Exception as e:
            print(f"Error retrieving pending orders: {str(e)}")
            return {}

    def process_signals(self, sentiment_signals: Dict[str, str], indicator_signals: Dict[str, str]) -> Dict[str, float]:
        target_allocations = {}
        universe = set(sentiment_signals.keys()) | set(indicator_signals.keys())

        scores = {}
        for symbol in universe:
            sentiment = sentiment_signals.get(symbol, "NEUTRAL")
            indicator = indicator_signals.get(symbol, "HOLD")
            sentiment_score = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}.get(sentiment, 0)
            indicator_score = {"BUY": 1, "HOLD": 0, "SELL": -1}.get(indicator, 0)
            combined_score = (sentiment_score + indicator_score) / 2
            scores[symbol] = combined_score

        long_candidates = {s: score for s, score in scores.items() if score > 0}
        total_score = sum(long_candidates.values())
        available_allocation = 1.0 - self.cash_reserve

        # Calculate preliminary allocations
        if total_score > 0:
            for symbol, score in long_candidates.items():
                allocation = (score / total_score) * available_allocation
                target_allocations[symbol] = round(min(allocation, self.max_position_size), 2)

        # Normalize allocations if they exceed the available allocation
        total_allocation = sum(target_allocations.values())
        if total_allocation > available_allocation:
            for symbol in target_allocations:
                target_allocations[symbol] *= available_allocation / total_allocation
                target_allocations[symbol] = round(target_allocations[symbol], 2)

        return target_allocations

    def execute_rebalance(self, target_allocations: Dict[str, float], dry_run: bool = False) -> List[Dict]:
        """
        Rebalance portfolio according to target allocations.

        Args:
            target_allocations: Dictionary mapping symbols to target allocation percentages
            dry_run: If True, don't execute trades but return what would be done

        Returns:
            List of executed or simulated orders
        """
        self.update_portfolio_state()

        # Get pending orders to avoid duplicate orders
        pending_orders = self.get_pending_orders()

        orders = []

        target_values = {symbol: allocation * self.equity  # check it should be cash or equity
                         for symbol, allocation in target_allocations.items()}

        # Determine what to sell (positions not in target or overallocated)
        for symbol, details in self.portfolio.items():
            current_value = details['market_value']
            target_value = target_values.get(symbol, 0)

            if symbol in pending_orders:
                print(f"Skipping {symbol} - pending orders exist")
                continue

            if symbol not in target_allocations or current_value > target_value + (
                    target_value * self.rebalance_threshold):  # sell by checking rebalance_threshold
                value_to_sell = current_value - target_value if symbol in target_allocations else current_value

                use_qty_order = False
                order_qty = None

                if symbol in target_allocations and target_value > 0:
                    # We're reducing position, not closing it completely
                    pass
                else:
                    use_qty_order = True
                    order_qty = details['qty']

                if value_to_sell > 0:
                    order_detail = {
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'day'
                    }

                    if use_qty_order:
                        order_detail['qty'] = order_qty
                    else:
                        order_detail['notional'] = value_to_sell

                    if not dry_run:
                        try:
                            if use_qty_order:
                                market_order_data = MarketOrderRequest(
                                    symbol=symbol,
                                    qty=order_qty,
                                    side=OrderSide.SELL,
                                    time_in_force=TimeInForce.DAY
                                )
                            else:
                                adjusted_value = value_to_sell * 0.995  # Use 99.5% of the calculated value
                                market_order_data = MarketOrderRequest(
                                    symbol=symbol,
                                    notional=adjusted_value,
                                    side=OrderSide.SELL,
                                    time_in_force=TimeInForce.DAY
                                )

                            # Submit order
                            order = self.trading_client.submit_order(order_data=market_order_data)
                            order_detail['id'] = order.id
                            order_detail['status'] = 'submitted'
                        except Exception as e:
                            print(f"Error selling {symbol}: {str(e)}")
                            order_detail['status'] = 'error'
                            order_detail['error'] = str(e)
                    else:
                        order_detail['status'] = 'simulated'

                    orders.append(order_detail)
                    print(
                        f"{'Simulated' if dry_run else 'Executed'} SELL order: ${value_to_sell:.2f} worth of {symbol}")

        if not dry_run and any(o['side'] == 'sell' for o in orders):
            time.sleep(2)
            self.update_portfolio_state()
            pending_orders = self.get_pending_orders()

        # Determine what to buy
        for symbol, target_allocation in target_allocations.items():
            # Skip if there are already pending orders for this symbol
            if symbol in pending_orders:
                print(f"Skipping {symbol} - pending orders exist")
                continue

            current_value = self.portfolio.get(symbol, {}).get('market_value', 0)
            target_value = target_allocation * self.equity

            if current_value < target_value - (target_value * self.rebalance_threshold):
                value_to_buy = target_value - current_value
                value_to_buy = min(value_to_buy, self.cash * 0.99)
                value_to_buy = round(value_to_buy, 2)

                if value_to_buy > 0:
                    order_detail = {
                        'symbol': symbol,
                        'notional': value_to_buy,
                        'side': 'buy',
                        'type': 'market',
                        'time_in_force': 'day'
                    }

                    if not dry_run:
                        try:
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                notional=value_to_buy,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.DAY
                            )
                            order = self.trading_client.submit_order(order_data=market_order_data)
                            order_detail['id'] = order.id
                            order_detail['status'] = 'submitted'
                            self.cash -= value_to_buy
                        except Exception as e:
                            print(f"Error buying {symbol}: {str(e)}")
                            order_detail['status'] = 'error'
                            order_detail['error'] = str(e)
                        else:
                            order_detail['status'] = 'simulated'
                        orders.append(order_detail)
                        print(
                            f"{'Simulated' if dry_run else 'Executed'} BUY order: ${value_to_buy:.2f} worth of {symbol}")

        if not dry_run:
            time.sleep(3)
            self.update_portfolio_state()

        return orders

    def cancel_all_pending_orders(self) -> int:
        try:
            pending_orders = self.get_pending_orders()
            cancelled_count = 0

            for symbol, orders in pending_orders.items():
                for order in orders:
                    order_id = order['id']
                    try:
                        self.trading_client.cancel_order_by_id(order_id)
                        print(f"Cancelled order {order_id} for {symbol}")
                        cancelled_count += 1
                    except Exception as e:
                        print(f"Error cancelling order {order_id}: {str(e)}")

            print(f"Cancelled {cancelled_count} pending orders")
            return cancelled_count
        except Exception as e:
            print(f"Error cancelling pending orders: {str(e)}")
            return 0

    def get_portfolio_analytics(self) -> Dict:
        self.update_portfolio_state()

        analytics = {
            "total_equity": self.equity,
            "cash": self.cash,
            "cash_allocation": self.cash / self.equity if self.equity > 0 else 0,
            "positions": len(self.portfolio),
            "allocations": {symbol: details['allocation'] for symbol, details in self.portfolio.items()},
            "timestamp": datetime.now().isoformat(),
            "equity_history": []
        }

        try:
            port_filter = GetPortfolioHistoryRequest(extended_hours=True, period='1M', timeframe='1D')
            history = self.trading_client.get_portfolio_history(history_filter=port_filter)

            if hasattr(history, 'equity') and history.equity:
                analytics["equity_history"] = history.equity
                if len(history.equity) > 1:
                    first_value = history.equity[0]
                    last_value = history.equity[-1]
                    if first_value > 0:
                        analytics["profit_loss_pct"] = (last_value / first_value - 1) * 100
                    else:
                        analytics["profit_loss_pct"] = 0
                else:
                    analytics["profit_loss_pct"] = 0
        except Exception as e:
            print(f"Could not get portfolio history: {str(e)}")

        # Add information about pending orders
        pending_orders = self.get_pending_orders()
        analytics["pending_orders"] = {
            "count": sum(len(orders) for orders in pending_orders.values()),
            "symbols": list(pending_orders.keys())
        }

        return analytics

    def liquidate_all_positions(self, dry_run: bool = False) -> List[Dict]:
        print(f"Liquidating all positions. Dry run: {dry_run}")
        self.update_portfolio_state()

        # Cancel any pending orders first if not dry run
        if not dry_run:
            self.cancel_all_pending_orders()

        orders = []
        for symbol, details in self.portfolio.items():
            # Use the notional value (market value) of the position
            notional_value = details['market_value']

            # Use qty-based order for liquidation to avoid precision issues
            order_qty = details['qty']

            order_detail = {
                'symbol': symbol,
                'qty': order_qty,
                'side': 'sell',
                'type': 'market',
                'time_in_force': 'day'
            }

            if not dry_run:
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=order_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )

                    order = self.trading_client.submit_order(order_data=market_order_data)
                    order_detail['id'] = order.id
                    order_detail['status'] = 'submitted'
                except Exception as e:
                    print(f"Error liquidating {symbol}: {str(e)}")
                    order_detail['status'] = 'error'
                    order_detail['error'] = str(e)
            else:
                order_detail['status'] = 'simulated'

            orders.append(order_detail)
            print(
                f"{'Simulated' if dry_run else 'Executed'} LIQUIDATION of ${notional_value:.2f} worth of {symbol}")

        if not dry_run and orders:
            time.sleep(3)
            self.update_portfolio_state()

        return orders

    def get_available_assets(self) -> List[str]:
        """
        Get a list of available tradable assets from Alpaca.

        Returns:
            List of symbols that are available for trading
        """
        try:
            request_params = GetAssetsRequest(status=AssetStatus.ACTIVE)
            assets = self.trading_client.get_all_assets(request_params)

            # Filter for tradable US equities
            tradable_symbols = [asset.symbol for asset in assets
                                if asset.tradable and asset.exchange != "CRYPTO"]

            print(f"Found {len(tradable_symbols)} tradable assets")
            return tradable_symbols
        except Exception as e:
            print(f"Error getting available assets: {str(e)}")
            return []


if __name__ == '__main__':
    load_dotenv()

    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

    portfolio_agent = PortfolioAgent(API_KEY, SECRET_KEY)

    sentiment_signals = {
        "AAPL": "POSITIVE",
        "MSFT": "NEUTRAL",
        "AMZN": "POSITIVE",
        "GOOGL": "NEGATIVE",
        "META": "NEUTRAL"
    }

    indicator_signals = {
        "AAPL": "BUY",
        "MSFT": "BUY",
        "AMZN": "HOLD",
        "GOOGL": "SELL",
        "META": "HOLD",
        "NVDA": "BUY"  # Note: This is in indicator signals but not sentiment
    }
    target_allocations_ = portfolio_agent.process_signals(sentiment_signals, indicator_signals)
    # portfolio_agent.execute_rebalance(target_allocations_)
    print(portfolio_agent.get_portfolio_analytics())