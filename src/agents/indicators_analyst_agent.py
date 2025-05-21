import json
import os.path

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.ticker as m_ticker


class DailyTradingIndicators:
    """Class to calculate technical indicators optimized for daily trading."""

    def __init__(self, ticker_symbol, period='5d', interval='15m'):
        """
        Initialize with a ticker symbol, period, and interval.

        Parameters:
            ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
            period (str): Period of historical data (default: '5d')
                        Options: '1d', '5d', '1mo', '3mo'
            interval (str): Data interval (default: '15m')
                          Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m'
        """
        self.ticker_symbol = ticker_symbol
        self.period = period
        self.interval = interval
        self.hist = None
        self.indicators = {}
        self.signals = {}

    def fetch_data(self):
        """Fetch historical data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(self.ticker_symbol)
            self.hist = ticker.history(period=self.period, interval=self.interval)
            self.hist = self.hist.dropna()
            if self.hist.empty:
                return {"error": f"No data found for ticker {self.ticker_symbol}"}
            return None
        except Exception as e:
            return {"error": f"Failed to fetch data: {str(e)}"}

    def calculate_price_data(self):
        """Calculate basic price metrics."""
        if self.hist.empty:
            return {}
        close = self.hist['Close']
        volume = self.hist['Volume']
        return {
            "current_price": round(close.iloc[-1], 2),
            "previous_close": round(close.iloc[-2], 2),
            "change": round(close.iloc[-1] - close.iloc[-2], 2),
            "change_percent": round(((close.iloc[-1] / close.iloc[-2]) - 1) * 100, 2),
            "day_high": round(self.hist['High'].iloc[-1], 2),
            "day_low": round(self.hist['Low'].iloc[-1], 2),
            "session_volume": int(volume.iloc[-1]),
            "vwap": round(sum(close * volume) / sum(volume), 2),
        }

    def calculate_moving_averages(self):
        """Calculate short-term moving averages (EMA and SMA)."""
        if self.hist.empty:
            return {}
        close = self.hist['Close']
        return {
            "ema_9": round(close.ewm(span=9, adjust=False).mean().iloc[-1], 2),
            "ema_20": round(close.ewm(span=20, adjust=False).mean().iloc[-1], 2),
            "ema_50": round(close.ewm(span=50, adjust=False).mean().iloc[-1], 2),
            "sma_5": round(close.rolling(window=5).mean().iloc[-1], 2),
            "sma_8": round(close.rolling(window=8).mean().iloc[-1], 2),
            "sma_13": round(close.rolling(window=13).mean().iloc[-1], 2),
        }

    def calculate_rsi(self):
        """Calculate RSI for 7 and 14 periods."""
        if self.hist.empty:
            return {}
        delta = self.hist['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain_14 = gain.rolling(window=14).mean()
        avg_gain_7 = gain.rolling(window=7).mean()
        avg_loss_14 = loss.rolling(window=14).mean()
        avg_loss_7 = loss.rolling(window=7).mean()

        rs_14 = avg_gain_14 / (avg_loss_14 + 1e-10)
        rs_7 = avg_gain_7 / (avg_loss_7 + 1e-10)
        rsi_14 = 100 - (100 / (1 + rs_14))
        rsi_7 = 100 - (100 / (1 + rs_7))

        return {
            "rsi_14": round(rsi_14.iloc[-1], 2) if not pd.isna(rsi_14.iloc[-1]) else None,
            "rsi_7": round(rsi_7.iloc[-1], 2) if not pd.isna(rsi_7.iloc[-1]) else None
        }

    def calculate_macd(self):
        """Calculate MACD and Fast MACD."""
        if self.hist.empty:
            return {}
        close = self.hist['Close']

        # Standard MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # Fast MACD
        fast_exp1 = close.ewm(span=6, adjust=False).mean()
        fast_exp2 = close.ewm(span=19, adjust=False).mean()
        fast_macd = fast_exp1 - fast_exp2
        fast_signal = fast_macd.ewm(span=6, adjust=False).mean()

        return {
            "macd": round(macd.iloc[-1], 2) if not pd.isna(macd.iloc[-1]) else None,
            "macd_signal": round(signal.iloc[-1], 2) if not pd.isna(signal.iloc[-1]) else None,
            "macd_histogram": round(macd.iloc[-1] - signal.iloc[-1], 2) if not pd.isna(macd.iloc[-1]) and not pd.isna(
                signal.iloc[-1]) else None,
            "fast_macd": round(fast_macd.iloc[-1], 2) if not pd.isna(fast_macd.iloc[-1]) else None,
            "fast_macd_signal": round(fast_signal.iloc[-1], 2) if not pd.isna(fast_signal.iloc[-1]) else None,
            "fast_macd_histogram": round(fast_macd.iloc[-1] - fast_signal.iloc[-1], 2) if not pd.isna(
                fast_macd.iloc[-1]) and not pd.isna(fast_signal.iloc[-1]) else None,
        }

    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands for 20 and 10 periods."""
        if self.hist.empty:
            return {}
        close = self.hist['Close']

        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)

        sma_10 = close.rolling(window=10).mean()
        std_10 = close.rolling(window=10).std()
        upper_band_10 = sma_10 + (std_10 * 2)
        lower_band_10 = sma_10 - (std_10 * 2)

        return {
            "bollinger_upper": round(upper_band.iloc[-1], 2) if not pd.isna(upper_band.iloc[-1]) else None,
            "bollinger_middle": round(sma_20.iloc[-1], 2) if not pd.isna(sma_20.iloc[-1]) else None,
            "bollinger_lower": round(lower_band.iloc[-1], 2) if not pd.isna(lower_band.iloc[-1]) else None,
            "bollinger_upper_10": round(upper_band_10.iloc[-1], 2) if not pd.isna(upper_band_10.iloc[-1]) else None,
            "bollinger_middle_10": round(sma_10.iloc[-1], 2) if not pd.isna(sma_10.iloc[-1]) else None,
            "bollinger_lower_10": round(lower_band_10.iloc[-1], 2) if not pd.isna(lower_band_10.iloc[-1]) else None,
        }

    def calculate_atr(self):
        """Calculate Average True Range for 14 and 7 periods."""
        if self.hist.empty:
            return {}
        high_low = self.hist['High'] - self.hist['Low']
        high_close = abs(self.hist['High'] - self.hist['Close'].shift())
        low_close = abs(self.hist['Low'] - self.hist['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return {
            "atr_14": round(true_range.rolling(window=14).mean().iloc[-1], 2) if not pd.isna(
                true_range.rolling(window=14).mean().iloc[-1]) else None,
            "atr_7": round(true_range.rolling(window=7).mean().iloc[-1], 2) if not pd.isna(
                true_range.rolling(window=7).mean().iloc[-1]) else None,
        }

    def calculate_stochastic_oscillator(self):
        """Calculate Stochastic Oscillator for 14 and 5 periods."""
        if self.hist.empty:
            return {}
        low_14 = self.hist['Low'].rolling(window=14).min()
        high_14 = self.hist['High'].rolling(window=14).max()
        low_5 = self.hist['Low'].rolling(window=5).min()
        high_5 = self.hist['High'].rolling(window=5).max()

        k_percent_14 = 100 * ((self.hist['Close'] - low_14) / (high_14 - low_14 + 1e-10))
        d_percent_14 = k_percent_14.rolling(window=3).mean()
        k_percent_5 = 100 * ((self.hist['Close'] - low_5) / (high_5 - low_5 + 1e-10))
        d_percent_5 = k_percent_5.rolling(window=3).mean()

        return {
            "stochastic_k_14": round(k_percent_14.iloc[-1], 2) if not pd.isna(k_percent_14.iloc[-1]) else None,
            "stochastic_d_14": round(d_percent_14.iloc[-1], 2) if not pd.isna(d_percent_14.iloc[-1]) else None,
            "stochastic_k_5": round(k_percent_5.iloc[-1], 2) if not pd.isna(k_percent_5.iloc[-1]) else None,
            "stochastic_d_5": round(d_percent_5.iloc[-1], 2) if not pd.isna(d_percent_5.iloc[-1]) else None,
        }

    def calculate_mfi(self):
        """Calculate Money Flow Index (MFI)."""
        if self.hist.empty:
            return {}
        typical_price = (self.hist['High'] + self.hist['Low'] + self.hist['Close']) / 3
        raw_money_flow = typical_price * self.hist['Volume']

        money_flow_positive = pd.Series(0, index=self.hist.index, dtype='float64')
        money_flow_negative = pd.Series(0, index=self.hist.index, dtype='float64')

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                money_flow_positive.iloc[i] = raw_money_flow.iloc[i]
            else:
                money_flow_negative.iloc[i] = raw_money_flow.iloc[i]

        mfi_period = 14
        positive_mf = money_flow_positive.rolling(window=mfi_period).sum()
        negative_mf = money_flow_negative.rolling(window=mfi_period).sum()

        money_ratio = positive_mf / (negative_mf + 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))

        return {
            "mfi": round(mfi.iloc[-1], 2) if not pd.isna(mfi.iloc[-1]) else None
        }

    def calculate_pivot_points(self):
        """Calculate pivot points and support/resistance levels."""
        if self.hist.empty or len(self.hist) < 2:
            return {"note": "Insufficient data for pivot points"}

        try:
            prev_data = self.hist.iloc[-2]
            prev_high = prev_data["High"]
            prev_low = prev_data["Low"]
            prev_close = prev_data["Close"]

            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)

            return {
                "pivot": round(pivot, 2),
                "r1": round(r1, 2),
                "r2": round(r2, 2),
                "r3": round(r3, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2),
                "s3": round(s3, 2),
            }
        except Exception as e:
            return {"note": f"Error calculating pivot points: {e}"}

    def calculate_imi(self):
        """Calculate Intraday Momentum Index (IMI)."""
        if self.hist.empty:
            return {}
        gains = np.where(self.hist['Close'] > self.hist['Open'], self.hist['Close'] - self.hist['Open'], 0)
        losses = np.where(self.hist['Open'] > self.hist['Close'], self.hist['Open'] - self.hist['Close'], 0)

        sum_gains = pd.Series(gains).rolling(window=14).sum()
        sum_losses = pd.Series(losses).rolling(window=14).sum()

        imi = 100 * (sum_gains / (sum_gains + sum_losses + 1e-10))

        return {
            "imi": round(imi.iloc[-1], 2) if not pd.isna(imi.iloc[-1]) else None
        }

    def calculate_volume_metrics(self):
        """Calculate relative volume and volume spike."""
        if self.hist.empty:
            return {}
        volume_5_avg = self.hist['Volume'].rolling(window=5).mean()
        relative_volume = (self.hist['Volume'].iloc[-1] / volume_5_avg.iloc[-1]
                           if not pd.isna(volume_5_avg.iloc[-1]) and volume_5_avg.iloc[-1] != 0 else 0)

        return {
            "relative_volume": round(relative_volume, 2),
            "volume_spike": relative_volume > 2.0
        }

    def calculate_keltner_channels(self):
        """Calculate Keltner Channels."""
        if self.hist.empty:
            return {}
        typical_price = (self.hist['High'] + self.hist['Low'] + self.hist['Close']) / 3
        ema_20 = typical_price.ewm(span=20, adjust=False).mean()

        high_low = self.hist['High'] - self.hist['Low']
        high_close = abs(self.hist['High'] - self.hist['Close'].shift())
        low_close = abs(self.hist['Low'] - self.hist['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_10 = true_range.rolling(window=10).mean()

        keltner_upper = ema_20 + (atr_10 * 1.5)
        keltner_lower = ema_20 - (atr_10 * 1.5)

        return {
            "keltner_upper": round(keltner_upper.iloc[-1], 2) if not pd.isna(keltner_upper.iloc[-1]) else None,
            "keltner_middle": round(ema_20.iloc[-1], 2) if not pd.isna(ema_20.iloc[-1]) else None,
            "keltner_lower": round(keltner_lower.iloc[-1], 2) if not pd.isna(keltner_lower.iloc[-1]) else None,
        }

    def calculate_price_position(self, moving_averages, bollinger_bands, keltner_channels):
        """Calculate price position relative to indicators."""
        if self.hist.empty:
            return {}
        close = self.hist['Close'].iloc[-1]

        position_in_bbands = ("upper" if close > bollinger_bands.get("bollinger_upper", float('inf'))
                              else "lower" if close < bollinger_bands.get("bollinger_lower", -float('inf'))
        else "middle")
        position_in_keltner = ("upper" if close > keltner_channels.get("keltner_upper", float('inf'))
                               else "lower" if close < keltner_channels.get("keltner_lower", -float('inf'))
        else "middle")

        return {
            "above_ema9": close > moving_averages.get("ema_9", float('inf')),
            "above_ema20": close > moving_averages.get("ema_20", float('inf')),
            "above_sma5": close > moving_averages.get("sma_5", float('inf')),
            "position_in_bbands": position_in_bbands,
            "position_in_keltner": position_in_keltner,
        }

    def compute_indicators(self):
        """Compute all trading indicators and return as a dictionary."""
        # Fetch data
        error = self.fetch_data()
        if error:
            return error

        # Calculate indicators
        price_data = self.calculate_price_data()
        moving_averages = self.calculate_moving_averages()
        rsi = self.calculate_rsi()
        macd = self.calculate_macd()
        bollinger_bands = self.calculate_bollinger_bands()
        atr = self.calculate_atr()
        stochastic = self.calculate_stochastic_oscillator()
        mfi = self.calculate_mfi()
        pivot_points = self.calculate_pivot_points()
        imi = self.calculate_imi()
        volume_metrics = self.calculate_volume_metrics()
        keltner_channels = self.calculate_keltner_channels()
        price_position = self.calculate_price_position(moving_averages, bollinger_bands, keltner_channels)

        # Combine technical indicators
        technical_indicators = {
            **rsi,
            **macd,
            **bollinger_bands,
            **atr,
            **stochastic,
            **mfi,
            **imi,
            **volume_metrics,
            **keltner_channels,
        }

        # Combine all indicators
        self.indicators = {
            "ticker": self.ticker_symbol,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "price_data": price_data,
            "moving_averages": moving_averages,
            "technical_indicators": technical_indicators,
            "pivot_points": pivot_points,
            "price_position": price_position,
        }

        return self.indicators

    def get_trading_signals(self, rsi_thresholds=(30, 70), stoch_thresholds=(20, 80),
                            bb_squeeze_threshold=0.85, pivot_proximity=0.2):
        """
        Generate trading signals based on technical indicators

        Parameters:
            rsi_thresholds (tuple): RSI oversold/overbought thresholds (default: (30, 70)).
            stoch_thresholds (tuple): Stochastic oversold/overbought thresholds (default: (20, 80)).
            bb_squeeze_threshold (float): Bollinger Band squeeze threshold (default: 0.85).
            pivot_proximity (float): Proximity threshold for pivot points (default: 0.2).

         Returns:
            dict: Trading signals and their explanations
        """

        if not self.indicators:
            self.compute_indicators()

        tech = self.indicators["technical_indicators"]
        price = self.indicators["price_data"]["current_price"]
        rsi_low, rsi_high = rsi_thresholds
        stoch_low, stoch_high = stoch_thresholds

        # RSI signals
        if tech["rsi_14"] is not None:
            if tech["rsi_14"] < rsi_low:
                self.signals["rsi"] = {"signal": "BUY", "strength": "STRONG",
                                       "explanation": f"RSI(14) is oversold below {rsi_low}"}
            elif tech["rsi_14"] > rsi_high:
                self.signals["rsi"] = {"signal": "SELL", "strength": "STRONG",
                                       "explanation": f"RSI(14) is overbought above {rsi_high}"}
            elif (tech["rsi_14"] < (rsi_low + 15)) and (tech["rsi_14"] >= rsi_low):
                self.signals["rsi"] = {"signal": "BUY", "strength": "WEAK",
                                       "explanation": "RSI(14) is nearing oversold territory"}
            elif (tech["rsi_14"] > (rsi_high - 15)) and (tech["rsi_14"] <= rsi_high):
                self.signals["rsi"] = {"signal": "SELL", "strength": "WEAK",
                                       "explanation": "RSI(14) is nearing overbought territory"}
            else:
                self.signals["rsi"] = {"signal": "HOLD", "strength": "NEUTRAL",
                                       "explanation": "RSI(14) is in hold zone"}

            # MACD signals
            if tech["macd"] is not None and tech["macd_signal"] is not None and len(self.hist) > 26:
                macd_diff = tech["macd"] - tech["macd_signal"]

                # Calculate previous MACD and signal
                exp1 = self.hist['Close'].ewm(span=12, adjust=False).mean()
                exp2 = self.hist['Close'].ewm(span=26, adjust=False).mean()
                macd_series = exp1 - exp2
                signal_series = macd_series.ewm(span=9, adjust=False).mean()

                if len(self.hist) > 1:
                    prev_diff = macd_series.iloc[-2] - signal_series.iloc[-2]

                    if (macd_diff > 0) and (prev_diff < 0):
                        self.signals["macd"] = {"signal": "BUY", "strength": "STRONG",
                                                "explanation": "MACD bullish crossover (signal line)"}
                    elif (macd_diff < 0) and (prev_diff > 0):
                        self.signals["macd"] = {"signal": "SELL", "strength": "STRONG",
                                                "explanation": "MACD bearish crossover (signal line)"}
                    elif (macd_diff > 0) and (macd_diff > prev_diff):
                        self.signals["macd"] = {"signal": "BUY", "strength": "WEAK",
                                                "explanation": "MACD momentum increasing above signal line"}
                    elif (macd_diff < 0) and (macd_diff < prev_diff):
                        self.signals["macd"] = {"signal": "SELL", "strength": "WEAK",
                                                "explanation": "MACD momentum decreasing below signal line"}
                    else:
                        self.signals["macd"] = {"signal": "HOLD", "strength": "NEUTRAL",
                                                "explanation": "No significant MACD signal"}

            # Bollinger Bands signals
            if tech["bollinger_lower"] is not None and tech["bollinger_upper"] is not None and len(self.hist) > 20:
                if price < tech["bollinger_lower"]:
                    self.signals["bbands"] = {"signal": "BUY", "strength": "STRONG",
                                              "explanation": "Price below lower Bollinger Band"}
                elif price > tech["bollinger_upper"]:
                    self.signals["bbands"] = {"signal": "SELL", "strength": "STRONG",
                                              "explanation": "Price above upper Bollinger Band"}
                else:
                    current_width = tech["bollinger_upper"] - tech["bollinger_lower"]
                    past_width = self.hist['Close'].rolling(window=20).std().iloc[-10] * 4

                    if not pd.isna(past_width) and (current_width < past_width * bb_squeeze_threshold):
                        self.signals["bbands"] = {"signal": "HOLD", "strength": "STRONG",
                                                  "explanation": "Bollinger Band squeeze - potential breakout"}
                    else:
                        self.signals["bbands"] = {"signal": "HOLD", "strength": "NEUTRAL",
                                                  "explanation": "Price within Bollinger Bands"}

            # Stochastic signals
            if tech["stochastic_k_5"] is not None and tech["stochastic_d_5"] is not None:
                k5 = tech["stochastic_k_5"]
                d5 = tech["stochastic_d_5"]

                if k5 < stoch_low and d5 < stoch_low:
                    self.signals["stoch"] = {"signal": "BUY", "strength": "STRONG",
                                             "explanation": f"Stochastic oversold (below {stoch_low})"}
                elif k5 > stoch_high and d5 > stoch_high:
                    self.signals["stoch"] = {"signal": "SELL", "strength": "STRONG",
                                             "explanation": f"Stochastic overbought (above {stoch_high})"}
                elif (k5 > d5) and (k5 <= stoch_low) and (d5 <= stoch_low):
                    self.signals["stoch"] = {"signal": "BUY", "strength": "STRONG",
                                             "explanation": "Stochastic %K crossed above %D in oversold territory"}
                elif (k5 < d5) and (k5 >= stoch_high) and (d5 >= stoch_high):
                    self.signals["stoch"] = {"signal": "SELL", "strength": "STRONG",
                                             "explanation": "Stochastic %K crossed below %D in overbought territory"}
                else:
                    self.signals["stoch"] = {"signal": "HOLD", "strength": "NEUTRAL",
                                             "explanation": "No significant stochastic signal"}

            # Moving Average signals
            ema9 = self.indicators["moving_averages"].get("ema_9")
            ema20 = self.indicators["moving_averages"].get("ema_20")
            if ema9 is not None and ema20 is not None:
                prev_close = self.indicators["price_data"]["previous_close"]

                if (ema9 > ema20) and (prev_close <= ema9) and (price > ema9):
                    self.signals["ma"] = {"signal": "BUY", "strength": "STRONG",
                                          "explanation": "Price crossed above EMA9 with EMA9 above EMA20 (bullish)"}
                elif (ema9 < ema20) and (prev_close >= ema9) and (price < ema9):
                    self.signals["ma"] = {"signal": "SELL", "strength": "STRONG",
                                          "explanation": "Price crossed below EMA9 with EMA9 below EMA20 (bearish)"}
                elif ema9 > ema20:
                    self.signals["ma"] = {"signal": "BUY", "strength": "WEAK",
                                          "explanation": "EMA9 above EMA20 (bullish trend)"}
                elif ema9 < ema20:
                    self.signals["ma"] = {"signal": "SELL", "strength": "WEAK",
                                          "explanation": "EMA9 below EMA20 (bearish trend)"}
                else:
                    self.signals["ma"] = {"signal": "HOLD", "strength": "NEUTRAL",
                                          "explanation": "No clear moving average signal"}

            # Volume signals
            rel_volume = tech["relative_volume"]
            if rel_volume > 2.0:
                if price > self.indicators["price_data"]["previous_close"]:
                    self.signals["volume"] = {"signal": "BUY", "strength": "STRONG",
                                              "explanation": f"High volume (+{round(rel_volume, 1)}x) on up move"}
                elif price < self.indicators["price_data"]["previous_close"]:
                    self.signals["volume"] = {"signal": "SELL", "strength": "STRONG",
                                              "explanation": f"High volume (+{round(rel_volume, 1)}x) on down move"}
            elif rel_volume > 1.5:
                if price > self.indicators["price_data"]["previous_close"]:
                    self.signals["volume"] = {"signal": "BUY", "strength": "WEAK",
                                              "explanation": f"Above average volume (+{round(rel_volume, 1)}x) on up move"}
                elif price < self.indicators["price_data"]["previous_close"]:
                    self.signals["volume"] = {"signal": "SELL", "strength": "WEAK",
                                              "explanation": f"Above average volume (+{round(rel_volume, 1)}x) on down move"}

            # Support/Resistance signals based on pivot points
            pivot_data = self.indicators.get("pivot_points", {})
            if pivot_data and "pivot" in pivot_data:
                pivot = pivot_data["pivot"]
                r1 = pivot_data.get("r1")
                s1 = pivot_data.get("s1")
                s2 = pivot_data.get("s2")
                r2 = pivot_data.get("r2")

                # Near support
                if (price < pivot) and (price > s1) and (price < (pivot - (pivot - s1) * pivot_proximity)):
                    self.signals["pivot"] = {"signal": "BUY", "strength": "WEAK",
                                             "explanation": "Price near S1 support level"}
                elif s2 is not None and (price < s1) and (price > s2) and (price < (s1 - (s1 - s2) * pivot_proximity)):
                    self.signals["pivot"] = {"signal": "BUY", "strength": "STRONG",
                                             "explanation": "Price near S2 support level"}

                # Near resistance
                elif (price > pivot) and (price < r1) and (price > (pivot + (r1 - pivot) * (1 - pivot_proximity))):
                    self.signals["pivot"] = {"signal": "SELL", "strength": "WEAK",
                                             "explanation": "Price near R1 resistance level"}
                elif r2 is not None and (price > r1) and (price < r2) and (
                        price > (r1 + (r2 - r1) * (1 - pivot_proximity))):
                    self.signals["pivot"] = {"signal": "SELL", "strength": "STRONG",
                                             "explanation": "Price near R2 resistance level"}

            # Generate overall signal based on weighted votes
            buy_signals = sum(1 for signal in self.signals.values() if signal["signal"] == "BUY")
            sell_signals = sum(1 for signal in self.signals.values() if signal["signal"] == "SELL")
            strong_buy = sum(
                1 for signal in self.signals.values() if signal["signal"] == "BUY" and signal["strength"] == "STRONG")
            strong_sell = sum(
                1 for signal in self.signals.values() if signal["signal"] == "SELL" and signal["strength"] == "STRONG")

            buy_weight = buy_signals + strong_buy
            sell_weight = sell_signals + strong_sell

            if buy_weight > sell_weight + 1:
                self.signals["overall"] = {"signal": "BUY",
                                           "explanation": f"{buy_signals} buy signals ({strong_buy} strong) vs {sell_signals} sell signals ({strong_sell} strong)"}
            elif sell_weight > buy_weight + 1:
                self.signals["overall"] = {"signal": "SELL",
                                           "explanation": f"{sell_signals} sell signals ({strong_sell} strong) vs {buy_signals} buy signals ({strong_buy} strong)"}
            else:
                self.signals["overall"] = {"signal": "HOLD", "explanation": "Mixed signals - no clear direction"}
        return self.signals

    def plot_indicators(self, show_signals=True, save_path=None, fig_size=(15, 12)):
        """
        Plot key technical indicators and trading signals.

        Parameters:
            show_signals (bool): Whether to display trading signals on the chart (default: True)
            save_path (str): File path to save the chart image (default: None - not saved)
            fig_size (tuple): Figure size as (width, height) in inches (default: (15, 12))

        Returns:
            matplotlib.figure.Figure: The chart figure object or None if error
        """
        # Better error handling and data preparation
        if self.hist is None or self.hist.empty:
            error = self.fetch_data()
            if error:
                print(f"Error fetching data: {error}")
                return None
            elif self.hist is None or self.hist.empty:  # Check again after a fetch attempt
                print("Error: No data available after fetch attempt")
                return None

        if not self.indicators:
            self.compute_indicators()

        if show_signals and self.signals is None:
            self.get_trading_signals()

        try:
            # Prepare data for plotting
            df = self.hist.copy()
            df.reset_index(inplace=True)

            # Convert datetime to numeric for candlestick
            date_col = next((col for col in ['Datetime', 'Date'] if col in df.columns), None)
            if not date_col:
                print("Error: No date/datetime column found in data")
                return None

            df['Date_num'] = m_dates.date2num(df[date_col])
            ohlc = df[['Date_num', 'Open', 'High', 'Low', 'Close']]

            # Pre-calculate common values used in multiple indicators
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Create a figure and GridSpec
            fig = plt.figure(figsize=fig_size)
            gs = GridSpec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.15)

            # SUBPLOT 1: Price Chart
            ax1 = fig.add_subplot(gs[0])
            self._plot_price_chart(ax1, df, ohlc)

            # SUBPLOT 2: Volume
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            self._plot_volume(ax2, df)

            # SUBPLOT 3: MACD
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            self._plot_macd(ax3, df, show_signals)

            # SUBPLOT 4: RSI
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            self._plot_rsi(ax4, df, show_signals)

            # SUBPLOT 5: Stochastic
            ax5 = fig.add_subplot(gs[4], sharex=ax1)
            self._plot_stochastic(ax5, df, show_signals)

            # SUBPLOT 6: ATR
            ax6 = fig.add_subplot(gs[5], sharex=ax1)
            self._plot_atr(ax6, df, true_range)

            # Format x-axis dates
            plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
            ax6.xaxis.set_major_locator(m_ticker.MaxNLocator(10))
            ax6.xaxis.set_major_formatter(m_dates.DateFormatter('%m-%d %H:%M'))

            # Add footer information
            self._add_footer_info()

            # Adjust layout
            plt.subplots_adjust(left=0.09, right=0.94, top=0.95, bottom=0.15)

            # Save or display figure
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            else:
                # plt.show()
                pass

            return fig

        except Exception as e:
            print(f"Error plotting indicators: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_price_chart(self, ax, df, ohlc):
        """Helper method to plot price chart with indicators"""
        # Plot candlesticks
        candlestick_ohlc(ax, ohlc.values, width=0.6 / 24, colorup='green', colordown='red', alpha=0.8)

        # Add Moving Averages
        ema9 = df['Close'].ewm(span=9, adjust=False).mean()
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()

        ax.plot(df['Date_num'], ema9, label='EMA(9)', color='blue', linewidth=1)
        ax.plot(df['Date_num'], ema20, label='EMA(20)', color='orange', linewidth=1)
        ax.plot(df['Date_num'], ema50, label='EMA(50)', color='purple', linewidth=1)

        # Add Bollinger Bands
        if len(df) >= 20:
            middle_band = df['Close'].rolling(window=20).mean()
            std_dev = df['Close'].rolling(window=20).std()
            upper_band = middle_band + (std_dev * 2)
            lower_band = middle_band - (std_dev * 2)

            ax.plot(df['Date_num'], upper_band, label='Upper BB', color='gray', linestyle='--', alpha=0.6)
            ax.plot(df['Date_num'], middle_band, label='Middle BB', color='gray', linestyle='-', alpha=0.6)
            ax.plot(df['Date_num'], lower_band, label='Lower BB', color='gray', linestyle='--', alpha=0.6)

        # Add Keltner Channels if available
        self._add_keltner_channels(ax, df)

        # Add Pivot Points
        self._add_pivot_points(ax)

        # Format x-axis
        ax.xaxis.set_major_formatter(m_dates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.set_title(f'{self.ticker_symbol} - Technical Indicators', fontsize=16)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # Add an overall trading signal if available
        if self.signals and 'overall' in self.signals:
            signal_color = {
                'BUY': 'green',
                'SELL': 'red',
                'HOLD': 'gray'
            }.get(self.signals['overall']['signal'], 'gray')

            ax.annotate(f"Signal: {self.signals['overall']['signal']}",
                        xy=(0.02, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc=signal_color, alpha=0.3),
                        fontsize=12)

    def _add_keltner_channels(self, ax, df):
        """Helper method to add Keltner Channels to price chart"""
        if 'technical_indicators' in self.indicators and 'keltner_upper' in self.indicators['technical_indicators']:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            ema_20 = typical_price.ewm(span=20, adjust=False).mean()

            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_10 = true_range.rolling(window=10).mean()

            keltner_upper = ema_20 + (atr_10 * 1.5)
            keltner_lower = ema_20 - (atr_10 * 1.5)

            ax.plot(df['Date_num'], keltner_upper, label='KC Upper', color='purple', linestyle='--', alpha=0.5)
            ax.plot(df['Date_num'], ema_20, label='KC Middle', color='purple', linestyle='-', alpha=0.5)
            ax.plot(df['Date_num'], keltner_lower, label='KC Lower', color='purple', linestyle='--', alpha=0.5)

    def _add_pivot_points(self, ax):
        """Helper method to add pivot points to price chart"""
        if 'pivot_points' in self.indicators and 'pivot' in self.indicators['pivot_points']:
            pivot = self.indicators['pivot_points']['pivot']
            r1 = self.indicators['pivot_points']['r1']
            r2 = self.indicators['pivot_points']['r2']
            s1 = self.indicators['pivot_points']['s1']
            s2 = self.indicators['pivot_points']['s2']

            ax.axhline(y=pivot, color='black', linestyle='-', alpha=0.5, label='Pivot')
            ax.axhline(y=r1, color='red', linestyle='--', alpha=0.5, label='R1')
            ax.axhline(y=r2, color='red', linestyle='-.', alpha=0.5, label='R2')
            ax.axhline(y=s1, color='green', linestyle='--', alpha=0.5, label='S1')
            ax.axhline(y=s2, color='green', linestyle='-.', alpha=0.5, label='S2')

    def _plot_volume(self, ax, df):
        """Helper method to plot volume chart"""
        volume_colors = np.where(df['Close'] >= df['Open'], 'green', 'red')
        ax.bar(df['Date_num'], df['Volume'], color=volume_colors, alpha=0.5, width=0.6 / 24)

        # Add a 5-day moving average of volume
        volume_ma = df['Volume'].rolling(window=5).mean()
        ax.plot(df['Date_num'], volume_ma, color='blue', linewidth=1, label='Volume MA(5)')

        ax.set_ylabel('Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            m_ticker.FuncFormatter(lambda x, p: f'{x / 1000:.0f}K' if x < 1e6 else f'{x / 1e6:.1f}M'))

        # Add relative volume info
        if 'technical_indicators' in self.indicators and 'relative_volume' in self.indicators['technical_indicators']:
            rel_vol = self.indicators['technical_indicators']['relative_volume']
            ax.annotate(f"Rel Vol: {rel_vol:.1f}x",
                        xy=(0.85, 0.8), xycoords='axes fraction',
                        fontsize=10)

    def _plot_macd(self, ax, df, show_signals=True):
        """Helper method to plot MACD"""
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        # Plot MACD
        ax.plot(df['Date_num'], macd, label='MACD', color='blue', linewidth=1)
        ax.plot(df['Date_num'], signal, label='Signal', color='red', linewidth=1)

        # Plot histogram
        positive = histogram > 0
        negative = histogram <= 0

        ax.bar(df.loc[positive, 'Date_num'], histogram.loc[positive], color='green', alpha=0.5, width=0.6 / 24)
        ax.bar(df.loc[negative, 'Date_num'], histogram.loc[negative], color='red', alpha=0.5, width=0.6 / 24)

        ax.set_ylabel('MACD', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # Add MACD signal if available
        if show_signals and self.signals and 'macd' in self.signals:
            ax.annotate(f"{self.signals['macd']['signal']}",
                        xy=(0.85, 0.8), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.5),
                        fontsize=10)

    def _plot_rsi(self, ax, df, show_signals=True):
        """Helper method to plot RSI"""
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Plot RSI
        ax.plot(df['Date_num'], rsi, label='RSI(14)', color='purple', linewidth=1)

        # Add overbought/oversold lines
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=50, color='black', linestyle='-', alpha=0.2)

        ax.set_ylabel('RSI', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add RSI signal if available
        if show_signals and self.signals and 'rsi' in self.signals:
            ax.annotate(f"{self.signals['rsi']['signal']}",
                        xy=(0.85, 0.8), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.5),
                        fontsize=10)

    def _plot_stochastic(self, ax, df, show_signals=True):
        """Helper method to plot Stochastic Oscillator"""
        # Calculate Stochastic Oscillator
        window = 14
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()

        # Avoid division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, 1e-10)

        k_percent = 100 * ((df['Close'] - low_min) / denominator)
        d_percent = k_percent.rolling(window=3).mean()

        # Plot Stochastic Oscillator
        ax.plot(df['Date_num'], k_percent, label='%K', color='blue', linewidth=1)
        ax.plot(df['Date_num'], d_percent, label='%D', color='red', linewidth=1)

        # Add overbought/oversold lines
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5)

        ax.set_ylabel('Stochastic', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # Add Stochastic signal if available
        if show_signals and self.signals and 'stoch' in self.signals:
            ax.annotate(f"{self.signals['stoch']['signal']}",
                        xy=(0.85, 0.8), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.5),
                        fontsize=10)

    @staticmethod
    def _plot_atr(ax, df, true_range=None):
        """Helper method to plot ATR"""
        # Calculate ATR if not provided
        if true_range is None:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = true_range.rolling(window=14).mean()

        # Plot ATR
        ax.plot(df['Date_num'], atr, label='ATR(14)', color='orange', linewidth=1)

        ax.set_ylabel('ATR', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    def _add_footer_info(self):
        """Add ticker and timestamp information to figure footer"""
        plt.figtext(0.02, 0.02, f"Ticker: {self.ticker_symbol}", fontsize=10)

        if 'timestamp' in self.indicators:
            plt.figtext(0.75, 0.02, f"Generated: {self.indicators['timestamp']}", fontsize=10)

    def run(self):
        self.indicators = self.compute_indicators()
        self.signals = self.get_trading_signals()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        plots_dir = os.path.join(project_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir,
                                 f"{self.ticker_symbol} - Trading indicators - period {self.period} - interval {self.interval}.png")
        self.plot_indicators(
            save_path=save_path
        )
        return {
            "info": self.indicators,
            "signals": self.signals,
            "plot": save_path,
        }


# if __name__ == "__main__":
#     trader = DailyTradingIndicators("AAPL", period="30d", interval="1h")
#     result = json.dumps(trader.run(), indent=2, default=str)
#     print(result)

