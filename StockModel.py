import yfinance as yf
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

class Stock:
    def __init__(self, ticker: str, start: str, end: str) -> None:
        self.ticker = ticker
        self.prices = self._fetch_prices(start, end)
        self.daily_returns = self._calculate_daily_returns()
        self.current_price = self._get_current_price()

    def _fetch_prices(self, start: str, end: str):
        return yf.download(self.ticker, start=start, end=end, progress=False)["Close"]

    def _get_current_price(self):
        return yf.Ticker(self.ticker).info["currentPrice"]

    def _calculate_daily_returns(self):
        return self.prices.pct_change()

    def annualized_return(self) -> float:
        return np.mean(self.daily_returns) * 252

    def annualized_volatility(self) -> float:
        return np.std(self.daily_returns) * np.sqrt(252)

    def sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        return (self.annualized_return() - risk_free_rate) / self.annualized_volatility()

    def moving_average(self, window: int) -> float:
        # average closing price over last `window` days
        return self.prices.tail(window).mean().iloc[-1]

    def entry_signal(self) -> str:
        # compare current price to 50-day moving average
        if self.current_price < self.moving_average(50):
            return "BUY SIGNAL"
        else:
            return "No signal"


class SP500Scanner:
    def __init__(self, max_price: float = 100.0, start: str = "2024-01-01", end: str = None):
        self.max_price = max_price
        self.start = start
        self.end = end
        self.tickers = self._get_sp500_tickers()
        self.stocks = []

    def _get_sp500_tickers(self) -> list[str]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        return pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0][
            "Symbol"].tolist()

    def _filter_by_price(self) -> list[Stock]:
        for ticker in self.tickers:
            try:
                stock = Stock(ticker, self.start, self.end)
                if stock.current_price <= self.max_price:
                    self.stocks.append(stock)
            except Exception as e:
                f"Skipping {ticker} due to error: {e}"
        return self.stocks

    def rank_by_sharpe(self, top_n: int = 10) -> list[Stock]:
        self._filter_by_price()
        sorted_stocks = sorted(self.stocks, key=lambda stock: stock.sharpe_ratio(), reverse=True)
        return sorted_stocks[:top_n]

    def run(self):
        print(f"Scanning S&P 500 for stocks under ${self.max_price}...\n")
        top_stocks = self.rank_by_sharpe(top_n=10)

        print(f"Top 10 Stocks by Sharpe Ratio (under ${self.max_price}):\n")
        print(f"{'Ticker':<10} {'Price':>8} {'Return':>10} {'Volatility':>12} {'Sharpe':>8} "
              f"{'Signal':>15}")
        print("-" * 65)

        for stock in top_stocks:
            print(f"{stock.ticker:<10} "
                  f"${stock.current_price:>7.2f} "
                  f"{stock.annualized_return():>9.2%} "
                  f"{stock.annualized_volatility():>11.2%} "
                  f"{stock.sharpe_ratio():>8.2f} "
                  f"{stock.entry_signal():>15}")


class MonteCarlo:
    def __init__(self, stock: Stock, days: int = 252, simulations: int = 1000):
        self.stock = stock
        self.days = days
        self.simulations = simulations

    def run(self) -> list:
        returns = self.stock.daily_returns.dropna().values.flatten()
        final_prices = []

        for _ in range(self.simulations):
            price = self.stock.current_price
            random_returns = np.random.choice(returns, size=self.days)
            for daily_return in random_returns:
                price = price * (1 + daily_return)
            final_prices.append(price)

        return final_prices

    def summary(self):
        results = self.run()
        current = self.stock.current_price
        expected = np.mean(results)
        best = np.max(results)
        worst = np.min(results)

        count = 0
        for p in results:
            if p > current:
                count += 1
        prob_profit = count / len(results)

        print(f"Current Price:      ${current:.2f}")
        print(f"Expected Price:     ${expected:.2f}")
        print(f"Best Case:          ${best:.2f}")
        print(f"Worst Case:         ${worst:.2f}")
        print(f"Probability of Profit: {prob_profit:.2%}")

        plt.figure(figsize=(10, 5))
        plt.hist(results, bins=50, color="steelblue", edgecolor="black")
        plt.axvline(current, color="red", linestyle="--", label=f"Current Price ${current:.2f}")
        plt.axvline(expected, color="green", linestyle="--",
                    label=f"Expected Price ${expected:.2f}")
        plt.title(f"Monte Carlo Simulation: {self.stock.ticker} — 1000 Simulations over 1 Year")
        plt.xlabel("Final Price ($)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig("monte_carlo.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    scanner = SP500Scanner(max_price=100.0)
    scanner.run()

    print("\n--- Monte Carlo Simulation: Top Pick ---\n")
    top_stock = scanner.rank_by_sharpe(top_n=1)[0]
    mc = MonteCarlo(stock=top_stock, days=252, simulations=1000)
    mc.summary()
