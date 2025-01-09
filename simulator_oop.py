
import abc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

class MarketEnvironment:
    """
    Simulates the market for one trading day (390 minutes for equities).
    Allows injection of various conditions (e.g. higher volatility, volume spikes)
    to observe how different strategies perform under changing markets.
    """
    def __init__(
        self,
        start_price: float = 100.0,
        num_points: int = 390,
        base_volume: int = 300,
        volatility_factor: float = 0.2,
        volume_spike_probability: float = 0.0,
        volume_spike_factor: float = 3.0,
        fixed_spread: float = 0.02
    ):
        """
        start_price: initial price of the asset at the beginning of the day
        num_points: number of time intervals (e.g. minutes in trading day)
        base_volume: mean for the Poisson distribution
        volatility_factor: standard deviation for random walk steps
        volume_spike_probability: probability of a volume spike occurring at each time step
        volume_spike_factor: multiplier for volume at spike time
        fixed_spread: distance between bid and ask
        """
        self.start_price = start_price
        self.num_points = num_points
        self.base_volume = base_volume
        self.volatility_factor = volatility_factor
        self.volume_spike_probability = volume_spike_probability
        self.volume_spike_factor = volume_spike_factor
        self.fixed_spread = fixed_spread
        
        self.market_data = self._generate_synthetic_data() # internal dataframe storing the market state at each timestep
        self.current_step = 0 # "current time"

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generates a random-walk price series and Poisson volume, with possible volume spikes.
        """
        timestamps = pd.date_range("2025-01-01 09:30:00", periods=self.num_points, freq='T')
        prices = [self.start_price]
        
        # Generate random-walk prices
        for _ in range(self.num_points - 1):
            step = np.random.normal(0, self.volatility_factor)
            new_price = prices[-1] + step
            prices.append(max(new_price, 0.01))  # ensure price doesn't go negative
        
        # Generate volumes
        volumes = np.random.poisson(self.base_volume, self.num_points)
        for i in range(self.num_points):
            if np.random.rand() < self.volume_spike_probability:
                volumes[i] = int(volumes[i] * self.volume_spike_factor)
        
        # Split into bid and ask volumes (assume 50-50)
        bid_volumes = np.floor(volumes * 0.5).astype(int)
        ask_volumes = volumes - bid_volumes  # remainder
        
        # Construct bid / ask prices
        df = pd.DataFrame({
            'timestamp': timestamps,
            'mid_price': prices,
            'total_volume': volumes,
            'bid_volume': bid_volumes,
            'ask_volume': ask_volumes
        })
        df['bid_price'] = df['mid_price'] - (self.fixed_spread / 2.0)
        df['ask_price'] = df['mid_price'] + (self.fixed_spread / 2.0)
        
        return df
        
    def get_current_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary of current bid/ask info and volumes.
        """
        if self.current_step >= self.num_points:
            return None
        row = self.market_data.iloc[self.current_step]
        return {
            'time': row['timestamp'],
            'bid_price': row['bid_price'],
            'ask_price': row['ask_price'],
            'bid_volume': row['bid_volume'],
            'ask_volume': row['ask_volume']
        }

    def step(self):
        """
        Moves the market environment to the next time step.
        """
        self.current_step += 1

    def reset(self):
        """
        Resets the environment to the start of the day.
        """
        self.current_step = 0


class Strategy(abc.ABC):
    """
    Abstract base class for all strategies. Requires at least one method
    to generate orders given the market state and an outstanding position.
    """
    
    def __init__(self, total_shares: int, strategy_name: str = "BaseStrategy", side: str = "BUY"):
        self.total_shares = total_shares
        self.strategy_name = strategy_name
        self.side = side
        self.accumulated_shares = 0  # how many shares have been bought/sold
    
    @abc.abstractmethod
    def generate_orders(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Must return dict with:
          {
            'shares_to_execute': int,
            'side': str,  # 'BUY' or 'SELL'
            'timestamp': pd.Timestamp
          }
        """
        pass


class TWAPStrategy(Strategy):
    """
    Slices the total order evenly across all time steps.
    """
    def __init__(self, total_shares: int, total_steps: int, side: str = "BUY"):
        super().__init__(total_shares, strategy_name="TWAP", side=side)
        self.shares_per_step = total_shares // total_steps if total_steps > 0 else 0
        self.remaining_shares = total_shares

    def generate_orders(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
        if market_info is None or self.remaining_shares <= 0:
            return {
                'shares_to_execute': 0, 
                'side': self.side,
                'timestamp': None
            }
        
        shares = min(self.shares_per_step, self.remaining_shares)
        self.accumulated_shares += shares
        self.remaining_shares -= shares
        
        return {
            'shares_to_execute': shares,
            'side': self.side,
            'timestamp': market_info['time']
        }

# class VWAPStrategy(Strategy):
#     """
#     Allocates order slices proportional to the volume traded.
#     Aims to match the overall intraday volume distribution.
#     """
#     def __init__(self, total_shares: int, market_env: MarketEnvironment):
#         super().__init__(total_shares, strategy_name="VWAP")
#         self.market_env = market_env
        
#         # Precompute the cumulative volume distribution for the day
#         self.vwap_weights = self._compute_vwap_weights()
#         self.remaining_shares = total_shares

#     def _compute_vwap_weights(self) -> List[float]:
#         """
#         Helper method; calculate the proportional weight for each time step based on forecasted volume.
#         """
#         total_volume = self.market_env.market_data['volume'].sum()
#         weights = self.market_env.market_data['volume'] / total_volume
#         return weights.tolist()

#     def generate_orders(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Allocate shares to be traded in the current time step based on the precomputed VWAP weights.
#         """
#         if market_info is None or self.remaining_shares <= 0:
#             return {'shares_to_execute': 0, 'price': 0.0, 'timestamp': None}

#         # Get the weight for the current step and calculate shares to trade
#         current_step = self.market_env.current_step
#         weight = self.vwap_weights[current_step]
#         shares_this_step = int(self.total_shares * weight)
#         shares_this_step = min(shares_this_step, self.remaining_shares)  # ensure we don't over-allocate

#         # Market info
#         price = market_info['price']
#         ts = market_info['time']

#         self.accumulated_shares += shares_this_step
#         self.remaining_shares -= shares_this_step

#         return {
#             'shares_to_execute': shares_this_step,
#             'price': price,
#             'timestamp': ts
#         }

class Backtester:
    """
    Orchestrates the simulation. On each time step:
      1. Fetches market environment state (bid/ask, volumes).
      2. Asks the strategy how many shares to trade (and side).
      3. Fills them incrementally, possibly moving the price if volume is exceeded.
      4. Records the fill details.
      5. Steps the environment.
    """
    
    def __init__(self, market_env: MarketEnvironment, strategy: Strategy):
        self.market_env = market_env
        self.strategy = strategy
        self.execution_records: List[Dict[str, Any]] = []
    
    def run(self):
        """
        Main simulation loop: gather orders from strategy, fill them, track partial fills & price movements.
        """
        self.market_env.reset()
        
        while True:
            market_info = self.market_env.get_current_state()
            if market_info is None:
                break  # end of simulation

            # 1. Strategy decides how many shares to trade this step
            order = self.strategy.generate_orders(market_info)
            shares_to_execute = order['shares_to_execute']
            side = order['side']
            ts = order['timestamp']

            if shares_to_execute <= 0 or ts is None: # no trade or no data
                self.market_env.step()
                continue

            # 2. Attempt to fill the order in increments:
            #  If BUY, we consume available ask volume at the current ask_price.
            #  If we exceed the ask volume, assume the price increments by some small tick 
            #  (e.g. 1 cent) and we get more volume (a naive approach).
            #  Same logic for SELL at the bid side.

            if side == "BUY":
                filled_shares, avg_fill_price = self._simulate_buy(
                    shares_to_execute,
                    initial_ask_price=market_info['ask_price'],
                    initial_ask_volume=market_info['ask_volume']
                )
            else:  # SELL
                filled_shares, avg_fill_price = self._simulate_sell(
                    shares_to_execute,
                    initial_bid_price=market_info['bid_price'],
                    initial_bid_volume=market_info['bid_volume']
                )

            if filled_shares > 0:
                self.execution_records.append({
                    'timestamp': ts,
                    'execution_price': avg_fill_price,
                    'execution_volume': filled_shares,
                    'side': side
                })

            self.market_env.step()

    def _simulate_buy(self, shares_to_buy: int, initial_ask_price: float, initial_ask_volume: int):
        """
        Naive partial-fill simulation for a BUY order
        """
        remaining = shares_to_buy
        current_price = initial_ask_price
        current_volume = initial_ask_volume
        
        fill_details = []  # (shares_filled, fill_price)
        price_impact_tick = 0.01  # how much the ask moves each time we deplete available volume

        while remaining > 0:
            if current_volume >= remaining: # we fill the entire remainder at current_price
                fill_details.append((remaining, current_price))
                current_volume -= remaining
                remaining = 0
            else: # we consume all of current_volume at current_price
                fill_details.append((current_volume, current_price))
                remaining -= current_volume
                current_price += price_impact_tick
                current_volume = initial_ask_volume # assume each price level has the same volume as the original ask_volume (naive implementation)

        # Compute the volume-weighted average fill price
        total_shares = sum([fd[0] for fd in fill_details])
        dollar_spent = sum([fd[0] * fd[1] for fd in fill_details])
        avg_price = dollar_spent / total_shares if total_shares > 0 else current_price

        return total_shares, avg_price

    def _simulate_sell(self, shares_to_sell: int, initial_bid_price: float, initial_bid_volume: int):
        """
        Naive partial-fill simulation for a SELL order
        """
        remaining = shares_to_sell
        current_price = initial_bid_price
        current_volume = initial_bid_volume
        
        fill_details = []  # (shares_filled, fill_price)
        price_impact_tick = 0.01  # how much the bid moves down each time we deplete available volume

        while remaining > 0:
            if current_volume >= remaining: # we fill the entire remainder at current_price
                fill_details.append((remaining, current_price))
                current_volume -= remaining
                remaining = 0
            else: # consume all of current_volume at current_price
                fill_details.append((current_volume, current_price))
                remaining -= current_volume
                current_price -= price_impact_tick
                current_volume = initial_bid_volume # Assume each price level has the same volume as the original ask_volume (naive implementation)

        # Volume-weighted average fill price
        total_shares = sum([fd[0] for fd in fill_details])
        dollar_gained = sum([fd[0] * fd[1] for fd in fill_details])
        avg_price = dollar_gained / total_shares if total_shares > 0 else current_price

        return total_shares, avg_price

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Summarize performance:
          - Overall "mid-price VWAP" from the entire day (for reference)
          - Average execution price (for buys and sells)
          - Execution cost vs. mid-price VWAP
          - Simple slippage measure, referencing the midprice
        """
        if not self.execution_records:
            return {
                'Overall Mid VWAP': 0.0,
                'Avg Execution Price': 0.0,
                'Execution Cost vs. Mid VWAP': 0.0,
                'Slippage (volume weighted average)': 0.0
            }

        exec_df = pd.DataFrame(self.execution_records)

        # 1. Overall mid-price VWAP across entire day
        mkt_df = self.market_env.market_data
        dollar_volume = (mkt_df['mid_price'] * mkt_df['total_volume']).sum()
        overall_volume = mkt_df['total_volume'].sum()
        overall_mid_vwap = dollar_volume / overall_volume

        # 2. Average execution price
        total_shares_exec = exec_df['execution_volume'].sum()
        total_exec_dollars = (exec_df['execution_price'] * exec_df['execution_volume']).sum()
        avg_exec_price = total_exec_dollars / total_shares_exec

        # 3. Execution cost vs. mid-price VWAP
        # If we are buying, cost is (avg_exec_price - mid_vwap).
        # If we are selling, cost could be (mid_vwap - avg_exec_price).
        # Here we just compute the difference for simplicity
        exec_cost = avg_exec_price - overall_mid_vwap

        # 4. VOlume weighted average slippage
        merged = pd.merge(
            exec_df,
            mkt_df[['timestamp', 'mid_price']],
            left_on='timestamp',
            right_on='timestamp',
            how='left'
        )
        merged['slippage_per_share'] = merged['execution_price'] - merged['mid_price']
        merged['weighted_slippage'] = merged['slippage_per_share'] * merged['execution_volume']
        total_slippage = merged['weighted_slippage'].sum()
        avg_slippage = total_slippage / total_shares_exec

        return {
            'Overall Mid VWAP': overall_mid_vwap,
            'Avg Execution Price': avg_exec_price,
            'Execution Cost vs. Mid VWAP': exec_cost,
            'Slippage (average)': avg_slippage
        }

    def plot_results(self):
        """
        Plot the mid_price over time along with execution points.
        """
        if not self.execution_records:
            print("No executions to plot.")
            return
        
        exec_df = pd.DataFrame(self.execution_records)
        mkt_df = self.market_env.market_data

        plt.figure(figsize=(10, 5))
        plt.plot(mkt_df['timestamp'], mkt_df['mid_price'], label='Mid Price')
        plt.scatter(exec_df['timestamp'], exec_df['execution_price'], 
                    color='red', label='Executions', s=10)
        plt.title(f"Executions vs. Market Mid Price - {self.strategy.strategy_name}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example 1: Low volatility, no volume spikes, BUY side
    market_env1 = MarketEnvironment(
        start_price=100.0,
        num_points=50,  # shorter day for demonstration
        base_volume=300,
        volatility_factor=0.1,
        volume_spike_probability=0.0,
        volume_spike_factor=2.0,
        fixed_spread=0.05
    )
    twap_buy_strategy = TWAPStrategy(total_shares=1000, total_steps=50, side="BUY")
    backtester_buy = Backtester(market_env1, twap_buy_strategy)
    backtester_buy.run()
    metrics_buy = backtester_buy.calculate_metrics()
    
    print("Example 1\n")
    for k, v in metrics_buy.items():
        print(f"{k}: {v:.4f}")
    backtester_buy.plot_results()

    # Example 2: Higher volatility, occasional volume spikes, SELL side
    market_env2 = MarketEnvironment(
        start_price=200.0,
        num_points=50,
        base_volume=500,
        volatility_factor=0.3,
        volume_spike_probability=0.2,
        volume_spike_factor=3.0,
        fixed_spread=0.10
    )
    twap_sell_strategy = TWAPStrategy(total_shares=1000, total_steps=50, side="SELL")
    backtester_sell = Backtester(market_env2, twap_sell_strategy)
    backtester_sell.run()
    metrics_sell = backtester_sell.calculate_metrics()
    
    print("Example 2\n")
    for k, v in metrics_sell.items():
        print(f"{k}: {v:.4f}")
    backtester_sell.plot_results()