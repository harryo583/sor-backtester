
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
        volume_spike_factor: float = 3.0
    ):
        """
        start_price: initial price of the asset at the beginning of the day
        num_points: number of time intervals (e.g. minutes in trading day)
        base_volume: mean for the Poisson distribution
        volatility_factor: standard deviation for random walk steps
        volume_spike_probability: probability of a volume spike occurring at each time step
        volume_spike_factor: multiplier for volume at spike time
        """
        self.start_price = start_price
        self.num_points = num_points
        self.base_volume = base_volume
        self.volatility_factor = volatility_factor
        self.volume_spike_probability = volume_spike_probability
        self.volume_spike_factor = volume_spike_factor
        
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
        
        # Apply volume spikes
        for i in range(self.num_points):
            if np.random.rand() < self.volume_spike_probability:
                volumes[i] = int(volumes[i] * self.volume_spike_factor)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })

    def get_current_state(self) -> Dict[str, Any]:
        """
        Returns current market information (time, price, volume).
        """
        if self.current_step >= self.num_points:
            return None  # no more data left
        row = self.market_data.iloc[self.current_step]
        return {
            'time': row['timestamp'],
            'price': row['price'],
            'volume': row['volume']
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
    
    def __init__(self, total_shares: int, strategy_name: str = "BaseStrategy"):
        self.total_shares = total_shares
        self.strategy_name = strategy_name
        self.accumulated_shares = 0  # how many shares have been bought/sold
    
    @abc.abstractmethod
    def generate_orders(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Each strategy must define how many shares to trade at the current step.
        Takes in market_info - a dictionary with current 'price' 'volume' etc.
        Returns a dictionary:
            {
                'shares_to_execute': int,
                'price': float,
                'timestamp': pd.Timestamp
            }
        """
        pass

class TWAPStrategy(Strategy):
    """
    Slices the total order evenly across all time steps.
    """
    def __init__(self, total_shares: int, total_steps: int):
        super().__init__(total_shares, strategy_name="TWAP")
        self.shares_per_step = total_shares // total_steps if total_steps > 0 else 0
        self.remaining_shares = total_shares

    def generate_orders(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a fixed slice at each step until all shares are allocated.
        """
        if market_info is None or self.remaining_shares <= 0:
            return {'shares_to_execute': 0, 'price': 0.0, 'timestamp': None}
        
        shares = min(self.shares_per_step, self.remaining_shares)
        price = market_info['price']
        ts = market_info['time']
        
        self.accumulated_shares += shares
        self.remaining_shares -= shares
        
        return {
            'shares_to_execute': shares,
            'price': price,
            'timestamp': ts
        }

class VWAPStrategy(Strategy):
    """
    Allocates order slices proportional to the volume traded.
    Aims to match the overall intraday volume distribution.
    """
    def __init__(self, total_shares: int, market_env: MarketEnvironment):
        super().__init__(total_shares, strategy_name="VWAP")
        self.market_env = market_env
        
        # Precompute the cumulative volume distribution for the day
        self.vwap_weights = self._compute_vwap_weights()
        self.remaining_shares = total_shares

    def _compute_vwap_weights(self) -> List[float]:
        """
        Helper method; calculate the proportional weight for each time step based on forecasted volume.
        """
        total_volume = self.market_env.market_data['volume'].sum()
        weights = self.market_env.market_data['volume'] / total_volume
        return weights.tolist()

    def generate_orders(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate shares to be traded in the current time step based on the precomputed VWAP weights.
        """
        if market_info is None or self.remaining_shares <= 0:
            return {'shares_to_execute': 0, 'price': 0.0, 'timestamp': None}

        # Get the weight for the current step and calculate shares to trade
        current_step = self.market_env.current_step
        weight = self.vwap_weights[current_step]
        shares_this_step = int(self.total_shares * weight)
        shares_this_step = min(shares_this_step, self.remaining_shares)  # ensure we don't over-allocate

        # Market info
        price = market_info['price']
        ts = market_info['time']

        self.accumulated_shares += shares_this_step
        self.remaining_shares -= shares_this_step

        return {
            'shares_to_execute': shares_this_step,
            'price': price,
            'timestamp': ts
        }


class Backtester:
    """
    Orchestrates the simulation, steps through each time increment, querying the
    strategy for orders, executing them, and recording results.
    """
    
    def __init__(self, market_env: MarketEnvironment, strategy: Strategy):
        self.market_env = market_env
        self.strategy = strategy
        self.execution_records: List[Dict[str, Any]] = [] # keep track of executions
        
    def run(self):
        """
        For each timestep, gather market state, get strategy orders, and record the execution info.
        """
        self.market_env.reset()
        
        while True: # main loop!
            market_info = self.market_env.get_current_state()
            if not market_info: break
            
            order = self.strategy.generate_orders(market_info)
            
            if order['shares_to_execute'] > 0:
                self.execution_records.append({
                    'timestamp': order['timestamp'],
                    'execution_price': order['price'],
                    'execution_volume': order['shares_to_execute']
                })
            
            self.market_env.step()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculates the following metrics:
          1. Overall VWAP.
          2. Average execution price.
          3. Execution cost vs VWAP.
          4. Simple slippage measure.
        """
        exec_df = pd.DataFrame(self.execution_records)
        if exec_df.empty:
            return {
                'Overall VWAP': 0.0,
                'Avg Execution Price': 0.0,
                'Execution Cost vs. VWAP': 0.0,
                'Slippage (average)': 0.0
            }
        
        # 1. Overall VWAP from market environment
        mkt_df = self.market_env.market_data
        total_dollar_volume = (mkt_df['price'] * mkt_df['volume']).sum()
        total_volume = mkt_df['volume'].sum()
        overall_vwap = total_dollar_volume / total_volume
        
        # 2. Average execution price
        total_exec_dollar = (exec_df['execution_price'] * exec_df['execution_volume']).sum()
        total_exec_volume = exec_df['execution_volume'].sum()
        avg_exec_price = total_exec_dollar / total_exec_volume
        
        # 3. Execution cost vs. VWAP
        exec_cost = avg_exec_price - overall_vwap
        
        # 4. Slippage
        merged = pd.merge(
            exec_df, 
            mkt_df, 
            left_on='timestamp', 
            right_on='timestamp',
            how='left'
        ) # merge execution times with corresponding market prices to see deviation
        merged['slippage_per_share'] = merged['execution_price'] - merged['price']
        merged['weighted_slippage'] = merged['slippage_per_share'] * merged['execution_volume']
        total_slippage = merged['weighted_slippage'].sum()
        avg_slippage = total_slippage / total_exec_volume
        
        return {
            'Overall VWAP': overall_vwap,
            'Avg Execution Price': avg_exec_price,
            'Execution Cost vs. VWAP': exec_cost,
            'Slippage (average)': avg_slippage
        }

    def plot_results(self):
        """
        Plot the market price and overlay execution points.
        """
        if not self.execution_records:
            print("No executions to plot.")
            return
        
        mkt_df = self.market_env.market_data
        exec_df = pd.DataFrame(self.execution_records)
        
        plt.figure(figsize=(10, 5))
        plt.plot(mkt_df['timestamp'], mkt_df['price'], label='Market Price')
        plt.scatter(exec_df['timestamp'], exec_df['execution_price'], 
                    color='red', label='Executions', s=10)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"{self.strategy.strategy_name} Execution vs. Market Price")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Simple scenario with standard volatility and no volume spike
    market_env1 = MarketEnvironment(
        start_price=100.0,
        num_points=390,
        base_volume=300,
        volatility_factor=0.2,
        volume_spike_probability=0.0,
        volume_spike_factor=3.0
    )
    twap_strategy1 = TWAPStrategy(total_shares=10000, total_steps=390)
    backtester1 = Backtester(market_env=market_env1, strategy=twap_strategy1)
    backtester1.run()
    metrics1 = backtester1.calculate_metrics()
    
    for k, v in metrics1.items():
        print(f"{k}: {v:.4f}")
    
    backtester1.plot_results()
    
    # Scenario with higher volatility and a midday volume spike
    market_env2 = MarketEnvironment(
        start_price=100.0,
        num_points=390,
        base_volume=300,
        volatility_factor=0.5, # higher volatility
        volume_spike_probability=0.1, # volume spike around midday
        volume_spike_factor=5.0
    )
    twap_strategy2 = TWAPStrategy(total_shares=10000, total_steps=390)
    backtester2 = Backtester(market_env=market_env2, strategy=twap_strategy2)
    backtester2.run()
    metrics2 = backtester2.calculate_metrics()
    
    for k, v in metrics2.items():
        print(f"{k}: {v:.4f}")
    
    backtester2.plot_results()
