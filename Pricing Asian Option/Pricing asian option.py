import numpy as np
from statistics import NormalDist, stdev
from math import exp, sqrt
from typing import Dict, Tuple


class Asset:
    """
    Base class for assets.

    Attributes
    ----------
    name : str
        The name of the asset.
    current_price : float
        The current price of the asset.
    dividend_yield : float
        The continuous dividend yield of the asset.
    """

    def __init__(self, name, current_price, *, dividend_yield=0.0):
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string.")
        if current_price <= 0:
            raise ValueError("Current price must be a positive number.")
        if dividend_yield < 0:
            raise ValueError("Dividend yield must be a non-negative number.")

        self.name = name
        self.current_price = current_price
        self.dividend_yield = dividend_yield


class BlackScholesAsset(Asset):
    def __init__(self, name, current_price, *, dividend_yield=0.0, volatility):
        """
        Subclass of Asset that follows a geometric Brownian motion model.

        Attributes
        ----------
        volatility : float
            The volatility of the asset.
        Initializes a Black-Scholes asset.
        """
        super().__init__(name, current_price, dividend_yield=dividend_yield)
        if volatility <= 0:
            raise ValueError("Volatility must be a positive number.")

        self.volatility = volatility

    def simulate_path(self, simulated_times, interest_rate, *, current_price=None):
        """
        Simulates the asset path using the geometric Brownian motion model.
        """
        if not simulated_times:
            raise ValueError("Simulated times must be a non-empty sequence.")
        if any(t <= 0 for t in simulated_times):
            raise ValueError(
                "Simulated times must contain positive values only.")
        if any(simulated_times[i] <= simulated_times[i - 1] for i in range(1, len(simulated_times))):
            raise ValueError("Simulated times must be strictly increasing.")
        if interest_rate <= 0:
            raise ValueError("Interest rate must be a positive number.")
        if current_price is not None and current_price <= 0:
            raise ValueError("Current price must be a positive number.")

        current_price = current_price if current_price is not None else self.current_price
        path = {0: current_price}

        for t in simulated_times:
            dt = t - max(path.keys())
            Z = np.random.normal(loc=0, scale=sqrt(dt))
            St = path[max(path.keys())] * exp(
                (interest_rate - self.dividend_yield - 0.5 *
                 self.volatility**2) * dt + self.volatility * Z
            )
            path[t] = St
        return path

    def _dst_ds0(self, path, time):
        """
        Computes the derivative of S(t) with respect to S(0).
        """
        if time not in path:
            raise ValueError("Time must be a key in the path.")
        S0 = path[0]
        St = path[time]
        return St / S0


class Option:
    def __init__(self, name: str, underlying: 'Asset'):
        """
        Base class for all options.
        """
        self.name = name
        self.underlying = underlying

    def monte_carlo_delta(self, simulations: int, interest_rate: float, *, confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate the delta (Δ) of this option using a Monte Carlo method.
        """
        if simulations < 2:
            raise ValueError("The number of simulations must be at least 2.")
        if not (0 < confidence_level < 1):
            raise ValueError(
                "The confidence level must be strictly between 0 and 1.")

        deltas = [
            self._path_delta(self.underlying.simulate_path(
                self.monitoring_times, interest_rate), interest_rate)
            for _ in range(simulations)
        ]

        mean_delta = sum(deltas) / simulations
        stddev_delta = stdev(deltas)

        z_score = NormalDist().inv_cdf((1 + confidence_level) / 2)
        delta_minus = mean_delta - (z_score * stddev_delta / sqrt(simulations))
        delta_plus = mean_delta + (z_score * stddev_delta / sqrt(simulations))

        return delta_minus, mean_delta, delta_plus

    def _path_delta(self, path: Dict[float, float], interest_rate: float) -> float:
        """
        Calculate delta for each path. To be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement this method to calculate delta for each path.")


class AsianOption(Option):
    """
    Class for Asian options with arithmetic averaging and floating strike.
    """

    def __init__(self, name, underlying, *, option_type, monitoring_times, strike_factor):
        super().__init__(name, underlying)

        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'.")
        if not monitoring_times or any(t <= 0 for t in monitoring_times):
            raise ValueError(
                "Monitoring times must be a strictly-increasing, non-empty sequence of positive times.")
        if any(monitoring_times[i] <= monitoring_times[i - 1] for i in range(1, len(monitoring_times))):
            raise ValueError("Monitoring times must be strictly increasing.")
        if strike_factor <= 0:
            raise ValueError("Strike factor must be a positive number.")

        self.option_type = option_type
        self.monitoring_times = monitoring_times
        self.strike_factor = strike_factor

    @property
    def maturity_time(self):
        """
        Property representing the maturity time of the option.
        """
        return self.monitoring_times[-1]

    def payoff(self, path):
        """
        Calculate the payoff of the Asian option.
        """
        if not all(t in path for t in self.monitoring_times):
            raise ValueError("Path must contain all monitoring times.")

        average_price = sum(
            path[t] for t in self.monitoring_times) / len(self.monitoring_times)

        if self.option_type == 'call':
            return max(path[self.maturity_time] - self.strike_factor * average_price, 0)
        else:
            return max(self.strike_factor * average_price - path[self.maturity_time], 0)

    def _path_delta(self, price_path: Dict[float, float], risk_free_rate: float) -> float:
        """
        Calculate the sensitivity (Delta) of the option value with respect to changes in the initial asset price.
        """
        missing_intervals = [
            time for time in self.monitoring_times if time not in price_path]
        if missing_intervals:
            raise ValueError(f"Price data missing for scheduled intervals: {
                             missing_intervals}")

        pv_factor = exp(-risk_free_rate * self.maturity_time)
        option_payoff = self.payoff(price_path)

        sensitivity_factors = [self.underlying._dst_ds0(
            price_path, time) for time in self.monitoring_times]
        average_sensitivity = sum(sensitivity_factors) / \
            len(sensitivity_factors)

        delta_value = pv_factor * option_payoff * average_sensitivity
        return delta_value


if __name__ == '__main__':
    # Testing the Asset class
    try:
        asset = Asset(name="Asset1", current_price=100, dividend_yield=0.02)
        print(f"Asset created: {asset.name}, Current Price: {
              asset.current_price}, Dividend Yield: {asset.dividend_yield}")
    except ValueError as e:
        print(f"Error creating Asset: {e}")

    # Testing the BlackScholesAsset class
    try:
        bs_asset = BlackScholesAsset(
            name="BSAsset1", current_price=100, dividend_yield=0.02, volatility=0.3)
        print(
            f"Black-Scholes Asset created: {bs_asset.name}, Volatility: {bs_asset.volatility}")

        # Simulating the asset path
        simulated_times = [0.25, 0.5, 0.75, 1.0]
        interest_rate = 0.05
        path = bs_asset.simulate_path(simulated_times, interest_rate)
        print("Simulated Path:", path)
    except ValueError as e:
        print(f"Error creating BlackScholesAsset: {e}")

    # Testing the AsianOption class
    try:
        monitoring_times = [0.25, 0.5, 0.75, 1.0]
        asian_option = AsianOption(name="AsianCallOption", underlying=bs_asset,
                                   option_type='call', monitoring_times=monitoring_times, strike_factor=1.0)
        print(f"Asian Option created: {asian_option.name}, Type: {
              asian_option.option_type}, Maturity Time: {asian_option.maturity_time}")

        # Calculating the payoff for the simulated path
        payoff = asian_option.payoff(path)
        print(f"Payoff for Asian Option: {payoff}")

        # Monte Carlo Delta calculation
        simulations = 1000
        delta_minus, mean_delta, delta_plus = asian_option.monte_carlo_delta(
            simulations, interest_rate=interest_rate)
        print(f"Delta confidence interval: ({
              delta_minus}, {mean_delta}, {delta_plus})")
    except ValueError as e:
        print(f"Error creating AsianOption: {e}")
