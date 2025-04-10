import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fetch_underlying_price(symbol, period="1y", interval="1d"):
    """
    Fetch historical closing prices for the underlying asset.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        raise ValueError(f"No historical data found for symbol '{symbol}'.")
    hist = hist[['Close']]
    hist.reset_index(inplace=True)
    return hist

def estimate_gbm_parameters(hist_df):
    """
    Estimate the drift (mu) and volatility (sigma) from historical returns using logarithmic returns.
    """
    # Calculate daily log returns
    hist_df['LogReturn'] = np.log(hist_df['Close'] / hist_df['Close'].shift(1))
    hist_df.dropna(inplace=True)
    mu = hist_df['LogReturn'].mean()
    sigma = hist_df['LogReturn'].std()
    return mu, sigma

def forecast_underlying_price(S0, mu, sigma, forecast_horizon=30, n_sim=100):
    """
    Simulate future underlying asset price paths using a geometric Brownian motion model.
    S0: current asset price
    mu: drift (mean daily return)
    sigma: volatility (daily standard deviation of returns)
    forecast_horizon: number of days to forecast
    n_sim: number of simulation paths
    Returns an array of simulated price paths and the corresponding time steps.
    """
    dt = 1  # one day per time unit
    simulation = np.zeros((n_sim, forecast_horizon + 1))
    simulation[:, 0] = S0
    
    for sim in range(n_sim):
        for t in range(1, forecast_horizon + 1):
            epsilon = np.random.normal()  # standard normal random shock
            simulation[sim, t] = simulation[sim, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)
    
    forecast_time = np.arange(forecast_horizon + 1)
    return simulation, forecast_time

def plot_underlying_forecast(hist_df, simulation, forecast_time, symbol):
    """
    Plot historical underlying asset prices along with simulated future price paths.
    The x-axis represents time and the y-axis represents the underlying asset price.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot historical closing prices from the fetched data.
    plt.plot(hist_df['Date'], hist_df['Close'], label='Historical Price', lw=2)
    
    # Prepare forecast dates starting from the last available historical date.
    last_date = hist_df['Date'].iloc[-1]
    forecast_dates = pd.date_range(last_date, periods=len(forecast_time), freq='B')  # using business days
    
    # Plot each simulation path in light gray.
    for path in simulation:
        plt.plot(forecast_dates, path, color='black', alpha=0.3)
    
    # Plot the mean forecast path in blue.
    mean_path = simulation.mean(axis=0)
    plt.plot(forecast_dates, mean_path, color='blue', lw=2, label='Mean Forecast')
    
    # Shade the area between the 5th and 95th percentiles to represent a confidence interval.
    lower_bound = np.percentile(simulation, 5, axis=0)
    upper_bound = np.percentile(simulation, 95, axis=0)
    plt.fill_between(forecast_dates, lower_bound, upper_bound, color='grey', alpha=0.4, label='5-95% Confidence Interval')
    
    plt.xlabel('Date')
    plt.ylabel('Underlying Asset Price (USD)')  # y-axis now represents the underlying asset price
    plt.title(f'Forecast of Underlying Asset Price for {symbol}')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    symbol = input("Ticker Symbol? ").strip().upper()

    forecast_horizon = 30  # number of days to forecast
    n_sim = 200  # number of simulation paths
    
    print(f"Fetching historical data for {symbol}...")
    hist_df = fetch_underlying_price(symbol, period="1y", interval="1d")
    
    print("Estimating drift and volatility from historical data...")
    mu, sigma = estimate_gbm_parameters(hist_df)
    print(f"Estimated daily drift: {mu:.6f}, Estimated daily volatility: {sigma:.6f}")
    
    # Use the last closing price as the starting price for forecasting.
    S0 = hist_df['Close'].iloc[-1]
    
    print("Forecasting future underlying asset prices using GBM simulation...")
    simulation, forecast_time = forecast_underlying_price(S0, mu, sigma, forecast_horizon, n_sim)
    
    print("Plotting forecast paths along with historical prices...")
    plot_underlying_forecast(hist_df, simulation, forecast_time, symbol)

try:
    main()
except Exception as ex:
    print(ex)
    main()
