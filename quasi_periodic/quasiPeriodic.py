import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import yfinance as yf

ticker = input("Ticker Symbol? ") 
start_date = '2025-01-01'
end_date = '2025-04-25'

# Data Collection
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    raise ValueError("No data fetched. Check the ticker symbol and date range.")

# Process the data
data = data[['Close']].copy().reset_index()
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = (data['Date'] - data['Date'].min()).dt.total_seconds() / (24 * 3600)  # Time in days

# Rolling mean and detrend
window_size = 3
rolling_mean = data['Close'].rolling(window=window_size, center=True).mean()
detrended = data['Close'] - rolling_mean
detrended = detrended.bfill().ffill()  # Safely fill NaNs
data['Detrended'] = detrended

# Normalize
mean = data['Detrended'].mean()
std = data['Detrended'].std()
data['Normalized'] = (data['Detrended'] - mean) / std

# Ensure no NaNs
if data['Normalized'].isna().any():
    raise ValueError("NaNs still present in normalized data!")

'''
            _______ MATH _______
                
-Not sure if anyone has used this function 
-Acts like a quasi-periodic fourier series
-Captures irrationality with root(n)*pi
-Cycles are captured by the cos^2(wt)
-Standard Form = cos^2(wt), w = root(n) * PI
-Frequencies are irrationally related
-The summation ensures they blend quasi-periodically

'''
def quasi_periodic_model(t, *a):
    N = len(a)
    result = np.zeros_like(t)
    for n in range(1, N + 1):
        # SUM[1,N+1](A*(cos(root(n)*pi*t))^2)
        result += a[n - 1] * np.cos(np.sqrt(n) * np.pi * t) ** 2
    return result

# Fit the model
t_values = data['Time'].values
y_values = data['Normalized'].values
N_terms = 10
initial_guess = np.ones(N_terms)

params, _ = curve_fit(quasi_periodic_model, t_values, y_values, p0=initial_guess)

# Plots
fitted_values = quasi_periodic_model(t_values, *params)

plt.figure(figsize=(14, 6))
plt.plot(data['Date'], y_values, label='Normalized Data', alpha=0.7)
plt.plot(data['Date'], fitted_values, label='Quasi-Periodic Fit', linestyle='--', linewidth=2)
plt.title(f'Quasi-Periodic Fit to {ticker} Stock Data')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
