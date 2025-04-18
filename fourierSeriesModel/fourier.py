import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ───── GLOBAL CONFIGURATION ─────
SYMBOL        = 'SPY'
DATA_START    = '2024-04-18'
FORECAST_DAYS = 30
INTERVAL      = '1d'
# always ends today
DATA_END      = datetime.now().strftime('%Y-%m-%d')  

# ───── FETCH HISTORICAL DATA ─────
df = yf.download(
    SYMBOL,
    start=DATA_START,
    end=DATA_END,
    interval=INTERVAL,
    progress=False
)
prices = df['Close'].values
dates  = df.index
n      = len(prices)

# ───── FFT ON HISTORICAL SERIES ─────
X      = np.fft.fft(prices)
freqs  = np.fft.fftfreq(n, d=1)
mask   = freqs >= 0
freqs_pos = freqs[mask]
X_pos     = X[mask]
ampl      = 2 * np.abs(X_pos) / n
phase     = np.angle(X_pos)

# ───── FUTURE DATES & TIME VECTOR ─────
# Forecast starts the day after the last historical date
start_forecast = dates[-1] + timedelta(days=1)
future_dates   = pd.date_range(start=start_forecast,
                               periods=FORECAST_DAYS,
                               freq='D')
t_future       = np.arange(n, n + FORECAST_DAYS)

# ───── NUMBER OF FORECAST SINUSOIDS ─────
numSine = 10

fig, ax = plt.subplots(figsize=(13, 6))
# sin(0) = 0, sin(date)=0, we can make a sum of sinusoids
all_y_future = np.zeros((numSine, FORECAST_DAYS)) 
ampFlag = False
for k in range(1, numSine+1):
    A_k = ampl[k].item()
    if A_k < .5:
        A_k *= 10
        ampFlag = True
    f_k = float(freqs_pos[k])
    y_future = A_k * np.cos(2 * np.pi * f_k * t_future + phase[k])
    all_y_future[k-1] = y_future
    ax.plot(future_dates, y_future,label=f'k={k}, f={f_k:.4f}, A={A_k:.2f}')
if ampFlag:
    print("Amplitude Scaled by 10")
# Sum of all sinusoids
avg_y_future = np.sum(all_y_future, axis=0)
ax.plot(future_dates, avg_y_future, color='black', linestyle='--', linewidth=2,label='Sum of Sinusoids')

# ───── FORMAT DATE AXIS ─────
locator   = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.set_ylim(-15, 15)
ax.set_title(f'Top {numSine} Fourier Sinusoids for {SYMBOL}')
ax.set_xlabel('Date')
ax.set_ylabel('Amplitude')
ax.grid(True)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
