# Extensive Imports
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

'''
Fourier Series MODEL
''' 

def render_plot_to_new_window(fig, title="Forecast Plot"):
    # Create a new top-level window
    plot_window = tk.Toplevel(root)
    plot_window.title(title)
    plot_window.geometry("1200x800")
    plot_window.configure(bg="#2E3440")

    # Make it resizable
    plot_window.rowconfigure(0, weight=1)
    plot_window.columnconfigure(0, weight=1)

    # Embed the figure in the new window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.grid(row=0, column=0, sticky="nsew")

    # Optional: add a close button
    close_btn = ttk.Button(plot_window, text="Close", command=plot_window.destroy)
    close_btn.grid(row=1, column=0, pady=10)


def fourierSeries(SYMBOL = 'SPY', FORECAST_DAYS = 100, numSine = 15):
    # Definitions
    DATA_START    = '2024-04-18'
    INTERVAL      = '1d'
    # always ends today
    DATA_END      = datetime.now().strftime('%Y-%m-%d')  

    # * FETCH HISTORICAL DATA *
    df = yf.download(SYMBOL,start=DATA_START,end=DATA_END,interval=INTERVAL,progress=False, auto_adjust=True)
    prices = df['Close'].values
    dates  = df.index
    n      = len(prices)

    # * FFT ON HISTORICAL SERIES *
    X      = np.fft.fft(prices)
    freqs  = np.fft.fftfreq(n, d=1)
    mask   = freqs >= 0
    freqs_pos = freqs[mask]
    X_pos     = X[mask]
    ampl      = 2 * np.abs(X_pos) / n
    phase     = np.angle(X_pos)

    # * FUTURE DATES & TIME VECTOR *
    # Forecast starts the day after the last historical date
    start_forecast = dates[-1] + timedelta(days=1)
    future_dates   = pd.date_range(start=start_forecast,
                                periods=FORECAST_DAYS,
                                freq='D')
    t_future       = np.arange(n, n + FORECAST_DAYS)

    # * PLOTTING *
    fig,ax = plt.subplots(figsize=(10, 7))
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
    
    # * FORMAT DATE AXIS *
    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylim(-25, 25)
    if numSine >= 20:
        ax.set_ylim(-50, 50)
    else:
        ax.legend(loc='upper right')
    ax.set_title(f'Top {numSine} Fourier Sinusoids for {SYMBOL}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    plt.tight_layout()
    render_plot_to_new_window(fig, title="Fourier Series")
    return
"""
Volatility Model
"""
def compute_features(data):
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['vol_5d'] = data['log_return'].rolling(5).std() * np.sqrt(252)
    data['vol_10d'] = data['log_return'].rolling(10).std() * np.sqrt(252)
    data['rsi'] = 100 - (100 / (1 + data['log_return'].rolling(14).mean() / data['log_return'].rolling(14).std()))
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_10'] = data['Close'].rolling(10).mean()
    data['sma_ratio'] = data['sma_5'] / data['sma_10']
    return data


def create_volatility_target(data, horizon=5):
    future_vol = data['log_return'].rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    data['future_vol'] = future_vol
    return data.dropna()


def train_volatility_model(df):
    features = ['vol_5d', 'vol_10d', 'rsi', 'sma_ratio']
    X = df[features]
    y = df['future_vol']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Volatility Prediction MSE: {mse:.6f}")

    # Start plotting
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Prediction plot
    test_dates = y_test.index
    ax[0].plot(test_dates, y_test.values, label='True Volatility', color='blue', marker='o')
    ax[0].plot(test_dates, preds, label='Predicted Volatility', color='red', linestyle='--', marker='x')
    ax[0].set_title('Volatility Prediction: True vs Predicted')
    ax[0].set_ylabel('Annualized Volatility')
    ax[0].legend()
    ax[0].grid(True)

    # 5-day volatility
    ax[1].plot(df.index, df['vol_5d'], label='5-Day Volatility', color='purple')
    ax[1].set_title('5-Day Annualized Volatility')
    ax[1].set_ylabel('Volatility')
    ax[1].legend()
    ax[1].grid(True)

    # 10-day volatility
    ax[2].plot(df.index, df['vol_10d'], label='10-Day Volatility', color='green')
    ax[2].set_title('10-Day Annualized Volatility')
    ax[2].set_ylabel('Volatility')
    ax[2].legend()
    ax[2].grid(True)

    # SMA ratio
    ax[3].plot(df.index, df['sma_ratio'], label='SMA Ratio (5-day / 10-day)', color='orange')
    ax[3].set_title('SMA Ratio')
    ax[3].set_ylabel('Ratio')
    ax[3].set_xlabel('Date')
    ax[3].legend()
    ax[3].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Render to GUI
    render_plot_to_new_window(fig, title="Volatility Model")

    return model, features


def fetch_and_train(ticker):
    try:
        hist = yf.Ticker(ticker).history(period='1y')
        if hist.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        df = compute_features(hist)
        df = create_volatility_target(df)
        return train_volatility_model(df)
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Data Error", str(e)))


def volatility(symbol='SPY'):
    fetch_and_train(symbol)


'''
SHOCK MODEL
'''
def fetch_option_chain(symbol):
    ticker = yf.Ticker(symbol)
    if not ticker.options:
        raise ValueError(f"No option chain found for symbol '{symbol}'. Please check the symbol or try a different one.")
    
    expirations = ticker.options
    print(f"Available expiration dates for {symbol}: {expirations}")

    options_data = []
    for expire in expirations:
        try:
            opt_chain = ticker.option_chain(expire)
            calls = opt_chain.calls
            calls['expiration'] = pd.to_datetime(expire)
            options_data.append(calls)
        except Exception as e:
            print(f"Warning: Failed to fetch options for expiration {expire}: {e}")

    if not options_data:
        raise ValueError(f"Failed to retrieve any valid option data for symbol '{symbol}'.")
        
    return pd.concat(options_data, ignore_index=True)

def process_option_data(df, max_days_to_expire=400):
    today = pd.Timestamp.now().normalize()
    df['time_to_expire'] = (df['expiration'] - today).dt.days
    df = df.dropna(subset=['strike', 'lastPrice', 'time_to_expire'])
    return df[df['time_to_expire'] <= max_days_to_expire]

def create_surface_grid(df, grid_res=100):
    strike_min, strike_max = df['strike'].min(), df['strike'].max()
    expire_min, expire_max = df['time_to_expire'].min(), df['time_to_expire'].max()
    grid_x, grid_y = np.meshgrid(np.linspace(strike_min, strike_max, grid_res),
                                 np.linspace(expire_min, expire_max, grid_res))
    points = df[['strike', 'time_to_expire']].values
    values = df['lastPrice'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    # Replace NaNs with the overall mean
    grid_z[np.isnan(grid_z)] = np.nanmean(grid_z)
    return grid_x, grid_y, grid_z

def simulate_shock_and_spread(grid_x, grid_y, grid_z, smoothing=2.0, scale_factor=5.0, shock_fraction=0.1, iterations=500):
    heightmap = gaussian_filter(grid_z, sigma=smoothing)
    shock_field = np.zeros_like(heightmap)
    num_shock_points = int(shock_field.size * shock_fraction)
    shock_points = np.random.choice(shock_field.size, num_shock_points, replace=False)
    np.put(shock_field, shock_points, 1.0)  # Inject shocks

    # Instead of iterative diffusion, use an effective sigma:
    effective_sigma = np.sqrt(iterations)
    shock_field = gaussian_filter(shock_field, sigma=effective_sigma)

    # Normalize and scale the shock field
    shock_field = (shock_field - shock_field.min()) / (shock_field.max() - shock_field.min())
    return shock_field * scale_factor

def plot_surface_with_shocks(grid_x, grid_y, grid_z, shock_field, symbol):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surface_colors = plt.cm.viridis((grid_z - grid_z.min()) / (grid_z.max() - grid_z.min()))
    shock_colors = plt.cm.Reds(shock_field / shock_field.max())

    ax.plot_surface(grid_x, grid_y, grid_z, facecolors=surface_colors, rstride=1, cstride=1, antialiased=True)
    ax.plot_surface(grid_x, grid_y, grid_z, facecolors=shock_colors, rstride=1, cstride=1, alpha=0.5, antialiased=False)
    cp = ax.contourf(grid_x, grid_y, grid_z, cmap='Reds', levels=100)
    plt.colorbar(cp, ax=ax, label='Shock Field Strength')

    ax.set_xlabel('Underlying Price (USD)')
    ax.set_ylabel('Time to Expire (days)')
    ax.set_zlabel('Call Option Price (USD)')
    ax.set_title(f'Option Price Surface with Simulated Shock Channels for {symbol}')

    render_plot_to_new_window(fig, title="Option Shock Surface")


def stochastic_gradient_descent(grid_x, grid_y, grid_z, shock_field, start_point,learning_rate=0.1, noise_scale=0.02, iterations=100, iteration_time_unit=1):
    dz_dy, dz_dx = np.gradient(grid_z)
    x_vals = grid_x[0, :]
    y_vals = grid_y[:, 0]
    grad_x_interp = RectBivariateSpline(y_vals, x_vals, dz_dx)
    grad_y_interp = RectBivariateSpline(y_vals, x_vals, dz_dy)
    shock_interp = RectBivariateSpline(y_vals, x_vals, shock_field)
    
    path = [start_point]
    current_point = np.array(start_point, dtype=float)
    time_record = [0]
    
    for _ in range(iterations):
        grad_x_val = grad_x_interp(current_point[1], current_point[0])[0][0]
        grad_y_val = grad_y_interp(current_point[1], current_point[0])[0][0]
        local_shock = shock_interp(current_point[1], current_point[0])[0][0]
        effective_lr = learning_rate * (1 + local_shock)
        noise = np.random.normal(0, noise_scale, size=2)
        update = effective_lr * np.array([grad_x_val, grad_y_val]) + noise
        current_point = current_point - update
        path.append(current_point.copy())
        time_record.append(time_record[-1] + iteration_time_unit)
    
    return np.array(path), np.array(time_record)

def plot_descent_path(grid_x, grid_y, grid_z, descent_path, symbol):
    fig, ax = plt.subplots(figsize=(12, 8))
    cp = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=100)
    plt.colorbar(cp, ax=ax, label='Call Option Price (USD)')

    ax.plot(descent_path[:, 0], descent_path[:, 1], marker='o', color='red', linewidth=2, markersize=4)
    ax.set_xlabel('Underlying Price (USD)')
    ax.set_ylabel('Time to Expire (days)')
    ax.set_title(f'Stochastic Gradient Descent Path on the Option Price Manifold for {symbol}')

    render_plot_to_new_window(fig, title="Gradient Descent Path")


def forecast_option_price(grid_x, grid_y, grid_z, shock_field, start_point,forecast_horizon=30, n_sim=100, learning_rate=0.1, noise_scale=0.02):
    x_vals = grid_x[0, :]
    y_vals = grid_y[:, 0]
    dz_dy, dz_dx = np.gradient(grid_z)
    grad_x_interp = RectBivariateSpline(y_vals, x_vals, dz_dx)
    grad_y_interp = RectBivariateSpline(y_vals, x_vals, dz_dy)
    shock_interp = RectBivariateSpline(y_vals, x_vals, shock_field)
    
    forecast_paths = np.zeros((n_sim, forecast_horizon + 1, 2))
    forecast_paths[:, 0, :] = start_point
    
    for sim in range(n_sim):
        current_point = np.array(start_point, dtype=float)
        for day in range(1, forecast_horizon + 1):
            grad_x_val = grad_x_interp(current_point[1], current_point[0])[0][0]
            grad_y_val = grad_y_interp(current_point[1], current_point[0])[0][0]
            local_shock = shock_interp(current_point[1], current_point[0])[0][0]
            effective_lr = learning_rate * (1 + local_shock)
            noise = np.random.normal(0, noise_scale, size=2)
            update = effective_lr * np.array([grad_x_val, grad_y_val]) + noise
            current_point = current_point + update
            forecast_paths[sim, day, :] = current_point.copy()
    
    forecast_time = np.arange(forecast_horizon + 1)
    return forecast_paths, forecast_time

def plot_forecast_paths(forecast_paths, forecast_time, symbol):
    n_sim = forecast_paths.shape[0]
    price_paths = forecast_paths[:, :, 0]
    mean_price = np.mean(price_paths, axis=0)
    lower_bound = np.percentile(price_paths, 5, axis=0)
    upper_bound = np.percentile(price_paths, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(n_sim):
        ax.plot(forecast_time, price_paths[i, :], color='black', alpha=0.3)
    ax.plot(forecast_time, mean_price, color='blue', lw=2, label='Mean Forecast')
    ax.fill_between(forecast_time, lower_bound, upper_bound, color='grey', alpha=0.6, label='5-95% Confidence Interval')

    ax.set_xlabel('Forecast Time (days)')
    ax.set_ylabel('Asset Price (USD)')
    ax.set_title(f'Forecast of Future Asset Price Dynamics for {symbol}')
    ax.legend()

    render_plot_to_new_window(fig, title="Forecast Path")


def shockModel(symbol = 'SPY'):
    try:
        call_options_df = fetch_option_chain(symbol)
    except Exception as e:
        print(e)
        symbol = input("Ticker Symbol? ")
        call_options_df = fetch_option_chain(symbol)
    
    print(f"Processing option data for {symbol}...")
    processed_df = process_option_data(call_options_df, max_days_to_expire=200)
    
    print("Creating option price manifold...")
    grid_x, grid_y, grid_z = create_surface_grid(processed_df, grid_res=100)
    
    print("Simulating shock events and their spread on the manifold...")
    shock_field = simulate_shock_and_spread(grid_x, grid_y, grid_z, shock_fraction=0.2, iterations=500)
    
    print("Plotting option price manifold with shock channels...")
    plot_surface_with_shocks(grid_x, grid_y, grid_z, shock_field, symbol)
    
    # Choose a starting point near the middle of the grid
    start_strike = grid_x[0, grid_x.shape[1] // 2]
    start_expire = grid_y[grid_y.shape[0] // 2, 0]
    start_point = [start_strike, start_expire]

    print(f"Forecasting future underlying price dynamics for {symbol}...")
    forecast_horizon = 30
    n_sim = 100
    forecast_paths, forecast_time = forecast_option_price(
        grid_x, grid_y, grid_z, shock_field, start_point,
        forecast_horizon=forecast_horizon, n_sim=n_sim, learning_rate=0.01, noise_scale=0.02)
    plot_forecast_paths(forecast_paths, forecast_time, symbol)
    return
'''
Geometric Brownian Motion
'''
def fetch_underlying_price(symbol, period="1y", interval="1d"):
    
    # Fetch historical closing prices for the underlying asset.
   
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        raise ValueError(f"No historical data found for symbol '{symbol}'.")
    hist = hist[['Close']]
    hist.reset_index(inplace=True)
    return hist

def estimate_gbm_parameters(hist_df):
    
    # Estimate the drift (mu) and volatility (sigma) from historical returns using logarithmic returns.
    
    # Calculate daily log returns
    hist_df['LogReturn'] = np.log(hist_df['Close'] / hist_df['Close'].shift(1))
    hist_df.dropna(inplace=True)
    mu = hist_df['LogReturn'].mean()
    sigma = hist_df['LogReturn'].std()
    return mu, sigma

def forecast_underlying_price(S0, mu, sigma, forecast_horizon=30, n_sim=100):
    '''
    S0: current asset price
    mu: drift (mean daily return)
    sigma: volatility (daily standard deviation of returns)
    forecast_horizon: number of days to forecast
    n_sim: number of simulation paths
    Returns an array of simulated price paths and the corresponding time steps.
    '''
    
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
    # Prepare forecast dates
    last_date = hist_df['Date'].iloc[-1]
    forecast_dates = pd.date_range(last_date, periods=len(forecast_time), freq='B')

    # Create a Matplotlib Figure instead of using plt.figure/show
    fig, ax = plt.subplots(figsize=(12, 8))

    # Historical
    ax.plot(hist_df['Date'], hist_df['Close'], label='Historical Price', lw=2)

    # Simulation paths
    for path in simulation:
        ax.plot(forecast_dates, path, color='black', alpha=0.3)
    mean_path = simulation.mean(axis=0)
    ax.plot(forecast_dates, mean_path, color='blue', lw=2, label='Mean Forecast')

    # Confidence band
    lower = np.percentile(simulation, 5, axis=0)
    upper = np.percentile(simulation, 95, axis=0)
    ax.fill_between(forecast_dates, lower, upper, color='grey', alpha=0.4, label='5-95% CI')

    ax.set_xlabel('Date')
    ax.set_ylabel('Underlying Asset Price (USD)')
    ax.set_title(f'Geometric Brownian Forecast of {symbol}')
    ax.legend()
    ax.grid(True)

    # send it to its own Tk window
    render_plot_to_new_window(fig, title=f"{symbol} GBM Forecast")


def gasDiffusion(symbol):
    symbol = symbol.strip().upper()
    forecast_horizon = 30  # number of days to forecast
    n_sim = 50  # number of simulation paths
    
    print(f"Fetching historical data for {symbol}...")
    hist_df = fetch_underlying_price(symbol, period="1y", interval="1d")
    
    print("Estimating drift and volatility from historical data...")
    mu, sigma = estimate_gbm_parameters(hist_df)
    print(f"Estimated daily drift: {mu:.6f}, Estimated daily volatility: {sigma:.6f}")
    
    # Use the last closing price as the starting price for forecasting.
    S0 = hist_df['Close'].iloc[-1]
    
    print("Geometric Brownian Motion Simulation...")
    simulation, forecast_time = forecast_underlying_price(S0, mu, sigma, forecast_horizon, n_sim)
    
    print("Plotting forecast paths along with historical prices...")
    plot_underlying_forecast(hist_df, simulation, forecast_time, symbol)

    # Function to run the selected model based on the choice
def run_model():
    try:
        model_choice = model_var.get()  # Get the selected model
        symbol = ticker_entry.get()  # Get the ticker symbol
        
        if model_choice == "Fourier Series":
            fourierSeries(symbol)
        elif model_choice == "Implied Volatility":
            volatility(symbol)
        elif model_choice == "Shock Response":
            shockModel(symbol)
        elif model_choice == "Geometric Brownian Motion":
            gasDiffusion(symbol)
        else:
            messagebox.showerror("Selection Error", "Please select a valid forecasting model.")
    except Exception as ex:
        messagebox.showerror("Error", str(ex))


# Function to run the model in a separate thread
def start_model_in_thread():
    try:
        threading.Thread(target=run_model, daemon=True).start()
    except Exception as thread_ex:
        print(f"Thread launch failed: {thread_ex}")


def main():
    global root, frame, model_var, ticker_entry

    # --- GUI SETUP ---
    root = tk.Tk()
    root.title("📈 Forecasting Models")
    root.geometry("1000x700")
    root.configure(bg="#2E3440")  # Dark background

    # Apply a custom ttk theme
    style = ttk.Style(root)
    style.theme_use("clam")

    # Style configurations
    style.configure("TFrame", background="#2E3440")
    style.configure("TLabel", background="#2E3440", foreground="#ECEFF4", font=("Helvetica", 20, "bold"))
    style.configure("TRadiobutton", background="#3B4252", foreground="#ECEFF4", font=("Helvetica", 16))
    style.map("TRadiobutton",
              background=[('active', '#434C5E')],
              foreground=[('selected', '#88C0D0')])
    style.configure("TEntry", fieldbackground="#ECEFF4", background="#ECEFF4", foreground="#2E3440", font=("Helvetica", 16))
    style.configure("TButton", background="#81A1C1", foreground="#2E3440", font=("Helvetica", 16, "bold"), padding=10)
    style.map("TButton", background=[('active', '#5E81AC')])

    # Main container
    frame = ttk.Frame(root, padding=(50, 30, 50, 30), style="TFrame")
    frame.place(relx=0.5, rely=0.5, anchor="center")

    # Title
    ttk.Label(frame, text="Select Forecasting Model:").grid(row=0, column=0, sticky=tk.W, pady=(0, 20))

    # Model options
    model_var = tk.StringVar(value="Geometric Brownian Motion")
    options = ["Fourier Series", "Implied Volatility", "Shock Response", "Geometric Brownian Motion"]

    for i, opt in enumerate(options, start=1):
        ttk.Radiobutton(frame, text=opt, variable=model_var, value=opt, style="TRadiobutton") \
            .grid(row=i, column=0, sticky=tk.W, pady=5)

    # Ticker symbol input
    ttk.Label(frame, text="Ticker Symbol:").grid(row=6, column=0, sticky=tk.W, pady=(30, 5))
    ticker_entry = ttk.Entry(frame, width=60)
    ticker_entry.grid(row=7, column=0, pady=5)
    ticker_entry.focus()

    # Run button (calls run_model_in_thread when clicked)
    run_button = ttk.Button(frame, text="Run Model", command=run_model)
    run_button.grid(row=8, column=0, pady=(40, 0))

    # Teardown once window closed
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.mainloop()

main()
