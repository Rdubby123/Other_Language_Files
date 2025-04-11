#!/usr/bin/env python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# Additional imports for the GUI
import tkinter as tk
from tkinter import ttk, messagebox
import threading

########################################
# MODEL 1: GBM Forecasting
########################################

def model_gbm(symbol):
    """Model 1: Forecast Underlying Price using Geometric Brownian Motion."""
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
        # Estimate the drift (mu) and volatility (sigma) from historical returns (log returns).
        hist_df['LogReturn'] = np.log(hist_df['Close'] / hist_df['Close'].shift(1))
        hist_df.dropna(inplace=True)
        mu = hist_df['LogReturn'].mean()
        sigma = hist_df['LogReturn'].std()
        return mu, sigma

    def forecast_underlying_price_func(S0, mu, sigma, forecast_horizon=30, n_sim=100):
        # Simulate future underlying asset price paths using a GBM model.
        dt = 1  # one day per time unit
        simulation = np.zeros((n_sim, forecast_horizon + 1))
        simulation[:, 0] = S0
        for sim in range(n_sim):
            for t in range(1, forecast_horizon + 1):
                epsilon = np.random.normal()  # standard normal shock
                simulation[sim, t] = simulation[sim, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)
        forecast_time = np.arange(forecast_horizon + 1)
        return simulation, forecast_time

    def plot_underlying_forecast(hist_df, simulation, forecast_time, symbol):
        # Plot historical underlying prices along with simulated future price paths.
        plt.figure(figsize=(12, 8))
        plt.plot(hist_df['Date'], hist_df['Close'], label='Historical Price', lw=2)
        # Prepare forecast dates
        last_date = hist_df['Date'].iloc[-1]
        forecast_dates = pd.date_range(last_date, periods=len(forecast_time), freq='B')  # business days
        # Plot each simulation in light gray.
        for path in simulation:
            plt.plot(forecast_dates, path, color='black', alpha=0.3)
        # Plot the mean forecast path.
        mean_path = simulation.mean(axis=0)
        plt.plot(forecast_dates, mean_path, color='blue', lw=2, label='Mean Forecast')
        # Add confidence interval shading.
        lower_bound = np.percentile(simulation, 5, axis=0)
        upper_bound = np.percentile(simulation, 95, axis=0)
        plt.fill_between(forecast_dates, lower_bound, upper_bound, color='grey', alpha=0.4, label='5-95% Confidence Interval')
        plt.xlabel('Date')
        plt.ylabel('Underlying Asset Price (USD)')
        plt.title(f'Forecast of Underlying Asset Price for {symbol}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Execution of Model 1:
    forecast_horizon = 30  # days to forecast
    n_sim = 50  # simulation paths

    hist_df = fetch_underlying_price(symbol, period="1y", interval="1d")
    mu, sigma = estimate_gbm_parameters(hist_df)
    S0 = hist_df['Close'].iloc[-1]
    simulation, forecast_time = forecast_underlying_price_func(S0, mu, sigma, forecast_horizon, n_sim)
    plot_underlying_forecast(hist_df, simulation, forecast_time, symbol)

########################################
# MODEL 2: River Flow Forecasting
########################################

def model_river(symbol):
    """Model 2: Forecast Option Price Surface using Simulated Rainfall and River Flow."""
    def fetch_option_chain(symbol):
        ticker = yf.Ticker(symbol)
        if not ticker.options:
            raise ValueError(f"No option chain found for symbol '{symbol}'.")
        expirations = ticker.options
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
            raise ValueError(f"Failed to retrieve valid option data for symbol '{symbol}'.")
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
        grid_z[np.isnan(grid_z)] = np.nanmean(grid_z)
        return grid_x, grid_y, grid_z

    def simulate_rain_and_flow(grid_x, grid_y, grid_z, smoothing=2.0, scale_factor=5.0, rain_fraction=0.1):
        # Create a heightmap with smoothing.
        heightmap = gaussian_filter(grid_z, sigma=smoothing)
        grad_y, grad_x = np.gradient(-heightmap)
        water_flow = np.zeros_like(heightmap)
        rainfall = np.zeros_like(heightmap)
        num_rain_points = int(rainfall.size * rain_fraction)
        rain_points = np.random.choice(np.arange(rainfall.size), num_rain_points, replace=False)
        np.put(rainfall, rain_points, 1.0)
        iterations = 500  
        for _ in range(iterations):
            water_flow += rainfall
            rainfall_shifted = np.roll(rainfall, shift=1, axis=0) * (grad_y > 0)
            rainfall_shifted += np.roll(rainfall, shift=-1, axis=0) * (grad_y < 0)
            rainfall_shifted += np.roll(rainfall, shift=1, axis=1) * (grad_x > 0)
            rainfall_shifted += np.roll(rainfall, shift=-1, axis=1) * (grad_x < 0)
            rainfall = rainfall_shifted / 4.0
        flow_normalized = (water_flow - np.min(water_flow)) / (np.max(water_flow) - np.min(water_flow))
        flow_scaled = flow_normalized * scale_factor
        return flow_scaled

    def plot_surface_with_rivers(grid_x, grid_y, grid_z, flow_accumulation, symbol):
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        surface_colors = plt.cm.viridis((grid_z - np.min(grid_z)) / (np.max(grid_z) - np.min(grid_z)))
        flow_colors = plt.cm.Blues(flow_accumulation)
        ax.plot_surface(grid_x, grid_y, grid_z, facecolors=surface_colors, rstride=1, cstride=1, antialiased=True)
        ax.plot_surface(grid_x, grid_y, grid_z, facecolors=flow_colors, rstride=1, cstride=1, alpha=0.5, antialiased=False)
        ax.set_xlabel('Underlying Price (USD)')
        ax.set_ylabel('Time to Expire (days)')
        ax.set_zlabel('Call Option Price (USD)')
        ax.set_title(f'Option Price Surface with Simulated River Channels for {symbol}')
        plt.show()

    def stochastic_gradient_descent_river(grid_x, grid_y, grid_z, flow_accumulation, start_point,
                                            learning_rate=0.1, noise_scale=0.02, iterations=100, iteration_time_unit=1):
        dz_dy, dz_dx = np.gradient(grid_z)
        x_vals = grid_x[0, :]
        y_vals = grid_y[:, 0]
        grad_x_interp = RectBivariateSpline(y_vals, x_vals, dz_dx)
        grad_y_interp = RectBivariateSpline(y_vals, x_vals, dz_dy)
        flow_interp = RectBivariateSpline(y_vals, x_vals, flow_accumulation)
        path = [start_point]
        current_point = np.array(start_point, dtype=float)
        for i in range(iterations):
            grad_x_val = grad_x_interp(current_point[1], current_point[0])[0][0]
            grad_y_val = grad_y_interp(current_point[1], current_point[0])[0][0]
            local_flow = flow_interp(current_point[1], current_point[0])[0][0]
            effective_lr = learning_rate * (1 + local_flow)
            noise = np.random.normal(0, noise_scale, size=2)
            update = effective_lr * np.array([grad_x_val, grad_y_val]) + noise
            current_point = current_point - update
            path.append(current_point.copy())
        return np.array(path)

    def plot_descent_path(grid_x, grid_y, grid_z, descent_path, symbol):
        fig, ax = plt.subplots(figsize=(12, 8))
        cp = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=50)
        plt.colorbar(cp, ax=ax, label='Call Option Price (USD)')
        ax.plot(descent_path[:, 0], descent_path[:, 1], marker='o', color='red', linewidth=2, markersize=4)
        ax.set_xlabel('Underlying Price (USD)')
        ax.set_ylabel('Time to Expire (days)')
        ax.set_title(f'Stochastic Gradient Descent Path on the Option Price Manifold for {symbol}')
        plt.show()

    def forecast_option_price_river(grid_x, grid_y, grid_z, flow_accumulation, start_point,
                                    forecast_horizon=30, n_sim=100, learning_rate=0.01, noise_scale=0.02):
        x_vals = grid_x[0, :]
        y_vals = grid_y[:, 0]
        dz_dy, dz_dx = np.gradient(grid_z)
        grad_x_interp = RectBivariateSpline(y_vals, x_vals, dz_dx)
        grad_y_interp = RectBivariateSpline(y_vals, x_vals, dz_dy)
        flow_interp = RectBivariateSpline(y_vals, x_vals, flow_accumulation)
        forecast_paths = np.zeros((n_sim, forecast_horizon+1, 2))
        forecast_paths[:, 0, :] = start_point
        for sim in range(n_sim):
            current_point = np.array(start_point, dtype=float)
            for day in range(1, forecast_horizon+1):
                grad_x_val = grad_x_interp(current_point[1], current_point[0])[0][0]
                grad_y_val = grad_y_interp(current_point[1], current_point[0])[0][0]
                local_flow = flow_interp(current_point[1], current_point[0])[0][0]
                effective_lr = learning_rate * (1 + local_flow)
                noise = np.random.normal(0, noise_scale, size=2)
                update = effective_lr * np.array([grad_x_val, grad_y_val]) + noise
                current_point = current_point + update
                forecast_paths[sim, day, :] = current_point.copy()
        forecast_time = np.arange(forecast_horizon+1)
        return forecast_paths, forecast_time

    def plot_forecast_paths_river(forecast_paths, forecast_time, symbol):
        n_sim = forecast_paths.shape[0]
        price_paths = forecast_paths[:, :, 0]
        mean_price = np.mean(price_paths, axis=0)
        lower_bound = np.percentile(price_paths, 5, axis=0)
        upper_bound = np.percentile(price_paths, 95, axis=0)
        plt.figure(figsize=(12, 8))
        for i in range(n_sim):
            plt.plot(forecast_time, price_paths[i, :], color='black', alpha=0.3)
        plt.plot(forecast_time, mean_price, color='blue', lw=2, label='Mean Forecast')
        plt.fill_between(forecast_time, lower_bound, upper_bound, color='grey', alpha=0.6, label='5-95% Confidence Interval')
        plt.xlabel('Forecast Time, T (days)')
        plt.ylabel('Asset Price, S (USD)')
        plt.title(f'Forecast of Future Asset Price Dynamics Based on Option Price Surface for {symbol}')
        plt.legend()
        plt.show()

    # Execution of Model 2:
    call_options_df = fetch_option_chain(symbol)
    processed_df = process_option_data(call_options_df, max_days_to_expire=200)
    grid_x, grid_y, grid_z = create_surface_grid(processed_df, grid_res=100)
    flow_accumulation = simulate_rain_and_flow(grid_x, grid_y, grid_z, rain_fraction=0.2)
    plot_surface_with_rivers(grid_x, grid_y, grid_z, flow_accumulation, symbol)
    # Choose a starting point near the middle of the grid.
    start_strike = grid_x[0, grid_x.shape[1] // 2]
    start_expire = grid_y[grid_y.shape[0] // 2, 0]
    start_point = [start_strike, start_expire]
    descent_path = stochastic_gradient_descent_river(
        grid_x, grid_y, grid_z, flow_accumulation, start_point,
        learning_rate=0.001, noise_scale=0.02, iterations=1000, iteration_time_unit=1)
    plot_descent_path(grid_x, grid_y, grid_z, descent_path, symbol)
    forecast_horizon = 30
    n_sim = 100
    forecast_paths, forecast_time = forecast_option_price_river(
        grid_x, grid_y, grid_z, flow_accumulation, start_point,
        forecast_horizon=forecast_horizon, n_sim=n_sim, learning_rate=0.01, noise_scale=0.02)
    plot_forecast_paths_river(forecast_paths, forecast_time, symbol)

########################################
# MODEL 3: Shock Forecasting
########################################

def model_shock(symbol):
    """Model 3: Forecast Option Price Surface using Simulated Shock Events."""
    def fetch_option_chain(symbol):
        ticker = yf.Ticker(symbol)
        if not ticker.options:
            raise ValueError(f"No option chain found for symbol '{symbol}'.")
        expirations = ticker.options
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
            raise ValueError(f"Failed to retrieve valid option data for symbol '{symbol}'.")
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
        grid_z[np.isnan(grid_z)] = np.nanmean(grid_z)
        return grid_x, grid_y, grid_z

    def simulate_shock_and_spread(grid_x, grid_y, grid_z, smoothing=2.0, scale_factor=5.0, shock_fraction=0.1, iterations=500):
        # Simulate shock events on the option price surface.
        heightmap = gaussian_filter(grid_z, sigma=smoothing)
        shock_field = np.zeros_like(heightmap)
        num_shock_points = int(shock_field.size * shock_fraction)
        shock_points = np.random.choice(shock_field.size, num_shock_points, replace=False)
        np.put(shock_field, shock_points, 1.0)
        effective_sigma = np.sqrt(iterations)
        shock_field = gaussian_filter(shock_field, sigma=effective_sigma)
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
        plt.show()

    def stochastic_gradient_descent_shock(grid_x, grid_y, grid_z, shock_field, start_point,
                                            learning_rate=0.1, noise_scale=0.02, iterations=100, iteration_time_unit=1):
        dz_dy, dz_dx = np.gradient(grid_z)
        x_vals = grid_x[0, :]
        y_vals = grid_y[:, 0]
        grad_x_interp = RectBivariateSpline(y_vals, x_vals, dz_dx)
        grad_y_interp = RectBivariateSpline(y_vals, x_vals, dz_dy)
        shock_interp = RectBivariateSpline(y_vals, x_vals, shock_field)
        path = [start_point]
        current_point = np.array(start_point, dtype=float)
        for _ in range(iterations):
            grad_x_val = grad_x_interp(current_point[1], current_point[0])[0][0]
            grad_y_val = grad_y_interp(current_point[1], current_point[0])[0][0]
            local_shock = shock_interp(current_point[1], current_point[0])[0][0]
            effective_lr = learning_rate * (1 + local_shock)
            noise = np.random.normal(0, noise_scale, size=2)
            update = effective_lr * np.array([grad_x_val, grad_y_val]) + noise
            current_point = current_point - update
            path.append(current_point.copy())
        return np.array(path)

    def plot_descent_path(grid_x, grid_y, grid_z, descent_path, symbol):
        fig, ax = plt.subplots(figsize=(12, 8))
        cp = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=100)
        plt.colorbar(cp, ax=ax, label='Call Option Price (USD)')
        ax.plot(descent_path[:, 0], descent_path[:, 1], marker='o', color='red', linewidth=2, markersize=4)
        ax.set_xlabel('Underlying Price (USD)')
        ax.set_ylabel('Time to Expire (days)')
        ax.set_title(f'Stochastic Gradient Descent Path on the Option Price Manifold for {symbol}')
        plt.show()

    def forecast_option_price_shock(grid_x, grid_y, grid_z, shock_field, start_point,
                                    forecast_horizon=30, n_sim=100, learning_rate=0.01, noise_scale=0.02):
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

    def plot_forecast_paths_shock(forecast_paths, forecast_time, symbol):
        n_sim = forecast_paths.shape[0]
        price_paths = forecast_paths[:, :, 0]
        mean_price = np.mean(price_paths, axis=0)
        lower_bound = np.percentile(price_paths, 5, axis=0)
        upper_bound = np.percentile(price_paths, 95, axis=0)
        plt.figure(figsize=(12, 8))
        for i in range(n_sim):
            plt.plot(forecast_time, price_paths[i, :], color='black', alpha=0.3)
        plt.plot(forecast_time, mean_price, color='blue', lw=2, label='Mean Forecast')
        plt.fill_between(forecast_time, lower_bound, upper_bound, color='grey', alpha=0.6, label='5-95% Confidence Interval')
        plt.xlabel('Forecast Time (days)')
        plt.ylabel('Asset Price (USD)')
        plt.title(f'Forecast of Future Asset Price Dynamics for {symbol}')
        plt.legend()
        plt.show()

    # Execution of Model 3:
    call_options_df = fetch_option_chain(symbol)
    processed_df = process_option_data(call_options_df, max_days_to_expire=200)
    grid_x, grid_y, grid_z = create_surface_grid(processed_df, grid_res=100)
    shock_field = simulate_shock_and_spread(grid_x, grid_y, grid_z, shock_fraction=0.2, iterations=500)
    plot_surface_with_shocks(grid_x, grid_y, grid_z, shock_field, symbol)
    start_strike = grid_x[0, grid_x.shape[1] // 2]
    start_expire = grid_y[grid_y.shape[0] // 2, 0]
    start_point = [start_strike, start_expire]
    descent_path = stochastic_gradient_descent_shock(
        grid_x, grid_y, grid_z, shock_field, start_point,
        learning_rate=0.01, noise_scale=0.02, iterations=1000, iteration_time_unit=1)
    plot_descent_path(grid_x, grid_y, grid_z, descent_path, symbol)
    forecast_horizon = 30
    n_sim = 100
    forecast_paths, forecast_time = forecast_option_price_shock(
        grid_x, grid_y, grid_z, shock_field, start_point,
        forecast_horizon=forecast_horizon, n_sim=n_sim, learning_rate=0.01, noise_scale=0.02)
    plot_forecast_paths_shock(forecast_paths, forecast_time, symbol)

########################################
# GUI Implementation using Tkinter
########################################

def run_forecasting_model():
    """Callback function invoked when the user clicks the 'Run Model' button."""
    model_choice = model_var.get()
    symbol = ticker_entry.get().strip().upper()
    if not symbol:
        messagebox.showerror("Input Error", "Please enter a ticker symbol.")
        return

    # Run the selected model in a separate thread so that the GUI remains responsive.
    def run_model():
        try:
            if model_choice == "GBM":
                model_gbm(symbol)
            elif model_choice == "River":
                model_river(symbol)
            elif model_choice == "Shock":
                model_shock(symbol)
            else:
                messagebox.showerror("Selection Error", "Please select a valid forecasting model.")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
    threading.Thread(target=run_model).start()

# Create the main GUI window.
root = tk.Tk()
root.title("Forecasting Models")

# Model selection frame
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Select Forecasting Model:").grid(row=0, column=0, sticky=tk.W)
model_var = tk.StringVar(value="GBM")
ttk.Radiobutton(frame, text="Underlying Price Forecast (Gas Diffusion/GBM)", variable=model_var, value="GBM").grid(row=1, column=0, sticky=tk.W)
ttk.Radiobutton(frame, text="Option Surface Forecast (Rainfall & River Flow)", variable=model_var, value="River").grid(row=2, column=0, sticky=tk.W)
ttk.Radiobutton(frame, text="Option Surface Forecast (Shock Events)", variable=model_var, value="Shock").grid(row=3, column=0, sticky=tk.W)

# Ticker symbol entry
ttk.Label(frame, text="Ticker Symbol:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
ticker_entry = ttk.Entry(frame, width=20)
ticker_entry.grid(row=5, column=0, sticky=(tk.W, tk.E))
ticker_entry.focus()

# Run button
run_button = ttk.Button(frame, text="Run Model", command=run_forecasting_model)
run_button.grid(row=6, column=0, pady=10)

# Start the GUI event loop.
root.mainloop()
