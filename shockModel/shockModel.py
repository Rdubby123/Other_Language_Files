import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter

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
    """
    Simulate shock events that diffuse across the price surface.
    Instead of iterative diffusion, apply a single Gaussian filter with an effective sigma.
    """
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
    # Normalize color maps for the surface and shock field
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

def stochastic_gradient_descent(grid_x, grid_y, grid_z, shock_field, start_point,
                                  learning_rate=0.1, noise_scale=0.02, iterations=100, iteration_time_unit=1):
    """
    Perform stochastic gradient descent where the local shock modulates the learning rate.
    """
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
    plt.show()

def forecast_option_price(grid_x, grid_y, grid_z, shock_field, start_point,
                          forecast_horizon=30, n_sim=100, learning_rate=0.1, noise_scale=0.02):
    """
    Forecast future asset price dynamics by simulating multiple paths.
    The shock field influences the update steps.
    """
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

def main():
    symbol = input("Ticker Symbol? ")
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
    
    print("Performing stochastic gradient descent on the manifold using shock modulation...")
    descent_path, _ = stochastic_gradient_descent(
        grid_x, grid_y, grid_z, shock_field, start_point,
        learning_rate=0.01, noise_scale=0.02, iterations=1000, iteration_time_unit=1)
    plot_descent_path(grid_x, grid_y, grid_z, descent_path, symbol)
    
    print(f"Forecasting future underlying price dynamics for {symbol}...")
    forecast_horizon = 30
    n_sim = 100
    forecast_paths, forecast_time = forecast_option_price(
        grid_x, grid_y, grid_z, shock_field, start_point,
        forecast_horizon=forecast_horizon, n_sim=n_sim, learning_rate=0.01, noise_scale=0.02)
    plot_forecast_paths(forecast_paths, forecast_time, symbol)

try:
    main()
except:
    main()
