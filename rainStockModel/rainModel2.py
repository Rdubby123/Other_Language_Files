import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from astropy.timeseries import LombScargle

def fetch_option_chain(symbol):
    ticker = yf.Ticker(symbol)

    # Check if the ticker has any options available
    if not ticker.options:
        raise ValueError(f"No option chain found for symbol '{symbol}'. Please check if the symbol is correct or try a different one.")

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

    # Final check to ensure we have at least some valid options data
    if not options_data:
        raise ValueError(f"Failed to retrieve any valid option data for symbol '{symbol}'.")

    full_calls_df = pd.concat(options_data, ignore_index=True)
    return full_calls_df

def process_option_data(df, max_days_to_expire=400):
    today = pd.Timestamp.now().normalize()
    df['time_to_expire'] = (df['expiration'] - today).dt.days
    df = df.dropna(subset=['strike', 'lastPrice', 'time_to_expire'])
    df = df[df['time_to_expire'] <= max_days_to_expire]
    return df

def create_surface_grid(df, grid_res=100):
    strike_range = (df['strike'].min(), df['strike'].max())
    expire_range = (df['time_to_expire'].min(), df['time_to_expire'].max())
    grid_x, grid_y = np.meshgrid(
        np.linspace(*strike_range, grid_res),
        np.linspace(*expire_range, grid_res)
    )
    points = df[['strike', 'time_to_expire']].values
    values = df['lastPrice'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    nan_mask = np.isnan(grid_z)
    grid_z[nan_mask] = np.nanmean(grid_z)
    return grid_x, grid_y, grid_z

def simulate_rain_and_flow(grid_x, grid_y, grid_z, smoothing=2.0, scale_factor=5.0, rain_fraction=0.1):
    heightmap = gaussian_filter(grid_z, sigma=smoothing)
    grad_y, grad_x = np.gradient(-heightmap)
    water_flow = np.zeros_like(heightmap)
    rainfall = np.zeros_like(heightmap)
    num_rain_points = int(np.prod(rainfall.shape) * rain_fraction)
    rain_points = np.random.choice(np.arange(heightmap.size), num_rain_points, replace=False)
    np.put(rainfall, rain_points, 1.0)
    iterations = 500  
    for _ in range(iterations):
        flow_x = grad_x * rainfall
        flow_y = grad_y * rainfall
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

def stochastic_gradient_descent(grid_x, grid_y, grid_z, flow_accumulation, start_point,
                                  learning_rate=0.1, noise_scale=0.02, iterations=100, iteration_time_unit=1):
    dz_dy, dz_dx = np.gradient(grid_z)
    x_vals = grid_x[0, :]
    y_vals = grid_y[:, 0]
    grad_x_interp = RectBivariateSpline(y_vals, x_vals, dz_dx)
    grad_y_interp = RectBivariateSpline(y_vals, x_vals, dz_dy)
    flow_interp = RectBivariateSpline(y_vals, x_vals, flow_accumulation)
    
    path = [start_point]
    current_point = np.array(start_point, dtype=float)
    time_record = [0]
    
    for i in range(iterations):
        grad_x_val = grad_x_interp(current_point[1], current_point[0])[0][0]
        grad_y_val = grad_y_interp(current_point[1], current_point[0])[0][0]
        local_flow = flow_interp(current_point[1], current_point[0])[0][0]
        effective_lr = learning_rate * (1 + local_flow)
        noise = np.random.normal(0, noise_scale, size=2)
        update = effective_lr * np.array([grad_x_val, grad_y_val]) + noise
        current_point = current_point - update
        path.append(current_point.copy())
        time_record.append(time_record[-1] + iteration_time_unit)
    
    return np.array(path), np.array(time_record)

def plot_descent_path(grid_x, grid_y, grid_z, descent_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    cp = ax.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=50)
    plt.colorbar(cp, ax=ax, label='Call Option Price (USD)')
    ax.plot(descent_path[:, 0], descent_path[:, 1], marker='o', color='red', linewidth=2, markersize=4)
    ax.set_xlabel('Underlying Price (USD)')
    ax.set_ylabel('Time to Expire (days)')
    ax.set_title('Stochastic Gradient Descent Path on the Option Price Manifold\n(Price in USD; Time to Expire in days)')
    plt.show()

def plot_lomb_scargle_of_path(descent_path, time_record, sigma=2):
    x_path = descent_path[:, 0]
    t = time_record
    new_t = np.linspace(t[0], t[-1], (len(x_path)-1)*24+1)
    x_fine = np.interp(new_t, t, x_path)
    x_smooth = gaussian_filter1d(x_fine, sigma=sigma)
    ls = LombScargle(new_t, x_smooth)
    frequency, power = ls.autopower()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequency, power, 'b-')
    ax.set_xlabel('Frequency (cycles per day)')
    ax.set_ylabel('Power (Arbitrary Units)')
    ax.set_title('Lomb-Scargle Periodogram of the Smoothed Descent Path\n(Underlying Price in USD vs. Time in days)')
    plt.show()

def forecast_option_price(grid_x, grid_y, grid_z, flow_accumulation, start_point,
                            forecast_horizon=30, n_sim=100, learning_rate=0.1, noise_scale=0.02):
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

def plot_forecast_paths(forecast_paths, forecast_time):
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
    plt.title('Forecast of Future Asset Price Dynamics Based on Option Price Surface\n')
    plt.legend()
    plt.show()

def main():
    symbol = 'AMD'  # You can change this to test others
    try:
        print(f"Fetching option data for {symbol}...")
        call_options_df = fetch_option_chain(symbol)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("Processing option data...")
    processed_df = process_option_data(call_options_df, max_days_to_expire=200)
    print("Creating option price manifold...")
    grid_x, grid_y, grid_z = create_surface_grid(processed_df, grid_res=100)
    print("Simulating rainfall and river flow (to highlight channels)...")
    flow_accumulation = simulate_rain_and_flow(grid_x, grid_y, grid_z, rain_fraction=0.2)
    print("Plotting option price surface with river channels...")
    plot_surface_with_rivers(grid_x, grid_y, grid_z, flow_accumulation, symbol)
    
    start_strike = grid_x[0, grid_x.shape[1] // 2]
    start_expire = grid_y[grid_y.shape[0] // 2, 0]
    start_point = [start_strike, start_expire]
    
    print("Performing stochastic gradient descent on the manifold...")
    descent_path, time_record = stochastic_gradient_descent(
        grid_x, grid_y, grid_z, flow_accumulation, start_point,
        learning_rate=0.1, noise_scale=0.02, iterations=100, iteration_time_unit=1)
    plot_descent_path(grid_x, grid_y, grid_z, descent_path)
    
    print("Performing Lomb-Scargle analysis on the descent path...")
    plot_lomb_scargle_of_path(descent_path, time_record)
    
    print("Forecasting future underlying price dynamics...")
    forecast_horizon = 30
    n_sim = 100
    forecast_paths, forecast_time = forecast_option_price(
        grid_x, grid_y, grid_z, flow_accumulation, start_point,
        forecast_horizon=forecast_horizon, n_sim=n_sim, learning_rate=0.1, noise_scale=0.02)
    plot_forecast_paths(forecast_paths, forecast_time)

main()
