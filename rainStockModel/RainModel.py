import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, binary_closing

def fetch_option_chain(symbol):
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    print(f"Available expiration dates for {symbol}: {expirations}")

    options_data = []
    for expiry in expirations:
        opt_chain = ticker.option_chain(expiry)
        calls = opt_chain.calls
        calls['expiration'] = pd.to_datetime(expiry)
        options_data.append(calls)

    full_calls_df = pd.concat(options_data, ignore_index=True)
    return full_calls_df

def process_option_data(df, max_days_to_expiry=400):
    today = pd.Timestamp.now().normalize()
    df['time_to_expire'] = (df['expiration'] - today).dt.days
    df = df.dropna(subset=['strike', 'lastPrice', 'time_to_expire'])
    
    # Filter out options that have more than 400 days to expiry
    df = df[df['time_to_expire'] <= max_days_to_expiry]
    
    return df

def create_surface_grid(df, grid_res=100):
    strike_range = (df['strike'].min(), df['strike'].max())
    expiry_range = (df['time_to_expire'].min(), df['time_to_expire'].max())

    grid_x, grid_y = np.meshgrid(
        np.linspace(*strike_range, grid_res),
        np.linspace(*expiry_range, grid_res)
    )

    points = df[['strike', 'time_to_expire']].values
    values = df['lastPrice'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    nan_mask = np.isnan(grid_z)
    grid_z[nan_mask] = np.nanmean(grid_z)

    return grid_x, grid_y, grid_z

def simulate_rain_and_flow(grid_x, grid_y, grid_z, smoothing=2.0, scale_factor=5.0, rain_fraction=0.1):
    # Apply a low-pass filter (Gaussian smoothing) to the surface to reduce high-frequency fluctuations
    heightmap = gaussian_filter(grid_z, sigma=smoothing)

    # Compute gradients (slopes)
    grad_y, grad_x = np.gradient(-heightmap)

    # Initialize water accumulation map
    water_flow = np.zeros_like(heightmap)

    # Simulate more rainfall: randomly rain on a higher fraction of points
    rainfall = np.zeros_like(heightmap)
    num_rain_points = int(np.prod(rainfall.shape) * rain_fraction)  # Number of points to rain on
    rain_points = np.random.choice(np.arange(heightmap.size), num_rain_points, replace=False)
    np.put(rainfall, rain_points, 1.0)  # Set rain at selected points

    # Simulate flow: basic flow routing
    iterations = 500  # Number of iterations for the flow to reach equilibrium
    for _ in range(iterations):
        # Apply flow to the rain
        flow_x = grad_x * rainfall
        flow_y = grad_y * rainfall

        water_flow += rainfall  # Accumulate water flow

        # Shift water downhill based on slope
        rainfall_shifted = np.roll(rainfall, shift=1, axis=0) * (grad_y > 0)
        rainfall_shifted += np.roll(rainfall, shift=-1, axis=0) * (grad_y < 0)
        rainfall_shifted += np.roll(rainfall, shift=1, axis=1) * (grad_x > 0)
        rainfall_shifted += np.roll(rainfall, shift=-1, axis=1) * (grad_x < 0)

        # Normalize to prevent infinite accumulation
        rainfall = rainfall_shifted / 4.0

    # Scale the water flow accumulation to highlight the channels
    flow_normalized = (water_flow - np.min(water_flow)) / (np.max(water_flow) - np.min(water_flow))
    flow_scaled = flow_normalized * scale_factor

    return flow_scaled

def plot_surface_with_rivers(grid_x, grid_y, grid_z, flow_accumulation, symbol):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color the surface by the height (option price)
    surface_colors = plt.cm.viridis((grid_z - np.min(grid_z)) / (np.max(grid_z) - np.min(grid_z)))

    # Map flow accumulation to colors (blueish for rivers)
    flow_colors = plt.cm.Blues(flow_accumulation)

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, facecolors=surface_colors, rstride=1, cstride=1, antialiased=True)

    # Plot the water flow (rivers)
    ax.plot_surface(grid_x, grid_y, grid_z, facecolors=flow_colors, rstride=1, cstride=1, alpha=0.5, antialiased=False)

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expire (days)')
    ax.set_zlabel('Call Option Price')
    ax.set_title(f'Rainfall and River Flow on Option Price Surface for {symbol}')

    plt.show()

def plot_option_price_vs_time(grid_x, grid_y, flow_accumulation):
    # We will plot the option price vs time with weighting based on the water flow (river channels)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sum the flow in each column to create a total weight for each strike price/time to expiry combination
    strike_weights = np.sum(flow_accumulation, axis=0)  # Sum by columns (time to expiration)
    price_weights = np.sum(flow_accumulation, axis=1)  # Sum by rows (strike prices)

    # Create weighted option price vs time to expiration graph
    ax.plot(grid_y[:, 0], strike_weights, label="Weighted Price for Strike Prices")
    ax.set_xlabel('Time to Expire (days)')
    ax.set_ylabel('Weighted Option Price')
    ax.set_title('Weighted Option Price vs Time to Expiry with River Flow')

    ax.legend()
    plt.show()

def plot_river_flow_2d(grid_x, grid_y, flow_accumulation):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the river flow intensity as a heatmap
    im = ax.imshow(flow_accumulation, cmap="Blues", origin="lower",
                   extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                   aspect='auto')

    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Flow Intensity (Water Accumulation)")

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expiry (days)')
    ax.set_title('Simulated River Flow Over Option Price Surface')

    plt.show()

def main():
    symbol = 'SPY'  # Change as needed

    print(f"Fetching option data for {symbol}...")
    call_options_df = fetch_option_chain(symbol)

    print("Processing option data...")
    processed_df = process_option_data(call_options_df, max_days_to_expiry=300)  # Filtering to 400 days

    print("Creating surface grid...")
    grid_x, grid_y, grid_z = create_surface_grid(processed_df, grid_res=100)

    print("Simulating rainfall and river flow...")
    flow_accumulation = simulate_rain_and_flow(grid_x, grid_y, grid_z, rain_fraction=0.2)  # Increased rain fraction

    print("Plotting surface with rivers...")
    plot_surface_with_rivers(grid_x, grid_y, grid_z, flow_accumulation, symbol)

    print("Plotting option price vs. time to expire with river weighting...")
    plot_option_price_vs_time(grid_x, grid_y, flow_accumulation)

    print("Plotting all river flows in 2D...")
    plot_river_flow_2d(grid_x, grid_y, flow_accumulation)


main()