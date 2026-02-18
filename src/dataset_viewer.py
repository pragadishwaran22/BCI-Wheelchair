import mne

# Path to your dataset file (change this to your actual path)
file_path = "data/B0101T.gdf"

# Load the dataset
raw = mne.io.read_raw_gdf(file_path, preload=True)

# Print info about the dataset
print(raw.info)

# Plot the signals (first few channels)
raw.plot(n_channels=10, duration=50)
raw.plot(n_channels=6, duration=100)

