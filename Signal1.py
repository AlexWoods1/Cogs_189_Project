from scipy import signal as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from scipy.signal import welch

# Load EEG and auxiliary data

data_paths = [
    r"C:\Users\ALEX\PythonProject3\COGS189project\Cogs_189_Project\data\misc\eeg_run-1.npy",
    r"C:\Users\ALEX\PythonProject3\COGS189project\Cogs_189_Project\data\misc\eeg_run-2.npy"
]
aux_paths = [
    r"C:\Users\ALEX\PythonProject3\COGS189project\Cogs_189_Project\data\misc\aux_run-1.npy",
    r"C:\Users\ALEX\PythonProject3\COGS189project\Cogs_189_Project\data\misc\aux_run-2.npy"
]

# Loading data
data_1, data_2 = [np.load(path) for path in data_paths]
aux_1, aux_2 = [np.load(path) for path in aux_paths]

# Convert EEG data to DataFrame
df = pd.DataFrame(data_1).transpose().iloc[1:]
df_2 = pd.DataFrame(data_2).transpose().iloc[1:]

# Bandpass filter function
def apply_band_filter(signal, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter."""
    b, a = sp.butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return sp.filtfilt(b, a, signal)

# Compute Power Spectral Density (PSD)
def compute_psd(signal, fs):
    """Compute power spectral density using Welch’s method."""
    freqs, psd = welch(signal, fs, nperseg=fs*2)  # 2-second window
    return freqs, psd

# EEG Filtering parameters
fs = 250  # Sampling frequency
order = 4  # Filter order
theta_band = (4, 7)
alpha_band = (8, 12)
beta_band = (13, 30)

# Process EEG signals for both datasets
filtered_df_1 = pd.DataFrame()
filtered_df_2 = pd.DataFrame()

for df_source, filtered_df in zip([df, df_2], [filtered_df_1, filtered_df_2]):
    for col in range(8):  # Assuming 8 EEG channels
        filtered_df[f'Ch{col}_Theta'] = apply_band_filter(df_source[col], *theta_band, fs, order)
        filtered_df[f'Ch{col}_Alpha'] = apply_band_filter(df_source[col], *alpha_band, fs, order)
        filtered_df[f'Ch{col}_Beta'] = apply_band_filter(df_source[col], *beta_band, fs, order)

# Add light sensor data and normalize
def normalize_light_sensor(aux_data):
    """Binarizes the light sensor data based on its median."""
    return (aux_data[1][1:] > np.median(aux_data[1][1:])).astype(int)

filtered_df_1["Light_sensor"] = normalize_light_sensor(aux_1)
filtered_df_2["Light_sensor"] = normalize_light_sensor(aux_2)

# Compute power features split by light sensor state
def compute_band_power(filtered_df):
    features_light_1 = []
    features_light_0 = []

    for col in range(8):  # Process each EEG channel
        light_1_data = filtered_df[filtered_df["Light_sensor"] == 1]
        light_0_data = filtered_df[filtered_df["Light_sensor"] == 0]

        for data, features in [(light_1_data, features_light_1), (light_0_data, features_light_0)]:
            if len(data) > 0:
                freqs, psd_theta = compute_psd(data[f'Ch{col}_Theta'], fs)
                freqs, psd_alpha = compute_psd(data[f'Ch{col}_Alpha'], fs)
                freqs, psd_beta = compute_psd(data[f'Ch{col}_Beta'], fs)

                # Extract power in frequency bands
                theta_power = np.sum(psd_theta[(freqs >= 4) & (freqs <= 7)])
                alpha_power = np.sum(psd_alpha[(freqs >= 8) & (freqs <= 12)])
                beta_power = np.sum(psd_beta[(freqs >= 13) & (freqs <= 30)])

                # Compute mental load indicators
                theta_alpha_ratio = theta_power / (alpha_power + 1e-6)  # Avoid division by zero
                beta_alpha_ratio = beta_power / (alpha_power + 1e-6)

                features.append({
                    "Channel": col,
                    "Theta_Power": theta_power,
                    "Alpha_Power": alpha_power,
                    "Beta_Power": beta_power,
                    "Theta/Alpha": theta_alpha_ratio,
                    "Beta/Alpha": beta_alpha_ratio,
                    "Light_State": 1 if features == features_light_1 else 0
                })

    return pd.DataFrame(features_light_1), pd.DataFrame(features_light_0)

# Compute features for both trials
features_light_1_df_1, features_light_0_df_1 = compute_band_power(filtered_df_1)
features_light_1_df_2, features_light_0_df_2 = compute_band_power(filtered_df_2)

# Load labels and convert time
labels = pd.read_csv(r"C:\Users\ALEX\PythonProject3\COGS189project\Cogs_189_Project\trial_2025-03-03.csv")
labels['time'] = labels['time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f'))

# Plot EEG Alpha band under different light conditions
plt.figure(figsize=(10, 5))
sns.kdeplot(features_light_1_df_1["Alpha_Power"], label="Thinking", shade=True)
sns.kdeplot(features_light_0_df_1["Alpha_Power"], label="Speaking", shade=True)
plt.xlabel("Alpha Power")
plt.ylabel("Density")
plt.title("Alpha Power Distribution")
plt.legend()
plt.show()
plt.savefig("Alpha Power Distribution.png")

plt.figure(figsize=(10, 5))
sns.kdeplot(features_light_1_df_1["Beta_Power"], label="Thinking", shade=True)
sns.kdeplot(features_light_0_df_1["Beta_Power"], label="Speaking", shade=True)
plt.xlabel("Beta Power")
plt.ylabel("Density")
plt.title("Beta Power Distribution")
plt.legend()
plt.show()
plt.savefig("Beta Power Distribution.png")

plt.figure(figsize=(10, 5))
sns.kdeplot(features_light_1_df_1["Theta_Power"], label="Thinking", shade=True)
sns.kdeplot(features_light_0_df_1["Theta_Power"], label="Speaking", shade=True)
plt.xlabel("Theta Power")
plt.ylabel("Density")
plt.title("Theta Power Distribution")
plt.legend()
plt.show()
plt.savefig('Theta Power Distribution.png')
# Save results for statistical analysis
features_light_1_df_1.to_csv("mental_load_light_on_trial_1.csv", index=False)
features_light_0_df_1.to_csv("mental_load_light_off_trial_1.csv", index=False)
features_light_1_df_2.to_csv("mental_load_light_on_trial_2.csv", index=False)
features_light_0_df_2.to_csv("mental_load_light_off_trial_2.csv", index=False)
