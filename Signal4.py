from scipy import signal as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# Load data
data_1, data_2 = [np.load(path) for path in data_paths]
aux_1, aux_2 = [np.load(path) for path in aux_paths]

# Convert EEG data to DataFrame
df_1 = pd.DataFrame(data_1).transpose().iloc[1:]
df_2 = pd.DataFrame(data_2).transpose().iloc[1:]


# Bandpass filter function
def apply_band_filter(signal, lowcut, highcut, fs, order=4):
    b, a = sp.butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return sp.filtfilt(b, a, signal)


# Compute PSD
def compute_psd(signal, fs):
    nperseg = min(len(signal), fs * 2)  # Ensure valid segment length
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    return freqs, psd


# EEG Filtering parameters
fs = 250
order = 4
theta_band = (4, 7)
alpha_band = (8, 12)
beta_band = (13, 30)


# Process EEG signals while keeping order
def process_eeg(df):
    filtered_df = pd.DataFrame()
    eeg_channels = [col for col in df.columns if
                    col not in ['Unnamed: 0', 'key_0', 'Light_sensor', 'rel_time', 'Change', 'truth_value']]

    for col in eeg_channels:
        filtered_df[f'{col}_Theta'] = apply_band_filter(df[col], *theta_band, fs, order)
        filtered_df[f'{col}_Alpha'] = apply_band_filter(df[col], *alpha_band, fs, order)
        filtered_df[f'{col}_Beta'] = apply_band_filter(df[col], *beta_band, fs, order)

    # Preserve time and truth_value **without splitting**
    filtered_df['rel_time'] = df['rel_time']
    filtered_df['truth_value'] = df['truth_value']

    return filtered_df


# Compute band power with a **sliding window**
def compute_band_power_sliding(filtered_df, window_size=500, step_size=250):
    results = []
    eeg_channels = [col.replace('_Theta', '') for col in filtered_df.columns if '_Theta' in col]

    for start in range(0, len(filtered_df) - window_size, step_size):
        window = filtered_df.iloc[start:start + window_size]  # Extract window
        truth_value = window["truth_value"].iloc[0]  # Assign truth label

        for col in eeg_channels:
            freqs, psd_theta = compute_psd(window[f'{col}_Theta'].values, fs)
            freqs, psd_alpha = compute_psd(window[f'{col}_Alpha'].values, fs)
            freqs, psd_beta = compute_psd(window[f'{col}_Beta'].values, fs)

            theta_power = np.sum(psd_theta[(freqs >= 4) & (freqs <= 7)])
            alpha_power = np.sum(psd_alpha[(freqs >= 8) & (freqs <= 12)])
            beta_power = np.sum(psd_beta[(freqs >= 13) & (freqs <= 30)])

            row = {
                "Channel": col,
                "Theta_Power": theta_power,
                "Alpha_Power": alpha_power,
                "Beta_Power": beta_power,
                "Theta/Alpha": theta_power / (alpha_power + 1e-6),
                "Beta/Alpha": beta_power / (alpha_power + 1e-6),
                "Truth_Value": truth_value,
                "Start_Time": filtered_df["rel_time"].iloc[start],  # Keep track of window start time
            }
            results.append(row)

    return pd.DataFrame(results)


# Normalize light sensor
def normalize_light_sensor(aux_data):
    return (aux_data[1][1:] > np.median(aux_data[1][1:])).astype(int)


# Load labels and extract truth_value
labels = pd.read_csv(r"C:\Users\ALEX\PythonProject3\COGS189project\Cogs_189_Project\trial_2025-03-03.csv")
if 'truth_value' not in labels.columns:
    raise KeyError("Expected 'truth_value' column in labels file")

# Assign truth values to EEG data
filtered_df_3 = pd.read_csv('no_talk_labelled.csv')
print(filtered_df_3.columns)

# Process EEG while keeping order
filtered_df = process_eeg(filtered_df_3)

# Compute features for full dataset **before filtering**
features = compute_band_power_sliding(filtered_df, window_size=500, step_size=250)

# Separate truthful and deceptive **after computing features**
truthful = features[features["Truth_Value"] == 1]
deceptive = features[features["Truth_Value"] == 0]

# Check if data ordering is preserved
print(features.head())
print(truthful.shape, deceptive.shape)

# Plot results
plt.figure(figsize=(10, 5))
sns.barplot(x="Channel", y="Alpha_Power", data=deceptive, color="red", label="Deceptive", alpha=0.5)
sns.barplot(x="Channel", y="Alpha_Power", data=truthful, color="blue", label="Truthful", alpha=0.5)
plt.xlabel("Channels")
plt.ylabel("Alpha Power")
plt.title("Alpha Power Distribution by Truth Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.kdeplot(deceptive["Beta_Power"], label="Deceptive", fill=True)
sns.kdeplot(truthful["Beta_Power"], label="Truthful", fill=True)

plt.xlabel("Beta Power")
plt.ylabel("Density")
plt.title("Beta Power Distribution by Truth Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.kdeplot(deceptive["Theta_Power"], label="Deceptive", fill=True)
sns.kdeplot(truthful["Theta_Power"], label="Truthful", fill=True)

plt.xlabel("Theta Power")
plt.ylabel("Density")
plt.title("Theta Power Distribution by Truth Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="Channel", y="Beta/Alpha", data=deceptive, color="red", label="Deceptive", alpha=0.25)
sns.barplot(x="Channel", y="Beta/Alpha", data=truthful, color="blue", label="Truthful", alpha=0.25)
plt.xlabel("Channels")
plt.ylabel("Beta/Alpha Ratio")
plt.title("Beta/Alpha Power Distribution by Truth Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="Channel", y="Theta/Alpha", data=deceptive, color="red", label="Deceptive", alpha=0.25)
sns.barplot(x="Channel", y="Theta/Alpha", data=truthful, color="blue", label="Truthful", alpha=0.25)

plt.xlabel("Channels")
plt.ylabel("Theta/Alpha Ratio")
plt.title("Theta/Alpha Power Distribution by Truth Value")
plt.legend()
plt.show()
