import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

minutes_per_hour = 60
minutes_per_day = 1440
heart_rate_data = np.linspace(0, 24, minutes_per_day)


def low_pass_filter(data, cutoff=0.01, order=3):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

smooth_noise = low_pass_filter(heart_rate_data)


def compute_hourly_averages(data, minutes_per_hour=60):
    hourly_averages = np.mean(data.reshape(-1, minutes_per_hour), axis=1)
    return hourly_averages


hourly_averages = compute_hourly_averages(smooth_noise)
plt.plot(time, heart_rate_data, label='Noisy Data', color='red', alpha=0.6)
plt.plot(time, smooth_noise, label='Smoothed Data', color='blue', linewidth=2)
plt.scatter(np.arange(24), hourly_averages, color='green', label='Hourly Averages', zorder=5)

def highlight_exceeding_intervals(data, threshold=100, consecutive_minutes=25):
    exceed_indices = np.where(data > threshold)[0]
    exceed_streaks = np.split(exceed_indices, np.where(np.diff(exceed_indices) > 1)[0] + 1)
    
    for streak in exceed_streaks:
        if len(streak) >= consecutive_minutes:
            plt.plot(time[streak], smoothed_traffic[streak], 'o','r', linestyle='none', label='Exceeding Traffic Threshold')

highlight_exceeding_intervals(smooth_noise)


plt.xlabel('Time (hours)')
plt.ylabel('Number of heart rate per Minute')
plt.show()
