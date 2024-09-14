import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Number of cache items
num_items = 500

# Random seed for reproducibility
np.random.seed(42)

# Generating synthetic data with corrected logic
access_frequency = np.random.randint(10, 100, num_items)  # How often the item is accessed
access_frequency_over_time = [np.random.randint(1, freq) for freq in access_frequency]  # Access frequency over a specific period

# Generate random URLs for the items
def generate_url(index):
    return f"http://localhost:5000/video/item_{index}.mp4"

data_corrected = {
    'URL': [generate_url(i) for i in range(num_items)],  # URL of the cached item
    'Access_Frequency': access_frequency,  # Total access frequency
    'Content_Size': np.random.randint(1, 500, num_items),  # Size of the content (in KB)
    'Last_Access_Time': [(datetime.now() - timedelta(seconds=np.random.randint(1, 1000000))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(num_items)],  # Timestamp of last access
    'Access_Frequency_Over_Time': access_frequency_over_time,  # Access frequency over a specific period
    'Storage_Cost': np.random.randint(1, 1000, num_items),  # Cost to store the item (financial measurement)
    'Content_Type': [random.choice(['Social Media', 'Live Video', 'Blogs', 'Ebook', 'Infographic', 'News', 'Advertisement', 'Article', 'Emails', 'Videos', 'Authority Content']) for _ in range(num_items)],  # Type of content
    'Data_Freshness': [random.choice(['Fresh', 'Stale']) for _ in range(num_items)]  # Data freshness status
}

# Creating a DataFrame with corrected data
df_corrected = pd.DataFrame(data_corrected)

# Saving the corrected data to a CSV file
file_path_corrected = 'cache_items_data_with_urls.csv'
df_corrected.to_csv(file_path_corrected, index=False)

print(f"CSV file saved at {file_path_corrected}")
