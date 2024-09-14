import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters for data generation
num_entries = 500  # Number of requests
num_files = 100  # Number of unique video files

# Generate synthetic data
data = {
    'URL': [f'http://localhost:5000/video/item_{random.randint(0, num_files - 1)}.mp4' for _ in range(num_entries)],  # Randomized URLs
    'Request_Time': [(datetime.now() - timedelta(seconds=np.random.randint(0, 3600))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(num_entries)]  # Random request times within the past hour
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
file_path = 'random_client_requests.csv'
df.to_csv(file_path, index=False)

print(f"CSV file saved at {file_path}")
