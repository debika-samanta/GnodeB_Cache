import pandas as pd
from collections import deque
import random

# Load the request.csv data
request_df = pd.read_csv('requests.csv')

# Initialize the lookup table
lookup_df = pd.DataFrame(columns=['URL', 'Access_Frequency', 'Content_Size', 'Last_Access_Time',
                                  'Access_Frequency_Over_Time', 'Storage_Cost', 'Content_Type',
                                  'Data_Freshness', 'Cache_Hit_Miss'])

# Initialize the cache
cache = deque(maxlen=50)  # Cache with a maximum size of 50
cache_hits = 0
cache_misses = 0

# Define a small epsilon value to avoid division by zero
epsilon = 1e-10

# Function to determine the next time a URL will be accessed
def next_access_time(url, current_index, request_df):
    future_requests = request_df.iloc[current_index + 1:]
    if url in future_requests['URL'].values:
        return future_requests[future_requests['URL'] == url].index[0]
    else:
        return float('inf')  # If it will not be accessed again

# Process each request in order
for i, row in request_df.iterrows():
    url = row['URL']
    request_time = row['Request_Time']
    
    # Determine cache hit or miss
    if url in cache:
        cache_hits += 1
        cache_hit_miss = 'Hit'
    else:
        cache_misses += 1
        cache_hit_miss = 'Miss'
        
        # If cache is full, find the optimal item to evict
        if len(cache) >= cache.maxlen:
            future_accesses = {cached_url: next_access_time(cached_url, i, request_df) for cached_url in cache}
            url_to_evict = max(future_accesses, key=future_accesses.get)
            cache.remove(url_to_evict)
        
        # Add the new URL to the cache
        cache.append(url)
    
    # Calculate other fields
    access_frequency = request_df[request_df['URL'] == url].shape[0]
    content_size = random.randint(1000, 10000)  # Random content size in KB
    last_access_time = row['Request_Time']
    first_access_time = request_df[request_df['URL'] == url]['Request_Time'].min()
    
    # Avoid division by zero using epsilon
    time_diff_days = (pd.to_datetime(request_time) - pd.to_datetime(first_access_time)).days + epsilon
    access_frequency_over_time = access_frequency / time_diff_days
    
    storage_cost = content_size * 0.01  # Simple storage cost formula
    content_type = url.split('.')[-1]  # Extract content type from URL
    
    # Avoid division by zero for data freshness as well
    time_since_last_access = (pd.to_datetime(request_time) - pd.to_datetime(last_access_time)).days + epsilon
    data_freshness = 1 / time_since_last_access
    
    # Append the data to the lookup table
    lookup_df = lookup_df.append({
        'URL': url,
        'Access_Frequency': access_frequency,
        'Content_Size': content_size,
        'Last_Access_Time': last_access_time,
        'Access_Frequency_Over_Time': access_frequency_over_time,
        'Storage_Cost': storage_cost,
        'Content_Type': content_type,
        'Data_Freshness': data_freshness,
        'Cache_Hit_Miss': cache_hit_miss
    }, ignore_index=True)

# Save the lookup table to a CSV file
lookup_df.to_csv('lookupT.csv', index=False)

# Print cache hit and miss statistics
print(f"Cache Hits: {cache_hits}")
print(f"Cache Misses: {cache_misses}")
