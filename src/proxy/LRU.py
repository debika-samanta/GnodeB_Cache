import requests
import pandas as pd
import os
import shutil
from flask import Flask, request, jsonify

app = Flask(__name__)

# File paths
lookup_csv_file = 'lookup_table.csv'
cache_directory = 'cache'  # Directory to store cached files
max_cache_size = 50 * 1024 * 1024  # Maximum cache size in bytes (e.g., 50MB)

# Create cache directory if it doesn't exist
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)

# Load lookup table if it exists, otherwise create a new DataFrame
lookup_df = pd.read_csv(lookup_csv_file) if os.path.exists(lookup_csv_file) else pd.DataFrame(columns=[
    'URL', 'Access_Frequency', 'Content_Size', 'Last_Access_Time',
    'Access_Frequency_Over_Time', 'Storage_Cost', 'Content_Type', 'Data_Freshness'
])

def fetch_from_server(server_url, url):
    """
    Fetch the content from the server and return it.
    """
    try:
        response = requests.get(f'{server_url}/video/{url}')
        if response.status_code == 200:
            # Save the file to the cache
            cache_path = os.path.join(cache_directory, url)
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            print(f"File '{url}' fetched from server and cached.")
            manage_cache_eviction()
            update_lookup_table(url)
            return response.content
        else:
            print(f"Failed to fetch '{url}' from server. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching '{url}' from the server: {e}")
        return None

@app.route('/cache/<path:url>', methods=['GET'])
def handle_request(url):
    # Check if the file is in the cache
    cache_path = os.path.join(cache_directory, url)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            content = f.read()
        # Update the last access time in the lookup table
        update_lookup_table(url, accessed=True)
        print(f"Serving '{url}' from cache.")
        return content
    else:
        # If not in cache, fetch from the server
        server_url = request.args.get('server_url')  # Expecting client to provide the server URL
        if server_url:
            content = fetch_from_server(server_url, url)
            if content:
                return content
            else:
                return "Error fetching content", 500
        else:
            return "Server URL not provided", 400

def update_lookup_table(url, accessed=False):
    """
    Update the lookup table with metadata for the given URL.
    If 'accessed' is True, update the last access time and access frequency.
    """
    global lookup_df
    cache_path = os.path.join(cache_directory, url)
    if not os.path.exists(cache_path):
        return

    content_size = os.path.getsize(cache_path)
    last_access_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    if accessed:
        lookup_df.loc[lookup_df['URL'] == url, 'Last_Access_Time'] = last_access_time
        lookup_df.loc[lookup_df['URL'] == url, 'Access_Frequency'] += 1
        lookup_df.loc[lookup_df['URL'] == url, 'Access_Frequency_Over_Time'] += 1
    else:
        # Example metadata, replace this with actual metadata if available
        metadata = {
            'URL': url,
            'Access_Frequency': 1,
            'Content_Size': content_size,
            'Last_Access_Time': last_access_time,
            'Access_Frequency_Over_Time': 1,
            'Storage_Cost': 10,  # Example storage cost
            'Content_Type': 'Video',  # Example content type
            'Data_Freshness': 'Fresh'  # Example freshness
        }
        lookup_df = lookup_df.append(metadata, ignore_index=True)

    # Save the updated lookup table
    lookup_df.to_csv(lookup_csv_file, index=False)
    print(f"Lookup table updated for URL '{url}'.")

def get_total_cache_size():
    """
    Calculate the total size of the files in the cache.
    """
    total_size = 0
    for filename in os.listdir(cache_directory):
        file_path = os.path.join(cache_directory, filename)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    return total_size

def manage_cache_eviction():
    """
    Evict least recently used items from the cache if it exceeds the maximum size.
    """
    global lookup_df

    total_cache_size = get_total_cache_size()
    if total_cache_size > max_cache_size:
        # Sort the lookup table by Last_Access_Time to find the least recently used items
        lookup_df.sort_values(by='Last_Access_Time', inplace=True)

        while total_cache_size > max_cache_size:
            # Evict the least recently used item
            url_to_evict = lookup_df.iloc[0]['URL']
            cache_path = os.path.join(cache_directory, url_to_evict)

            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Evicted '{url_to_evict}' from cache.")
                total_cache_size = get_total_cache_size()
                # Remove the entry from the lookup table
                lookup_df = lookup_df[lookup_df['URL'] != url_to_evict]
                lookup_df.to_csv(lookup_csv_file, index=False)

if __name__ == '__main__':
    app.run(host='localhost', port=5004)
