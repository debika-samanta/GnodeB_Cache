import requests
import pandas as pd
import os
import shutil
from flask import Flask, request, jsonify

app = Flask(__name__)

# File paths
lookup_csv_file = 'lookup_table.csv'
cache_directory = 'cache'  # Directory to store cached files

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
        print(f"Serving '{url}' from cache.")
        return content
    else:
        # If not in cache, fetch from the server
        server_url = request.args.get('server_url')  # Expecting client to provide the server URL
        if server_url:
            content = fetch_from_server(server_url, url)
            if content:
                update_lookup_table(url)  # Update lookup table after caching
                return content
            else:
                return "Error fetching content", 500
        else:
            return "Server URL not provided", 400

def update_lookup_table(url):
    """
    Update the lookup table with metadata for the given URL.
    """
    # Check if URL is already in the lookup table
    if lookup_df[lookup_df['URL'] == url].empty:
        # Example metadata, replace this with actual metadata if available
        metadata = {
            'URL': url,
            'Access_Frequency': 1,  # Example frequency
            'Content_Size': os.path.getsize(os.path.join(cache_directory, url)),
            'Last_Access_Time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Access_Frequency_Over_Time': 1,  # Example frequency over time
            'Storage_Cost': 10,  # Example storage cost
            'Content_Type': 'Video',  # Example content type
            'Data_Freshness': 'Fresh'  # Example freshness
        }
        lookup_df.loc[len(lookup_df)] = metadata
        # Save the updated lookup table
        lookup_df.to_csv(lookup_csv_file, index=False)
        print(f"Lookup table updated for URL '{url}'.")
    else:
        print(f"URL '{url}' already exists in the lookup table.")

if __name__ == '__main__':
    app.run(host='localhost', port=5004)
