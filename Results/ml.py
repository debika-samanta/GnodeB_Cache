import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# Cache Simulation
class CacheSimulator:
    def __init__(self, capacity, data_file=None):
        self.capacity = capacity
        self.cache = deque(maxlen=capacity)
        self.cache_set = set()
        self.df, self.label_encoders = self.load_and_preprocess_data(data_file) if data_file else (None, {})
        self.cache_hits = 0  # Initialize cache hits counter
        
    def load_and_preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        df['Last_Access_Time'] = pd.to_datetime(df['Last_Access_Time'])
        df['Last_Access_Time'] = (df['Last_Access_Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df['Last_Access_Time'] = df['Last_Access_Time'] / 10**9
        return df, {}

    def access(self, url):
        if url in self.cache_set:
            self.cache.remove(url)
            self.cache.append(url)
            self.cache_hits += 1  # Increment cache hits if the URL is already in the cache
        else:
            if len(self.cache) >= self.capacity:
                self.evict()
            self.cache.append(url)
            self.cache_set.add(url)
            
    def evict(self):
        evict_url = self.cache.popleft()
        self.cache_set.remove(evict_url)
    
    def simulate(self, request_sequence):
        cache_hits_per_request = []
        for url in request_sequence:
            self.access(url)
            cache_hits_per_request.append(self.cache_hits)
        return cache_hits_per_request

# Modified function to plot a line graph
def plot_cache_hits(cache_hits_per_request):
    plt.figure(figsize=(12, 8))
    plt.title('Cache Hits Over Time')
    
    # Plotting the line graph
    plt.plot(range(1, len(cache_hits_per_request) + 1), cache_hits_per_request, marker='', linestyle='-', color='b', linewidth=2)
    
    plt.xlabel('Number of Requests')
    plt.ylabel('Number of Cache Hits')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to save the x, y values to a CSV file
def save_results_to_csv(x_values, y_values, filename='cache_hits_results.csv'):
    df = pd.DataFrame({'Number_of_Requests': x_values, 'Cache_Hits': y_values})
    df.to_csv(filename, index=False)
    print(f'Results saved to {filename}')

# Main function to run the simulation
def main():
    capacity = 5
    test_data_file = 'random_client_requests.csv'  # Path to your test data file

    # Load test data
    df_test = pd.read_csv(test_data_file)
    df_test['Request_Time'] = pd.to_datetime(df_test['Request_Time'])
    test_requests = df_test['URL'].tolist()

    # Initialize and run the cache simulator
    simulator = CacheSimulator(capacity)
    cache_hits_per_request = simulator.simulate(test_requests)
    
    # Generate x values (number of requests)
    x_values = range(1, len(cache_hits_per_request) + 1)

    # Plot the cache hits vs. number of requests
    plot_cache_hits(cache_hits_per_request)
    
    # Save the results to a CSV file
    save_results_to_csv(x_values, cache_hits_per_request)

if __name__ == '__main__':
    main()
