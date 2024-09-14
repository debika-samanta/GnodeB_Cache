import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Sample cache metadata generation (Replace with actual metadata generation logic)
def generate_metadata(url):
    return {
        'Access_Frequency': random.randint(1, 100),
        'Content_Size': random.randint(1, 500),  # Size in KB
        'Last_Access_Time': (datetime.now() - timedelta(seconds=random.randint(1, 1000000))).strftime('%Y-%m-%d %H:%M:%S'),
        'Access_Frequency_Over_Time': random.randint(1, 10),
        'Storage_Cost': random.randint(1, 1000),
        'Content_Type': random.choice(['Social Media', 'Live Video', 'Blogs', 'Ebook', 'Infographic', 'News', 'Advertisement', 'Article', 'Emails', 'Videos', 'Authority Content']),
        'Data_Freshness': random.choice(['Fresh', 'Stale'])
    }

# Fetch client requests and generate lookup table
def process_requests(requests_file, lookup_file):
    requests_df = pd.read_csv(requests_file)
    lookup_df = pd.DataFrame(columns=['URL', 'Access_Frequency', 'Content_Size', 'Last_Access_Time', 'Access_Frequency_Over_Time', 'Storage_Cost', 'Content_Type', 'Data_Freshness'])
    
    metadata_list = []
    for _, row in requests_df.iterrows():
        url = row['URL']
        if lookup_df[lookup_df['URL'] == url].empty:
            metadata = generate_metadata(url)
            metadata['URL'] = url
            metadata_list.append(metadata)
    
    lookup_df = pd.concat([lookup_df, pd.DataFrame(metadata_list)], ignore_index=True)
    lookup_df.to_csv(lookup_file, index=False)
    return requests_df, lookup_df

# LFU, LRU, FIFO, and MRU Cache implementations
class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.freq = defaultdict(int)
        self.least_freq = defaultdict(deque)
    
    def get(self, key):
        if key in self.cache:
            self.freq[key] += 1
            freq = self.freq[key]
            # Remove key from the old frequency list
            if freq - 1 in self.least_freq and key in self.least_freq[freq - 1]:
                self.least_freq[freq - 1].remove(key)
                if not self.least_freq[freq - 1]:
                    del self.least_freq[freq - 1]
            # Add key to the new frequency list
            self.least_freq[freq].append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            # Find the least frequently used items
            min_freq = min(self.least_freq)
            lfu_key = self.least_freq[min_freq].popleft()
            del self.cache[lfu_key]
            del self.freq[lfu_key]
            if not self.least_freq[min_freq]:
                del self.least_freq[min_freq]
        # Add the new key-value pair
        self.cache[key] = value
        self.freq[key] = 1
        self.least_freq[1].append(key)

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            lru_key = self.order.popleft()
            del self.cache[lru_key]
        self.cache[key] = value
        self.order.append(key)

class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
    
    def get(self, key):
        return self.cache.get(key, None)
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            fifo_key = self.order.popleft()
            del self.cache[fifo_key]
        self.cache[key] = value
        self.order.append(key)

class MRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
    
    def get(self, key):
        return self.cache.get(key, None)
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            mru_key = self.order.pop()
            del self.cache[mru_key]
        self.cache[key] = value
        self.order.append(key)

# Function to save the averaged results to CSV files
def save_results_to_csv(filename, requests_avg, hits_avg):
    df = pd.DataFrame({'Number_of_Requests': requests_avg, 'Average_Cache_Hits': hits_avg})
    df.to_csv(filename, index=False)
    print(f'Results saved to {filename}')

# Function to average y-values over intervals while keeping x-values fixed
def average_y_values(y_values, interval=20):
    averaged_y = []
    
    num_intervals = (len(y_values) + interval - 1) // interval
    for i in range(num_intervals):
        start = i * interval
        end = min((i + 1) * interval, len(y_values))
        
        y_segment = y_values[start:end]
        y_avg = sum(y_segment) / len(y_segment) if y_segment else 0
        
        averaged_y.append(y_avg)
    
    return averaged_y

# Simulate cache and plot results
def simulate_and_plot(requests_df, capacity, lookup_df):
    lfu_cache = LFUCache(capacity)
    lru_cache = LRUCache(capacity)
    fifo_cache = FIFOCache(capacity)
    mru_cache = MRUCache(capacity)
    
    lfu_hits = []
    lru_hits = []
    fifo_hits = []
    mru_hits = []
    
    hit_count_lfu = 0
    hit_count_lru = 0
    hit_count_fifo = 0
    hit_count_mru = 0
    
    for idx, row in requests_df.iterrows():
        url = row['URL']
        
        # LFU Cache
        if lfu_cache.get(url) is not None:
            hit_count_lfu += 1
        else:
            lfu_cache.put(url, url)
        lfu_hits.append(hit_count_lfu)
        
        # LRU Cache
        if lru_cache.get(url) is not None:
            hit_count_lru += 1
        else:
            lru_cache.put(url, url)
        lru_hits.append(hit_count_lru)
        
        # FIFO Cache
        if fifo_cache.get(url) is not None:
            hit_count_fifo += 1
        else:
            fifo_cache.put(url, url)
        fifo_hits.append(hit_count_fifo)
        
        # MRU Cache
        if mru_cache.get(url) is not None:
            hit_count_mru += 1
        else:
            mru_cache.put(url, url)
        mru_hits.append(hit_count_mru)
    
    # x-axis values (fixed intervals)
    num_requests = list(range(0, len(requests_df) + 20, 20))
    
    # Average y-values over intervals
    lfu_hits_avg = average_y_values(lfu_hits, interval=20)
    lru_hits_avg = average_y_values(lru_hits, interval=20)
    fifo_hits_avg = average_y_values(fifo_hits, interval=20)
    mru_hits_avg = average_y_values(mru_hits, interval=20)
    
    # Adjust x-axis to match averaged y-values
    requests_avg = list(range(0, len(lfu_hits_avg) * 20, 20))
    
    # Save results to CSV files
    save_results_to_csv('lfu_cache_hits.csv', requests_avg, lfu_hits_avg)
    save_results_to_csv('lru_cache_hits.csv', requests_avg, lru_hits_avg)
    save_results_to_csv('fifo_cache_hits.csv', requests_avg, fifo_hits_avg)
    save_results_to_csv('mru_cache_hits.csv', requests_avg, mru_hits_avg)
    
    # Plotting
    plt.figure(figsize=(16, 8))
    
    plt.plot(requests_avg, lfu_hits_avg, label='LFU Cache Hits')
    plt.plot(requests_avg, lru_hits_avg, label='LRU Cache Hits')
    plt.plot(requests_avg, fifo_hits_avg, label='FIFO Cache Hits')
    plt.plot(requests_avg, mru_hits_avg, label='MRU Cache Hits')
    plt.xlabel('Number of Requests')
    plt.ylabel('Average Cache Hits per 20 Requests')
    plt.title('Average Cache Hits Comparison: LFU, LRU, FIFO, and MRU')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    requests_df, lookup_df = process_requests('requests.csv', 'lookup.csv')
    simulate_and_plot(requests_df, capacity=10, lookup_df=lookup_df)

if __name__ == "__main__":
    main()
