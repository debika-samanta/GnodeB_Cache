import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cache Implementations
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
            if freq - 1 in self.least_freq:
                self.least_freq[freq - 1].remove(key)
                if not self.least_freq[freq - 1]:
                    del self.least_freq[freq - 1]
            self.least_freq[freq].append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            min_freq = min(self.least_freq)
            lfu_key = self.least_freq[min_freq].popleft()
            del self.cache[lfu_key]
            del self.freq[lfu_key]
            if not self.least_freq[min_freq]:
                del self.least_freq[min_freq]
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

# Sample cache metadata generation
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

# Convert lookup_df for ML
def prepare_ml_data(lookup_df):
    lookup_df = lookup_df.copy()
    lookup_df['Data_Freshness'] = lookup_df['Data_Freshness'].map({'Fresh': 1, 'Stale': 0})
    lookup_df['Content_Type'] = pd.factorize(lookup_df['Content_Type'])[0]  # Convert categorical to numeric
    features = lookup_df[['Access_Frequency', 'Content_Size', 'Access_Frequency_Over_Time', 'Storage_Cost', 'Content_Type', 'Data_Freshness']]
    return features

# ML-based Cache Strategy
class MLCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.model = RandomForestClassifier()
        self.trained = False
    
    def train(self, features, labels):
        self.model.fit(features, labels)
        self.trained = True
    
    def get(self, key, metadata):
        if not self.trained:
            raise RuntimeError("Model not trained yet")
        
        # Ensure metadata is a DataFrame with the same format as training data
        feature = metadata[['Access_Frequency', 'Content_Size', 'Access_Frequency_Over_Time', 'Storage_Cost', 'Content_Type', 'Data_Freshness']].copy()
        
        # Convert categorical features to numeric values
        feature['Content_Type'] = feature['Content_Type'].map({
            'Social Media': 0, 'Live Video': 1, 'Blogs': 2, 'Ebook': 3,
            'Infographic': 4, 'News': 5, 'Advertisement': 6, 'Article': 7,
            'Emails': 8, 'Videos': 9, 'Authority Content': 10
        })
        feature['Data_Freshness'] = feature['Data_Freshness'].map({'Fresh': 1, 'Stale': 0})
        
        feature_values = feature.values
        
        # Predict
        prediction = self.model.predict(feature_values)
        return self.cache.get(key, None) if prediction[0] == 1 else None
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            oldest_key = list(self.cache.keys())[0]  # Simple FIFO removal for ML cache
            del self.cache[oldest_key]
        self.cache[key] = value

def simulate_and_plot(requests_df, capacity, lookup_df):
    # Initialize caches
    lfu_cache = LFUCache(capacity)
    lru_cache = LRUCache(capacity)
    fifo_cache = FIFOCache(capacity)
    mru_cache = MRUCache(capacity)
    ml_cache = MLCache(capacity)
    
    # Prepare data for ML
    features = prepare_ml_data(lookup_df)
    labels = [1 if i % 2 == 0 else 0 for i in range(len(features))]  # Example labels; adjust as necessary
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    ml_cache.train(X_train, y_train)
    
    # Track cache hits
    lfu_hits, lru_hits, fifo_hits, mru_hits, ml_hits = [], [], [], [], []
    lfu_requests, lru_requests, fifo_requests, mru_requests, ml_requests = [], [], [], [], []
    
    hit_count_lfu, hit_count_lru, hit_count_fifo, hit_count_mru, hit_count_ml = 0, 0, 0, 0, 0
    
    for idx, row in requests_df.iterrows():
        url = row['URL']
        metadata = lookup_df[lookup_df['URL'] == url].iloc[0]
        metadata = pd.DataFrame([metadata])
        
        # LFU Cache
        if lfu_cache.get(url) is not None:
            hit_count_lfu += 1
        else:
            lfu_cache.put(url, url)
        lfu_hits.append(hit_count_lfu)
        lfu_requests.append(idx + 1)
        
        # LRU Cache
        if lru_cache.get(url) is not None:
            hit_count_lru += 1
        else:
            lru_cache.put(url, url)
        lru_hits.append(hit_count_lru)
        lru_requests.append(idx + 1)
        
        # FIFO Cache
        if fifo_cache.get(url) is not None:
            hit_count_fifo += 1
        else:
            fifo_cache.put(url, url)
        fifo_hits.append(hit_count_fifo)
        fifo_requests.append(idx + 1)
        
        # MRU Cache
        if mru_cache.get(url) is not None:
            hit_count_mru += 1
        else:
            mru_cache.put(url, url)
        mru_hits.append(hit_count_mru)
        mru_requests.append(idx + 1)
        
        # ML Cache
        if ml_cache.get(url, metadata) is not None:
            hit_count_ml += 1
        else:
            ml_cache.put(url, url)
        ml_hits.append(hit_count_ml)
        ml_requests.append(idx + 1)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    plt.plot(lfu_requests, lfu_hits, label='LFU Cache Hits')
    plt.plot(lru_requests, lru_hits, label='LRU Cache Hits')
    plt.plot(fifo_requests, fifo_hits, label='FIFO Cache Hits')
    plt.plot(mru_requests, mru_hits, label='MRU Cache Hits')
    plt.plot(ml_requests, ml_hits, label='ML Cache Hits')
    plt.xlabel('Requests')
    plt.ylabel('Hits')
    plt.title('Cache Hits Comparison')
    plt.legend()
    plt.show()

# Example usage
requests_file = 'requests.csv'
lookup_file = 'lookup_table.csv'
requests_df, lookup_df = process_requests(requests_file, lookup_file)
simulate_and_plot(requests_df, capacity=10, lookup_df=lookup_df)
