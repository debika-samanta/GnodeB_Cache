import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import deque
import matplotlib.pyplot as plt

# Load the lookupT.csv data
lookup_df = pd.read_csv('lookupT.csv')

# Ensure column names match expected names and strip any extra spaces
lookup_df.columns = lookup_df.columns.str.strip()

# Encode the categorical 'Content_Type' feature
le = LabelEncoder()
lookup_df['Content_Type'] = le.fit_transform(lookup_df['Content_Type'])

# Normalize the numeric features
scaler = StandardScaler()
numeric_features = ['Access_Frequency', 'Content_Size', 'Access_Frequency_Over_Time', 
                     'Storage_Cost', 'Data_Freshness']
lookup_df[numeric_features] = scaler.fit_transform(
    lookup_df[numeric_features]
)

# Define features and target variable
X = lookup_df[numeric_features]
y = lookup_df['Cache_Hit_Miss'].apply(lambda x: 1 if x == 'Hit' else 0)  # 1 for Hit, 0 for Miss

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Load the request.csv data
request_df = pd.read_csv('requests.csv')

# Ensure column names match expected names and strip any extra spaces
request_df.columns = request_df.columns.str.strip()

# Initialize the cache and metrics
cache = deque(maxlen=50)
cache_hits = []
cache_misses = []

# Function to preprocess request data before prediction
def preprocess_request(row, scaler, le, lookup_df):
    # Ensure URL is properly formatted
    url = row['URL']
    if url not in lookup_df['URL'].values:
        raise ValueError(f"URL {url} not found in lookupT.csv")
    
    content_type = le.transform([url.split('.')[-1]])[0]
    access_frequency = lookup_df[lookup_df['URL'] == url].shape[0]
    content_size = np.random.randint(1000, 10000)  # Assuming random content size
    last_access_time = row['Request_Time']
    first_access_time = lookup_df[lookup_df['URL'] == url]['Last_Access_Time'].min()
    time_diff_days = (pd.to_datetime(last_access_time) - pd.to_datetime(first_access_time)).days + 1e-10
    access_frequency_over_time = access_frequency / time_diff_days
    storage_cost = content_size * 0.01
    time_since_last_access = (pd.to_datetime(last_access_time) - pd.to_datetime(last_access_time)).days + 1e-10
    data_freshness = 1 / time_since_last_access
    
    # Create NumPy array for the features
    features_array = np.array([
        access_frequency,
        content_size,
        access_frequency_over_time,
        storage_cost,
        data_freshness
    ]).reshape(1, -1)

    # Scale the features
    scaled_features = scaler.transform(features_array)
    return scaled_features

# Process each request
for i, row in request_df.iterrows():
    features = preprocess_request(row, scaler, le, lookup_df)
    prediction = model.predict(features)[0]
    
    if prediction == 1 and row['URL'] in cache:
        cache_hits.append(len(cache_hits) + 1)
    else:
        cache_misses.append(len(cache_misses) + 1)
        
        # Evict if necessary and append the new URL
        if len(cache) >= cache.maxlen:
            cache.popleft()
        cache.append(row['URL'])

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cache_hits) + 1), np.cumsum(cache_hits), label='Cache Hits (Cumulative)', color='green')
plt.plot(range(1, len(cache_misses) + 1), np.cumsum(cache_misses), label='Cache Misses (Cumulative)', color='red')
plt.xlabel('Number of Requests')
plt.ylabel('Cumulative Hits')
plt.title('Cache Hits vs. Number of Requests')
plt.legend()
plt.show()

