import pandas as pd
import matplotlib.pyplot as plt

# Function to read data from CSV files and plot
def read_and_plot_data(filenames):
    plt.figure(figsize=(16, 8))

    for filename in filenames:
        df = pd.read_csv(filename)
        plt.plot(df['Number_of_Requests'], df['Average_Cache_Hits'], label=filename.split('_')[0].upper() + ' Cache Hits')

    plt.xlabel('Number of Requests')
    plt.ylabel('Average Cache Hits per 100 Requests')
    plt.title('Average Cache Hits Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('cache_hits_comparison_from_csv.png')
    plt.show()

# List of CSV files to read and plot
csv_files = [
    'lfu_cache_hits.csv',
    'lru_cache_hits.csv',
    'fifo_cache_hits.csv',
    'mru_cache_hits.csv',
    'ml_cache_hits.csv'  # Including this as per your update
]

# Read data and plot
read_and_plot_data(csv_files)
