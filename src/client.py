import requests
import time

def request_video(proxy_url, server_url, url):
    try:
        response = requests.get(f'{proxy_url}/cache/{url}', params={'server_url': server_url})
        if response.status_code == 200:
            print(f"Video '{url}' retrieved successfully.")
        else:
            print(f"Failed to retrieve video '{url}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    proxy_url = 'http://localhost:5004'
    server_url = 'http://localhost:5000'  # The actual server hosting the videos
    requests_list = [
        'sample.mp4',
        'video.mp4',
        'content.mp4'
    ]
    
    for url in requests_list:
        request_video(proxy_url, server_url, url)
        time.sleep(5)  # Wait for 5 seconds before the next request

if __name__ == '__main__':
    main()
