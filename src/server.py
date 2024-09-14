from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)

# Define the directory where videos are stored
VIDEO_DIRECTORY = os.path.dirname(os.path.abspath(__file__))  # Same directory as server.py

@app.route('/video/<path:url>', methods=['GET'])
def get_video(url):
    try:
        return send_from_directory(VIDEO_DIRECTORY, url)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
