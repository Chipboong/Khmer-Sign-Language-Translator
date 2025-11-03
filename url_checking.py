import requests
import json

def check_youtube_video_existence(url):
    """
    Checks if a YouTube video with the given ID still exists.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        bool: True if the video exists, False otherwise.
    """
    try:
        response = requests.get(url, timeout=5)  # Set a timeout for the request
        # Check if the response contains "Video unavailable" or similar phrases
        if "Video unavailable" in response.text or "This video is no longer available" in response.text:
            return False
        # You can also check for specific HTTP status codes if the video is truly removed
        # For example, a 404 Not Found status might indicate removal
        if response.status_code == 404:
            return False
        return True
    except requests.exceptions.RequestException:
        # Handle network errors, timeouts, etc.
        return False

# Example usage:
# video_id_exists = "dQw4w9WgXcQ"  # A valid YouTube video ID (Rick Astley - Never Gonna Give You Up)
# video_id_deleted = "OpA2ZxnRs6" # An example of a potentially deleted video ID

# if check_youtube_video_existence(video_id_exists):
#     print(f"Video {video_id_exists} exists.")
# else:
#     print(f"Video {video_id_exists} does not exist or is unavailable.")

# if check_youtube_video_existence(video_id_deleted):
#     print(f"Video {video_id_deleted} exists.")
# else:
#     print(f"Video {video_id_deleted} does not exist or is unavailable.")
if __name__ == '__main__':
   with open('D:\VS Code\CV\American Sign Language Translator\MS-ASL\MSASL_train.json', 'r') as f: 
    data = json.load(f)
   with open('D:\VS Code\CV\American Sign Language Translator\New_MSL_Dataset\MSL_train.json', 'r') as wf:
    new_data = json.load(wf)
    print(len(new_data))
    print(len(data))
    # for i in range(len(data)):
    #    if i > 11467:
    #     url = data[i]['url']
    #     if check_youtube_video_existence(url):
    #      with open(r"D:\VS Code\CV\American Sign Language Translator\New_MSL_Dataset\MSL_train.json", "a", encoding="utf-8") as wf:
    #       json.dump(data[i], wf, ensure_ascii=False)
    #       wf.write(',\n')