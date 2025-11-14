# import requests
# import json

# def check_youtube_video_existence(url):
#     """
#     Checks if a YouTube video with the given ID still exists.

#     Args:
#         video_id (str): The ID of the YouTube video.

#     Returns:
#         bool: True if the video exists, False otherwise.
#     """
#     try:
#         response = requests.get(url, timeout=5)  # Set a timeout for the request
#         # Check if the response contains "Video unavailable" or similar phrases
#         if "Video unavailable" in response.text or "This video is no longer available" in response.text:
#             return False
#         # You can also check for specific HTTP status codes if the video is truly removed
#         # For example, a 404 Not Found status might indicate removal
#         if "unavailable_video.png" in response.text:
#             return False
#         if response.status_code == 404:
#             return False
#         return True
#     except requests.exceptions.RequestException:
#         # Handle network errors, timeouts, etc.
#         return False

# # Example usage:
# # video_id_exists = "dQw4w9WgXcQ"  # A valid YouTube video ID (Rick Astley - Never Gonna Give You Up)
# # video_id_deleted = "OpA2ZxnRs6" # An example of a potentially deleted video ID

# # if check_youtube_video_existence(video_id_exists):
# #     print(f"Video {video_id_exists} exists.")
# # else:
# #     print(f"Video {video_id_exists} does not exist or is unavailable.")

# # if check_youtube_video_existence(video_id_deleted):
# #     print(f"Video {video_id_deleted} exists.")
# # else:
# #     print(f"Video {video_id_deleted} does not exist or is unavailable.")
# if __name__ == '__main__':
#    with open(r'D:\VS Code\CV\American Sign Language Translator\New_MSL_Dataset\MSL_train.json', 'r', encoding="utf-8") as f: 
#     data = json.load(f)
#     for i in range(len(data)):
#         url = data[i]['url']
#         if check_youtube_video_existence(url):
#          with open(r"D:\VS Code\CV\American Sign Language Translator\New_MSL_Dataset\MSL_NewTrain.json", "a", encoding="utf-8") as wf:
#           json.dump(data[i], wf, ensure_ascii=False)
#           wf.write(',\n')
#     # response = requests.get('https://www.youtube.com/watch?v=1AyT77LqJzQ', timeout=5)
#     # j= response.status_code
#     # print(j)
#     # print("unavailable_video.png" in response.text)
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_youtube_video_existence(url):
    """
    Checks if a YouTube video with the given URL still exists.
    Returns True if the video exists, False otherwise.
    """
    try:
        response = requests.get(url, timeout=5)
        if (
            "Video unavailable" in response.text
            or "This video is no longer available" in response.text
            or "unavailable_video.png" in response.text
            or response.status_code == 404
        ):
            return False
        return True
    except requests.exceptions.RequestException:
        return False


def process_video(entry):
    """
    Process a single video entry: check its availability and return if exists.
    """
    url = entry["url"]
    if check_youtube_video_existence(url):
        return entry
    return None


if __name__ == "__main__":
    input_path = r"D:\VS Code\CV\American Sign Language Translator\New_MSL_Dataset\MSL_train.json"
    output_path = r"D:\VS Code\CV\American Sign Language Translator\New_MSL_Dataset\MSL_NewTrain.json"

    # Load dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_videos = []
    total = len(data)
    print(f"Checking {total} videos...")

    # Use multithreading for faster processing
    max_workers = 10  # adjust number of workers depending on your CPU and network
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, entry): entry for entry in data}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                valid_videos.append(result)
            if i % 50 == 0:  # progress feedback every 50
                print(f"Checked {i}/{total} videos...")

    # Write all valid entries at once (valid JSON array)
    with open(output_path, "w", encoding="utf-8") as wf:
        json.dump(valid_videos, wf, ensure_ascii=False, indent=2)

    print(f"✅ Done! {len(valid_videos)} valid videos saved to {output_path}")
