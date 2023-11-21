from googleapiclient.discovery import build
import retrieve
import base64
import requests
from openai import OpenAI

# Instantiate the OpenAI client


# Replace 'your_api_key_here' with your actual OpenAI API key
client.api_key = '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'

# The image URL to be analyzed


# Make a request to the supposed model "gpt-4-vision-preview" to analyze the image


def vision(image_url): #call gpt 4 vision using a thumbnail_url
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text", 
            "text": "Analyze the key elements of this YouTube thumbnail, such as color usage, font styles, presence of faces, body language, and overall composition. Provide a detailed report with your findings that will help understand what makes this thumbnail effective or not."
            },
            {
            "type": "image_url", 
            "image_url": {"url": thumbnail_url}
            }
        ]
        }
    ],
    max_tokens=3333  # Increased the number of tokens to get a more detailed response
    )

    # Print out the model's response
    print(response.choices[0])
        return response.choices[0]

def prepare_image_for_vision_api(thumbnail_url):
    image_response = requests.get(thumbnail_url)
    # Ensure the request was successful
    image_response.raise_for_status()
    
    # Convert image content to base64
    image_base64 = base64.b64encode(image_response.content).decode('utf-8')
    return image_base64


def get_video_thumbnail(video_id):
    # This assumes you have the video ID that you want the thumbnail for
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()

    # Extract thumbnail URL from the response
    thumbnail_url = video_response['items'][0]['snippet']['thumbnails']['high']['url']
    return thumbnail_url


def analyze_thumbnail(video_id) #Call needed functions to get data, process it and pass through gpt 4 vision to analyze
    thumbnail_url = self.get_video_thumbnail(video_id)
    prepared_image = self.prepare_image_for_vision_api(thumbnail_url):
    response = self.vision(prepared_image, "As a Youtube Expert - analyze this Video's Thumbnail. Report on the design elements and style, text design and style,facial expressions and body languge. IOdentify any other objects in the image. Rate the thumnbail from 1-10 and explain your rating.")

def get_top_videos_by_views(youtube, query, start_date, end_date, max_results=10):
    # Convert dates to YouTube API format
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Search for videos within the specified time frame
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        publishedAfter=start_date_str,
        publishedBefore=end_date_str,
        maxResults=max_results,
        order='viewCount'
    ).execute()

    # Extract video IDs from the search response
    video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

    # Get details of the videos including view count and thumbnails
    video_response = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        id=','.join(video_ids)
    ).execute()

    top_videos = []
    for item in video_response.get('items', []):
        video_id = item['id']
        title = item['snippet']['title']
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        view_count = item['statistics']['viewCount']

        top_videos.append({
            'video_id': video_id,
            'title': title,
            'thumbnail_url': thumbnail_url,
            'view_count': view_count
        })

    return top_videos

def get_top_videos_by_likes(youtube, query, start_date, end_date, max_results=10):
    # Convert dates to YouTube API format
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Search for videos within the specified time frame
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        publishedAfter=start_date_str,
        publishedBefore=end_date_str,
        maxResults=max_results
    ).execute()

    # Extract video IDs from the search response
    video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

    # Get details of the videos including likes count and thumbnails
    video_response = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        id=','.join(video_ids)
    ).execute()

    videos_with_likes = []
    for item in video_response.get('items', []):
        video_id = item['id']
        title = item['snippet']['title']
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        like_count = item['statistics'].get('likeCount', 0)

        videos_with_likes.append({
            'video_id': video_id,
            'title': title,
            'thumbnail_url': thumbnail_url,
            'like_count': int(like_count)
        })

    # Sort videos by like count
    top_videos = sorted(videos_with_likes, key=lambda x: x['like_count'], reverse=True)

    return top_videos[:max_results]
