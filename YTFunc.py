from googleapiclient.discovery import build
from googleapiclient import errors
from googleapiclient.errors import HttpError
from google.api_core.exceptions import ServiceUnavailable, GoogleAPICallError
from google.cloud import vision
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi, NoTranscriptFound
import spacy
import pprint
import os
import openai
from openai import OpenAI
from nltk.corpus import stopwords
import pytube
import whisper
import logging
import pprint
import csv
import time
import json
import math
import random
import ast
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
import sys
import pandas as pd
from datetime import datetime, timedelta
import spacy
from fpdf import FPDF
from PIL import Image
import shutil
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
from pytube import YouTube
import ffmpeg
import subprocess
import yt_dlp



#.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#spacy.cli.download("en_core_web_sm")
#nltk.download('stopwords')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YTFunc():
    def __init__(self):
        """
        Initializes the YTFunc class with necessary attributes for processing YouTube videos.
        Loads the NLP model, sets API keys, and initializes various attributes for video data management.
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/dell/Python/gold-episode-404807-f2668d34aa21.json"
        self.nlp = spacy.load("en_core_web_sm")
        self.video_data = {} # a dictionary of keys = video_ids values = dictionarys containing data relating to video.
        self.yt_v3_key = os.getenv("GOOGLE_API_KEY")
        self.openai_key =  os.getenv("OPENAI_API_KEY")
        self.youtube = build('youtube', 'v3', developerKey = self.yt_v3_key)
        openai.api_key = self.openai_key
        self.gpt_client = openai.OpenAI(
        api_key=self.openai_key,
        organization='org-zlvexewsMwb1sDoESZRaNq6u',
        project='proj_Y9r8e8wUWckQGpcx316JvNq7'
        )
        self.valid_videos = [] 
        self.pp = pprint.PrettyPrinter(indent=4)
        self.last_subject_searched = ''
        self.videos_downloadable =[]
        self.next_page_token = ''
        self.script_report = []
        self.script_pre_prompt = "Analyze the following youtube script: " 
        self.script_prompts = ["Content Type: Classify as educational, entertainment, or a blend.", "Target Audience: Identify problems/pain points, interests, and desires.", "Introduction Quality: Evaluate opening lines for engagement and clarity of expectations.", "Narrative Structure: Assess structure (problem-solution, chronological, storytelling) for audience interest.", "Pacing and Rhythm: Examine content flow and balance between information and engagement.", "CTAs: Analyze placement and nature of calls to action.", "Language and Tone: Review language appropriateness and tone consistency.", "Storytelling Elements: Examine anecdotes, metaphors, and personal stories.", "Educational Content Delivery: For educational scripts, assess clarity and structure of information.", "Entertainment Value: For entertainment scripts, evaluate humor, drama, suspense.", "Viewer Engagement Strategies: Analyze interactive elements like questions and community involvement.", "Use of Keywords: Assess SEO-friendly keywords.", "Conclusion Effectiveness: Evaluate conclusion for summarization and viewer direction.", "Length and Segmentation: Review script length and topic segmentation.", "Cross-Platform References: Identify social media and platform integration.", "Data and Fact Accuracy: Verify accuracy of presented information."]
        self.script_post_prompt = " | using the LEAST amount of words possible while focusing your analysis ONLY on the topic of - "
        logging.info("YTFunc instance created.\n")
        self.subject_tokens = {}
        self.file_name = ''
        self.numeric_fields_as_int = True # Set to false for functions that need data that are numbers to be Strings (not fully implemented in logic)
        # Increase the maximum number of open figures before warning
        plt.rcParams['figure.max_open_warning'] = 150  # Change this number as needed

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        #nltk.download('punkt')
        #nltk.download('averaged_perceptron_tagger')
        #self.number_of_videos_max = number_of_videos_max
    
    def get_channel_id(self, username):
        request = self.youtube.channels().list(
        part='id',
        forUsername=username
        )
        response = request.execute()
        if 'items' in response and response['items']:
            print("Channel ID: " + response['items'][0]['id'])
            return response['items'][0]['id']
        else:
            print("No channel ID found")
            return None
        
    def get_top_videos_channel(self, channel_id, query, max_results = 10, next_page_token = None): # Find top (max_results) videos based on views and using (next_page_token). Retuns list of video ids
        """
        Search YouTube API for top videos by viewCount on a given query.
        Handles pagination with next page token.
        use get_channel_id("youtube_account_name") for the first parameter channel_id
        Returns a list of video IDs up to max_results.
        """
        self.logger.info("Starting search for top videos for query: '%s'", query)
        self.last_subject_searched = query
        valid_videos = []
        search_attempt = 0
        backoff_time = 1
        max_backoff_time = 32
        saved_due_to_quota = False
        retry_attempt = 0

        while len(valid_videos) < max_results and search_attempt < 10:
            try:
                request_params = {
                        'q': query,
                        'part': 'snippet',
                        'type': 'video',
                        'maxResults': max_results - len(valid_videos),
                        'order': 'viewCount',
                        'channelId': channel_id
                    }
                # Add the pageToken parameter only if next_page_token is valid
                if next_page_token and next_page_token.lower() != "nan":
                    request_params['pageToken'] = next_page_token

                request = self.youtube.search().list(**request_params)
                response = request.execute()
                #  Make the request

                # Update next_page_token with the new token from the response
                next_page_token = response.get('nextPageToken')
                self.next_page_token = next_page_token

                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    if video_id not in self.video_data.keys():
                        valid_videos.append(video_id)
                    if len(valid_videos) >= max_results:
                        break
                    
                if len(valid_videos) >= max_results:
                        break
                
                self.logger.info("Found %d videos", len(valid_videos))
                valid_videos = list(set(valid_videos))
                self.logger.info(f"Videos after duplicates: {len(valid_videos)}")
                backoff_time = 1

            except errors.HttpError as e:
                retry_attempt += 1
                if retry_attempt > 5:
                    self.logger.info(f"Saving {len(self.video_data.keys())} videos to file due to excessive retries.\nSaved to quota_exceeded_for_{self.last_subject_searched}.csv")
                    self.load_data_from_ids(valid_videos)
                    self.save_video_data_to_csv(f"quota_exceeded_for_{self.last_subject_searched}.csv")
                    saved_due_to_quota = True
                    return

                if e.resp.status in [403, 429] and not saved_due_to_quota:
                    self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                else:
                    self.logger.error("HTTP error occurred: %s", e)

                print(f"Retry attempt number: {retry_attempt}")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)

            except Exception as e:
                self.logger.error("Error during search request: %s", e)
                self.logger.warning("Backing off for %d seconds.", backoff_time)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                retry_attempt += 1
                if retry_attempt > 6:
                    self.load_data_from_ids(valid_videos)
                    self.save_video_data_to_csv(f"quota_exceeded_for_{self.last_subject_searched}.csv")
                    saved_due_to_quota = True
                    return
                continue

            finally:
                search_attempt += 1

        # Process chunks of video IDs
        if valid_videos:
            self.logger.info("Found %d valid videos for query: '%s'", len(valid_videos), query)
            self.load_data_from_ids(valid_videos)

        self.logger.info(f"get_top_videos() is now Returning top videos from {channel_id} a list of video ids")
        return valid_videos
    
    def get_recent_videos(self, query, max_results = 10, next_page_token = None, time_frame = "week"):
        """
        Search YouTube API for recent videos related to a given subject, sorted by views.
        Handles pagination with next page token.
        The time_frame can be 'hour', 'today', 'week', 'month', or 'year'.
        Returns a list of video IDs up to max_results.
        """
        self.logger.info("Starting search for recent videos for query: '%s'", query)
        self.last_subject_searched = query
        valid_videos = []
        search_attempt = 0
        backoff_time = 1
        max_backoff_time = 32

        while len(valid_videos) < max_results and search_attempt < 10:
            try:
                request = self.youtube.search().list(
                    q=query,
                    part='snippet',
                    type='video',
                    maxResults=max_results,
                    order='viewCount',  # Order by upload date
                    publishedAfter=self._get_published_after(time_frame),  # Filter by upload date
                    pageToken=next_page_token
                )
                response = request.execute()
                next_page_token = response.get('nextPageToken')
                self.next_page_token = next_page_token

                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    if video_id not in self.video_data.keys():
                        valid_videos.append(video_id)
                    if len(valid_videos) >= max_results:
                        break

                self.logger.debug("Found %d videos", len(valid_videos))

                backoff_time = 1

            except errors.HttpError as e:
                if e.resp.status in [403, 429]:
                    self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                else:
                    self.logger.error("HTTP error occurred: %s", e)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)

            except Exception as e:
                self.logger.error("Error during search request: %s", e)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                continue

            finally:
                search_attempt += 1
                self.pp.pprint(search_attempt)
        
        return valid_videos

    def _get_published_after(self, time_frame):
        """
        Utility function to calculate the 'publishedAfter' parameter based on the time frame.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        if time_frame == "hour":
            delta = datetime.timedelta(hours=1)
        elif time_frame == "today":
            delta = datetime.timedelta(days=1)
        elif time_frame == "week":
            delta = datetime.timedelta(weeks=1)
        elif time_frame == "month":
            delta = datetime.timedelta(days=30)
        elif time_frame == "year":
            delta = datetime.timedelta(days=365)
        else:
            raise ValueError("Invalid time frame specified")

        return (now - delta).isoformat()
    
    def get_top_videos_by_likes(self, query, max_results=10, next_page_token=None):
        """

        Search YouTube API for top videos by likeCount on a given query.
        Handles pagination with next page token.
        Returns a list of video IDs up to max_results.
        """
        self.logger.info("Starting search for top videos by likes for query: '%s'", query)
        self.last_subject_searched = query
        valid_videos = []
        search_attempt = 0
        backoff_time = 1
        max_backoff_time = 32

        while len(valid_videos) < max_results and search_attempt < 10:
            try:
                request = self.youtube.search().list(
                    q=query,
                    part='snippet',
                    type='video',
                    maxResults=max_results,
                    order='likeCount',  # Changed from 'viewCount' to 'likeCount' - 'likeCount' not recognized to sorty
                    pageToken=next_page_token
                )
                response = request.execute()
                next_page_token = response.get('nextPageToken')
                self.next_page_token = next_page_token

                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    if video_id not in self.video_data.keys():
                        valid_videos.append(video_id)
                    if len(valid_videos) >= max_results:
                        break
                self.logger.debug("Found %d videos", len(valid_videos))

                backoff_time = 1  # Reset the backoff time after a successful attempt

                # Implement backoff logic here if needed

            except errors.HttpError as e:
                if e.resp.status in [403, 429]:
                    self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                else:
                    self.logger.error("HTTP error occurred: %s", e)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
            except Exception as e:
                self.logger.exception("An error occurred: %s", e)
                search_attempt += 1
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time)
            finally:
                search_attempt +=1
                self.pp.pprint(search_attempt)

        self.load_data_from_ids(valid_videos)

        return valid_videos
    
    def get_top_videos(self, query, max_results = 10, next_page_token = None): # Find top (max_results) videos based on views and using (next_page_token). Retuns list of video ids
        """
        Search YouTube API for top videos by viewCount on a given query.
        Handles pagination with next page token.
        Returns a list of video IDs up to max_results.
        """
        self.logger.info("Starting search for top videos for query: '%s'", query)
        self.last_subject_searched = query
        valid_videos = []
        search_attempt = 0
        backoff_time = 1
        max_backoff_time = 32
        saved_due_to_quota = False
        retry_attempt = 0

        while len(valid_videos) < max_results and search_attempt < 5:
            try:
                
                request_params = {
                        'q': query,
                        'part': 'snippet',
                        'type': 'video',
                        'maxResults': max_results - len(valid_videos),
                        'order': 'viewCount'
                    }
                # Add the pageToken parameter only if next_page_token is valid
                if next_page_token not in ["nan", None, "None"]:
                    request_params['pageToken'] = next_page_token

                request = self.youtube.search().list(**request_params)
                response = request.execute()
                #  Make the request

                # Update next_page_token with the new token from the response
                next_page_token = response.get('nextPageToken')
                self.next_page_token = next_page_token

                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    if video_id not in self.video_data.keys():
                        valid_videos.append(video_id)
                    if len(valid_videos) >= max_results:
                        break
                
                self.logger.info("Found %d videos", len(valid_videos))
                valid_videos = list(set(valid_videos))
                self.logger.info(f"Videos after duplicates: {len(valid_videos)}")
                backoff_time = 1

            except errors.HttpError as e:
                if e.resp.status in [403, 429] and not saved_due_to_quota:
                    self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                    if retry_attempt >= 5:
                        self.logger.info(f"Saving {len(self.video_data.keys())} videos to to file due to excessive retries.\n Saved to quota_exceeded_for_{self.last_subject_searched}.csv")
                        self.save_video_data_to_csv(f"quota_exceeded_for_{self.last_subject_searched}.csv")
                        saved_due_to_quota = True
                        return
                else:
                    self.logger.error("HTTP error occurred: %s", e)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)

            except Exception as e:
                self.logger.error("Error during search request: %s", e)
                self.logger.warning("Backing off for %d seconds.", backoff_time)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                retry_attempt += 1
                if retry_attempt > 8:
                    return
                continue


            finally:
                search_attempt += 1
                self.pp.pprint(search_attempt)

        # Process chunks of video IDs
        if valid_videos:
            self.logger.info("Found %d valid videos for query: '%s'", len(valid_videos), query)
            self.load_data_from_ids(valid_videos)

        self.logger.info("get_top_videos() is complete. Returning top videos as a list of video ids")
        self.logger.info(f"list of video ids: {valid_videos}")
        
        if valid_videos:
                self.logger.info("Found %d valid videos for query: '%s'", len(valid_videos), query)
                self.load_data_from_ids(valid_videos)

        return valid_videos
    
    def get_playlist_id(self, channel_name):
        try:
            channel_response = self.youtube.channels().list(
                part='contentDetails', forUsername=channel_name).execute()

            # Check if 'items' key is in the response
            if 'items' not in channel_response or not channel_response['items']:
                self.logger.error(f"No channel found with the name {channel_name} or no items in response.")
                return None

            content_details = channel_response['items'][0].get('contentDetails')
            if content_details:
                uploads_playlist_id = content_details['relatedPlaylists']['uploads']
                return uploads_playlist_id
            else:
                self.logger.error(f"No content details found for channel {channel_name}")
                return None

        except HttpError as e:
            self.logger.error(f"An HTTP error occurred: {e}")
            return None
        except KeyError as e:
            self.logger.error(f"A KeyError occurred: {e}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return None
    
    def search_youtube_videos(self, query = None, max_results=10, next_page_token=None, **kwargs):
        """
        Search YouTube API for videos based on different criteria.
        Allows for sorting by views, likes, etc., and filtering by channel ID, playlist ID,
        and time frame for recent videos.
        General Purpose version of get_top_vdeos

        Parameters:
        - query (str): The search query for finding videos.
        - max_results (int, optional): The maximum number of video IDs to return. Default is 10.
        - next_page_token (str, optional): Token for pagination. Default is None.

        Keyword Arguments:
        - sort_order (str, optional): Specifies the sort order of the search results.
            Possible values include 'viewCount', 'likeCount', etc. Defaults to 'viewCount'.
        - channel_id (str, optional): Filters search results to a specific YouTube channel. Default is None.
        - playlist_id (str, optional): Filters search results to a specific YouTube playlist. Default is None.
        - time_frame (str, optional): Specifies the time frame for recent videos. 
            Possible values are 'hour', 'today', 'week', 'month', or 'year'. Only applicable when sort_order is 'viewCount'.

        Returns:
        - list: A list of video IDs that match the search criteria up to max_results.
        """
        sort_order = kwargs.get('sort_order', 'viewCount')  # Default to viewCount if not specified
        channel_id = kwargs.get('channel_id', None)
        playlist_id = kwargs.get('playlist_id', None)
        time_frame = kwargs.get('time_frame', None)  # Used for recent videos

        valid_videos = []
        search_attempt = 0
        retry_attempt = 0
        backoff_time = 1
        max_backoff_time = 32

        while len(valid_videos) < max_results and search_attempt < 10:
            try:
                request_params = {
                    'part': 'snippet',
                    'type': 'video',
                    'maxResults': max_results - len(valid_videos),
                    'order': sort_order
                }
                if query:
                    request_params['q'] = query
                if next_page_token not in ["nan", None, "None"]:
                    request_params['pageToken'] = next_page_token
                if channel_id:
                    request_params['channelId'] = channel_id
                if playlist_id:
                    request_params['playlistId'] = playlist_id
                if time_frame and sort_order == 'viewCount':
                    request_params['publishedAfter'] = self._get_published_after(time_frame)

                response = self.youtube.search().list(**request_params).execute()
                next_page_token = response.get('nextPageToken')
                self.next_page_token = next_page_token

                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    if video_id not in self.video_data.keys():
                        valid_videos.append(video_id)
                    if len(valid_videos) >= max_results:
                        break

                self.logger.info("Found %d videos", len(valid_videos))
                valid_videos = list(set(valid_videos))
                backoff_time = 1
                search_attempt += 1

            except errors.HttpError as e:
                if e.resp.status in [403, 429] and not saved_due_to_quota:
                    self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                    if retry_attempt >= 5:
                        self.logger.info(f"Saving {len(self.video_data.keys())} videos to to file due to excessive retries.\n Saved to quota_exceeded_for_{self.last_subject_searched}.csv")
                        self.save_video_data_to_csv(f"quota_exceeded_for_{self.last_subject_searched}.csv")
                        saved_due_to_quota = True
                        return
                else:
                    self.logger.error("HTTP error occurred: %s", e)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)

            except Exception as e:
                self.logger.error("Error during search request: %s", e)
                self.logger.warning("Backing off for %d seconds.", backoff_time)
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                retry_attempt += 1
                if retry_attempt > 8:
                    return
                continue
            self.load_data_from_ids(valid_videos)

            if valid_videos:
                self.logger.info("Found %d valid videos for query: '%s'", len(valid_videos), query)
                self.load_data_from_ids(valid_videos)

            self.logger.info(f"get_top_videos() is complete after {search_attempt} calls. Returning top videos as a list of video ids")
            self.logger.info(f"list of video ids: {valid_videos}")
            return valid_videos
        
    def load_data_from_ids(self, video_ids, max_backoff_time = 240): #call Youtube data API by chunking video_ids 
        # Split video IDs into smaller chunks if necessary
        """
        Call YouTube API to retrieve details and statistics 
        for a list of video IDs.  
        Handles chunking API requests if too many IDs provided.
        Checks to see if any existing video IDs are within chunks to avoid duplicates.
        """
        self.logger.info("Loading video details for %d IDs", len(video_ids))

        video_id_chunks = [video_ids[i:i + 50] for i in range(0, len(video_ids), 50)]

        self.logger.debug("Created %s chunks of 'video_ids' passed into load_data_from_ids()", len(video_id_chunks))
        video_id_chunks = self.remove_existing_video_ids(video_id_chunks)
        #aboved noted out due to a unhashable type :'list' error

        for chunk in video_id_chunks:
            backoff_time = 30
            video_id_str = ','.join(chunk)

            try:
                video_request = self.youtube.videos().list(
                    part='snippet,contentDetails,statistics',
                    id=video_id_str
                )
                video_response = video_request.execute()

                for item in video_response['items']:
                    video_detail = self.extract_video_detail(item)
                    self.store_video_detail(item['id'], video_detail)

            except Exception as e:
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                self.logger.error(
                    "An error occurred during YouTube API request. "
                    "Details:\n"
                    " - Error Type: %s\n"
                    " - Error Message: %s\n"
                    " - Chunk Size: %d\n"
                    " - Chunk Data: %s\n"
                    " - Video IDs in the Failed Chunk: %s",
                    type(e).__name__,
                    str(e),
                    len(chunk),
                    str(chunk),
                    video_id_str
                )
        
        self.logger.debug("Processed %d chunks of video IDs", len(video_id_chunks))

    def extract_video_detail(self, item): #Using youtube API item as input to create list of dictonaries
        try:
            current_date_time = datetime.now()
            formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")

            video_detail = {
                'video_id': item['id'],
                'subject': self.last_subject_searched,
                'title': item['snippet'].get('title', ''),
                'description': item['snippet'].get('description', ''),
                'tags': item['snippet'].get('tags', []),
                'url': f'https://www.youtube.com/watch?v={item["id"]}',
                'view_count': item['statistics'].get('viewCount', '0'),
                'like_count': item['statistics'].get('likeCount', '0'),
                'comment_count': item['statistics'].get('commentCount', '0'),  # Adding comment count here
                'upload_date': item['snippet'].get('publishedAt', ''),
                'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                'next_page_token': self.next_page_token,
                'token_time_stamp': formatted_date_time
            }
            #print("IN (extract_video_detail) HERE IS (video_detail) VARIABLE \n")
            #self.pp.pprint(video_detail)
            self.logger.debug(f"Extracting details for video {video_detail['video_id']}")
            return video_detail
        except KeyError as e:
            self.logger.error("KeyError in extract_video_detail: Missing key %s in item %s", str(e), item)
            raise
        except Exception as e:
            self.logger.debug(f"General exception in extract_video_detail: {type(e).__name__}, Error message: {e}")
            raise 


    def store_video_detail(self, video_id, video_detail): #Store information from video id as a list of dictionaries into self.video_data dictionary
        try:
            if video_id in self.video_data:
                self.video_data[video_id].append(video_detail)
            else:
                self.video_data[video_id] = [video_detail]
            self.logger.debug(f"Added details for {video_id} To self.video_data \n")
        except KeyError as e:
            self.logger.error("KeyError in store_video_detail: Missing key %s for video_id %s", str(e), video_id)
            raise 
        except Exception as e:
            self.logger.error("General exception in store_video_detail: Exception type: %s, Error message: %s, Video ID: %s", type(e).__name__, str(e), video_id)
            raise
    
    def remove_existing_video_ids(self, video_id_chunks):
        """
        Removes video IDs from the chunks that are already present in self.video_data.

        Parameters:
        video_id_chunks (list of list of str): List of chunks, where each chunk is a list of video IDs.
        """
        self.logger.info(f"Cleaning up this following chunk:\n {video_id_chunks}")

        cleaned_video_id_chunks = []
        for chunk in video_id_chunks:
            cleaned_chunk = [video_id for video_id in chunk if video_id not in self.video_data]
            cleaned_video_id_chunks.append(cleaned_chunk)

        return cleaned_video_id_chunks
            
    def list_of_next_page_tokens(self): #Helper function to generate list of unique next_page_tokens
        # Using a set to avoid duplicates
        token_set = set()
        for video_data_list in self.video_data.values():
            for video_data in video_data_list:
                if 'next_page_token' in video_data and video_data['next_page_token']:
                    token_set.add(video_data['next_page_token'])

        return list(token_set)

    def search_next_page_for_subjects(self, num_vids): #Conduct search using next page information
        """
        Conducts a search for each subject using the next page token and updates the tokens.
        """
        subjects_and_tokens = self.subject_to_tokens_dict()
        total_tokens = sum(len(str(token)) for tokens in subjects_and_tokens.values() for token in tokens)

        self.logger.info(f"Number of Subjects: {len(subjects_and_tokens.keys())} - Number of Next Page Tokens {total_tokens}")
        
        for subject, tokens in subjects_and_tokens.items():
            for token in tokens:
                if token is not None and token.strip() != "":
                    self.logger.info(f"Fetching videos for subject {subject} with token {token}")
                    self.get_top_videos(subject, max_results=num_vids, next_page_token=token)


    def subject_to_tokens_dict(self): # Helper function to produce dictonariy containing a list[next page tokens] based on subject (key)
        """
        Builds a dictionary mapping subjects to their next page tokens.
        """
        subject_token_dict = {}

        for video_data_list in self.video_data.values():
            for video_data in video_data_list:
                subject = video_data.get('subject', '')
                token = video_data.get('next_page_token', '')

                if subject and token:
                    if subject not in subject_token_dict:
                        subject_token_dict[subject] = []

                    if token not in subject_token_dict[subject]:
                        subject_token_dict[subject].append(str(token))
        
        return subject_token_dict
    
    def find_min_view_count(self, subject):
        """
        Finds the minimum view count among the videos for a given subject.

        Parameters:
        subject (str): The subject to search for in the video data profile.

        Returns:
        int: The minimum view count or None if no videos are found.
        """
        min_views = None
        for video_id, video_data_list in self.video_data.items():
            for video_data in video_data_list:
                if video_data.get('subject') == subject:
                    view_count = int(video_data.get('view_count', 0))
                    if min_views is None or view_count < min_views:
                        min_views = view_count
        return min_views

    def is_transcript_available(self, video_id):
        try:
            YouTubeTranscriptApi.get_transcript(video_id)
            return True
        except (NoTranscriptFound, TranscriptsDisabled):
            return False

    def is_video_downloadable(self, video_id):
        try:
            yt = YouTube(video_id)
            if yt.streams:
                return True
            return False
        except pytube_exceptions.PytubeError as e:
            print(f"Error checking downloadability for video ID {video_id}: {e}")
            return False
        
   


    def download_youtube_audio_clip(self, url, output_filename, start_time=None, end_time=None):
        download_dir = os.path.expanduser("~/Music/yt-dl")
        
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            logging.info(f"Created directory {download_dir}.")
        
        try:
            logging.info("Starting download of YouTube audio.")
            download_path = self._download_with_ytdlp(url, download_dir, is_audio=True, output_filename=output_filename)
            logging.info(f"Downloaded audio to {download_path}")

            try:
                output_path = os.path.join(download_dir, output_filename)
                
                if start_time and end_time:
                    logging.info("Trimming the audio.")
                    input_audio = ffmpeg.input(download_path, ss=start_time, to=end_time)
                    ffmpeg.output(input_audio, output_path, format='mp3').run(overwrite_output=True)
                    logging.info(f"Trimmed audio saved to {output_path}.")
                else:
                    os.rename(download_path, output_path)
                    logging.info(f"Full audio saved to {output_path} without trimming.")
                    
            except ffmpeg.Error as e:
                logging.error(f"FFmpeg error: {e}")
                raise

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

    def download_video(self, url, output_filename=None):
        download_dir = os.path.expanduser("~/Videos/yt-dl")
        
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            logging.info(f"Created directory {download_dir}.")

        try:
            logging.info("Starting download of YouTube video.")
            download_path = self._download_with_ytdlp(url, download_dir, is_audio=False, output_filename=output_filename)
            logging.info(f"Downloaded video to {download_path}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

    def _download_with_ytdlp(self, url, download_dir, is_audio=True, output_filename=None):
        try:
            if output_filename:
                base, ext = os.path.splitext(output_filename)
                if not ext:
                    ext = ".mp4" if not is_audio else ".mp3"
                ydl_opts = {
                    'format': 'bestaudio/best' if is_audio else 'bestvideo+bestaudio/best',
                    'outtmpl': os.path.join(download_dir, f'{base}{ext}'),
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}] if is_audio else [],
                    'merge_output_format': 'mp4' if not is_audio else None
                }
            else:
                ydl_opts = {
                    'format': 'bestaudio/best' if is_audio else 'bestvideo+bestaudio/best',
                    'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}] if is_audio else [],
                    'merge_output_format': 'mp4' if not is_audio else None
                }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            downloaded_file = None
            for file in os.listdir(download_dir):
                if (is_audio and file.endswith(".mp3")) or (not is_audio and file.endswith(".mp4")):
                    downloaded_file = os.path.join(download_dir, file)
                    break

            if not downloaded_file:
                raise FileNotFoundError("Downloaded file not found.")

            return downloaded_file

        except Exception as e:
            logging.error(f"An error occurred during download: {e}")
            raise



    def download_youtube_video_clip(url, start_time, end_time, output_filename="video_clip.mp4"):
        """
        Downloads a specific segment of a YouTube video.

        Args:
            url (str): The URL of the YouTube video.
            start_time (str): The start time of the clip in the format 'HH:MM:SS'.
            end_time (str): The end time of the clip in the format 'HH:MM:SS'.
            output_filename (str): The filename to save the trimmed video.

        Returns:
            None
        """
        # Directory where the full YouTube video will be downloaded
        download_dir = os.path.expanduser("~/Videos/yt-dl")
        
        # Ensure the directories exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            logging.info(f"Created directory {download_dir}.")
        
        try:
            # Download the video
            logging.info("Starting download of YouTube video.")
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                'merge_output_format': 'mp4'
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the downloaded file
            download_path = None
            for file in os.listdir(download_dir):
                if file.endswith(".mp4"):
                    download_path = os.path.join(download_dir, file)
                    break

            if not download_path:
                raise FileNotFoundError("Downloaded video file not found.")
            
            logging.info(f"Downloaded video to {download_path}.")

            try:
                # Full path for the trimmed video output
                output_path = os.path.join(download_dir, output_filename)
                
                # Trim the video using FFmpeg
                logging.info("Trimming the video.")
                input_video = ffmpeg.input(download_path, ss=start_time, to=end_time)
                # Ensure the output is in a widely compatible format like MP4
                ffmpeg.output(input_video, output_path, format='mp4').run(overwrite_output=True)
                logging.info(f"Trimmed video saved to {output_path}.")

            except ffmpeg.Error as e:
                logging.error(f"FFmpeg error: {e}")
                raise

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

    def extract_images(self, video_path, output_dir, fps=3, duration=None, start_time=0):
        """
        Extracts images from a video file at a specified number of frames per second.

        Args:
            video_path (str): The path to the video file.
            output_dir (str): The directory to save the extracted images.
            fps (int): The number of frames to extract per second. Default is 3.
            duration (int, optional): The duration (in seconds) to extract frames from. If not specified, extraction continues until the end of the video.
            start_time (int): The start time (in seconds) to begin extraction. Default is 0.

        Raises:
            subprocess.CalledProcessError: If the ffmpeg command fails.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory {output_dir}.")
        
        # Construct the ffmpeg command
        if duration:
            command = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-vf', f'fps={fps}',
                os.path.join(output_dir, 'frame_%04d.png')
            ]
        else:
            command = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_path,
                '-vf', f'fps={fps}',
                os.path.join(output_dir, 'frame_%04d.png')
            ]
        
        try:
            subprocess.run(command, check=True)
            logging.info(f"Extracted images to {output_dir}")
        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred while extracting images: {e}")
            raise

    def is_video_valid(self, video_id):
        transcript_available = self.is_transcript_available(video_id)
        downloadable = self.is_video_downloadable(video_id)

        if transcript_available:
            print(f"Transcript available for video ID: {video_id}")
        else:
            print(f"No transcript available for video ID: {video_id}")

        if downloadable:
            print(f"Video is downloadable for video ID: {video_id}")
            self.videos_downloadable.append(video_id)
        else:
            print(f"Video is not downloadable for video ID: {video_id}")

        return transcript_available


    def get_transcripts(self, video_ids):
        """
        Fetches transcripts for a list of YouTube video IDs and updates the video data in self.video_data.

        Args:
            video_ids (list): A list of video IDs.
        """
        backoff_time = 1
        max_backoff_time = 32
        saved_due_to_quota = False
        retry_attempt = 0

        for video_id in video_ids:
            if video_id in self.video_data:
                if 'transcript' in self.video_data[video_id][0]:  # Assuming the first dictionary has the 'transcript' key
                    self.logger.info(f"Transcript already loaded for video ID: {video_id}. Skipping.")
                    continue

                for each_video_data in self.video_data[video_id]:  # Iterate through the list of dictionaries
                    while not saved_due_to_quota and retry_attempt < 5:
                        try:
                            # Attempt to fetch the transcript
                            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                            transcript = transcript_list.find_generated_transcript(['en']).fetch()
                            text = ' '.join([t['text'] for t in transcript])

                            # Update video data with transcript
                            each_video_data['transcript'] = text  # Set transcript for each dictionary
                            self.logger.info(f"Transcript added for video ID: {video_id}")
                            break  # Exit the while loop after successful fetch

                        except (NoTranscriptFound, TranscriptsDisabled) as e:
                            self.logger.warning(f"No transcript found for video ID: {video_id}. Error: {e}")
                            each_video_data['transcript'] = "Transcript NOT Found"  # Set transcript for each dictionary
                            break  # Exit the while loop as transcript is not available

                        except errors.HttpError as e:
                            if e.resp.status in [403, 429]:
                                self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                                time.sleep(backoff_time)
                                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                                retry_attempt += 1
                                if retry_attempt >= 5:
                                    self.logger.info(f"Saving {len(self.video_data.keys())} videos to file due to excessive retries.\n Saved to quota_exceeded_for_{video_id}.csv")
                                    self.save_video_data_to_csv(f"quota_exceeded_for_{video_id}.csv")
                                    saved_due_to_quota = True
                                    break  # Exit the while loop as retry limit reached
                            else:
                                self.logger.error("HTTP error occurred: %s", e)
                                break  # Exit the while loop as it's not a rate limit error

                        except Exception as e:
                            self.logger.exception(f"Error occurred for video ID: {video_id}. Error: {e}")
                            self.logger.warning("Backing off for %d seconds.", backoff_time)
                            time.sleep(backoff_time)
                            backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                            retry_attempt += 1
                            if retry_attempt > 8:
                                break  # Exit the while loop as maximum retries reached

            else:
                self.logger.warning(f"Video ID {video_id} not found in video_data_profile.")
        # Extract filename and extension
        filename, file_extension = os.path.splitext(self.file_name)
        
        # Append to filename and add the extension back
        modified_file_name = filename + "_ts" + file_extension
        
        # Save the data to the modified file name
        print("Saving updated transcript data to file.")
        self.save_video_data_to_csv(modified_file_name)


    def analyze_transcripts(self, seperate_save=False):
        script_reports = []
        save_filename_suffix = "_tsa"
        new_data_list = []  # List to store newly fetched data for separate save

        # Define a delay between API calls
        delay_between_requests = 3  # Number of seconds to wait between each request

        for video_id, video_data_list in self.video_data.items():
            for video_data in video_data_list:
                transcript = video_data.get('transcript', 'value not found')

                # Ensure transcript is a string
                if isinstance(transcript, float):
                    transcript = str(transcript)

                if transcript and transcript not in ['value not found', 'No transcript available', '', None, '[]', 'NaN']:
                    logging.info(f"Analyzing transcript from video: {video_data.get('title', 'Unknown Title')}")

                    for prompt in self.script_prompts:
                        try:
                            logging.info(f"Asking for: {prompt}")
                            full_text = self.script_pre_prompt + str(transcript) + self.script_post_prompt + prompt
                            script_response = self.gpt_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are an expert at video content creation and a YouTube Script Analyst"},
                                    {"role": "user", "content": full_text}
                                ]
                            )

                            script_report = script_response.choices[0].message.content
                            script_reports.append(f"SCRIPT REPORT FOR | Video id: {video_id}: {script_report}\n")
                            video_data[prompt] = script_report
                            logging.info(f"Analysis completed for video: {video_data.get('title', 'Unknown Title')}")

                            if seperate_save:
                                new_data = video_data.copy()
                                new_data_list.append(new_data)
                                self.save_video_data_to_csv(new_data_list, self.file_name + save_filename_suffix)
                                logging.info(f"New data saved to {self.file_name + save_filename_suffix} after processing video ID: {video_id}.")

                        except Exception as e:
                            logging.error(f"An error occurred during GPT-4 analysis for video {video_id}: {e}")
                            video_data[prompt] = f"Error occurred: {e}"

                        # Introduce a delay between API calls
                        time.sleep(delay_between_requests)

                else:
                    logging.warning(f"No valid transcript for video: {video_data.get('title', 'Unknown Title')}")

        self.save_video_data_to_csv(self.file_name + save_filename_suffix)
        logging.info(f"Video data saved to {self.file_name + save_filename_suffix} after analyzing all transcripts.")
        logging.info("All transcripts have been analyzed.")
        return script_reports

    def analyze_comments(self):
        """
        Analyze the comments of each video in the self.video_data using OpenAI's GPT-4.
        """
        base_delay = 1.0  # Base delay in seconds for controlling request rate
        max_retries = 5  # Maximum number of retries for handling 429 errors

        for video_id, video_data_list in self.video_data.items():
            for video_data in video_data_list:
                comments = video_data.get('comment_log', [])

                # If comments is a string that represents a list, convert it back to a list
                if isinstance(comments, str):
                    try:
                        comments = ast.literal_eval(comments)
                    except (ValueError, SyntaxError):
                        logging.error(f"Error parsing comments for video ID {video_id}. Skipping.")
                        continue

                # Ensure comments is a list of strings
                if not isinstance(comments, list) or not all(isinstance(comment, str) for comment in comments):
                    logging.warning(f"Invalid comments format for video ID {video_id}. Expected a list of strings. Skipping.")
                    continue

                if comments and 'comments_analysis' not in video_data:
                    logging.info(f"Analyzing comments from video: {video_data.get('title', 'Unknown Title')}")

                    comments_text = " ".join(comments)
                    prompt = "Analyze the following comments from a YouTube video to identify and summarize key themes, popular opinions, and frequent complaints. Provide detailed feedback on what viewers liked and disliked, and offer actionable insights for creating new content that aligns with audience preferences and addresses common criticisms."


                    retry_attempt = 0
                    delay = base_delay
                    while retry_attempt < max_retries:
                        try:
                            analysis_request = self.gpt_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are an expert at analyzing audience sentiments and providing valuable insightful summaries."},
                                    {"role": "user", "content": prompt + "\n\n" + comments_text}
                                ]
                            )

                            comments_analysis = analysis_request.choices[0].message.content
                            logging.info(f"Comments analysis for video: {video_data.get('title', 'Unknown Title')}\n{comments_analysis}")

                            # Update the current video_data dictionary with the GPT-4 response
                            video_data['comments_analysis'] = comments_analysis
                            break  # Break the loop if request was successful

                        except openai.RateLimitError as e:
                            logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds.")
                            time.sleep(delay)
                            delay *= 2  # Double the delay for exponential backoff
                            retry_attempt += 1
                        except Exception as e:
                            logging.error(f"An error occurred during GPT-4 analysis for video {video_id}: {e}")
                            break  # Break the loop for non-rate limit errors

                    time.sleep(base_delay)  # Wait before processing the next video to control request rate

                else:
                    if 'comments_analysis' in video_data:
                        logging.info(f"Comments analysis already completed for video ID {video_id}. Skipping.")
                    else:
                        logging.warning(f"{video_data.get('comment_count', '0')} comments found for video: {video_data.get('title', 'Unknown Title')}")

        logging.info("All video comments have been analyzed.")
        if self.filename:
            # Extract filename and extension
            filename, file_extension = os.path.splitext(self.file_name)
            
            # Append "_c" to filename and add the extension back
            modified_file_name = filename + "_cman" + file_extension
            
            # Save the data to the modified file name
            self.save_video_data_to_csv(modified_file_name)
        else:
            self.save_video_data_to_csv("placeholder_file_name_cman_results.default.save")

    def print_video_data(self):
        for video_data in self.video_data.values():
            print(f"\n Start of second for loop in report()\n")

            print(f"\nTitle: {video_data['title']}")
            print(f"URL: {video_data['url']}")
            print(f"View Count: {video_data['view_count']}")
            print(f"Upload Date: {video_data['upload_date']}")

    def calculate_engagement_metrics(self, video_id):
        """
        Calculates engagement metrics for a given video.
        Args:
            video_id (str): The ID of the video to analyze.
        Returns:
            dict: A dictionary of engagement metrics.
        """
        if video_id not in self.video_data:
            self.logger.error(f"Video ID {video_id} not found in video_data_profile.")
            return None

        video_data = self.video_data[video_id][0]

        try:
            view_count = int(video_data.get('view_count', '0') or '0')  # Default to '0' if key not present or empty
            like_count = int(video_data.get('like_count', '0') or '0')
            comment_count = int(video_data.get('comment_count', '0') or '0')
        except (TypeError, ValueError) as e:
            self.logger.exception(f"Data type error in video data for video ID {video_id}: {e}")
            return None

        metrics = self.calculate_metrics(view_count, like_count, comment_count)
        self.logger.info(f"Engagement metrics calculated for video ID {video_id}.")
        return metrics

    def calculate_metrics(self, view_count, like_count, comment_count):
        """ Helper function to calculate engagement metrics. """
        return {
            'view_to_like_ratio': view_count / like_count if like_count else float('inf'),
            'view_to_comment_ratio': view_count / comment_count if comment_count else float('inf'),
            'like_to_comment_ratio': like_count / comment_count if comment_count else float('inf'),
            'comment_to_view_ratio': comment_count / view_count if view_count else 0,
            'likes_per_view_percentage': (like_count / view_count * 100) if view_count else 0,
        }
    def oai_thumbnail_analysis(self, url):
        """
        Performs an analysis of a thumbnail image using GPT-4 Vision model.

        This method sends a request to the GPT-4 Vision API with the thumbnail URL and a predefined prompt.
        The prompt requests a succinct analysis of the thumbnail, including text description, object identification,
        and human body language analysis.

        Args:
            url (str): The URL of the thumbnail image to be analyzed.

        Returns:
            The response from the GPT-4 Vision model, containing the analysis of the thumbnail.
        """
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide a succinct thumbnail analysis. List and describe any text present, including its style and design. Identify and name all visible objects. Analyze any human presence, noting facial expressions and body language. Focus on being brief yet comprehensive."},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                        },
                        },
                    ],
                    }
                ],
                max_tokens=300,
                )
        except openai.BadRequestError as e:
            print("A request error occurred: ", e)

        self.logger.info(f"Open Ai Vision Thumbnail Anlysis: \n {response.choices[0].message.content}")
        return response.choices[0].message.content
    
    def oai_thumbnails_to_csv(self, number_calls, filename):
        """
        Processes a bulk analysis of thumbnail images using the GPT-4 Vision model, limited by a specified number of calls.

        This function iterates over the video data, performing thumbnail analysis using 'oai_tn_analysis' method,
        up to the specified number of calls. It associates each video ID with its corresponding thumbnail analysis.
        Then saves to a .csv file using pandas library.

        Args:
            number_calls (int): The maximum number of thumbnail analyses to perform.

        Returns:
            dict: A dictionary where each key is a video ID and the value is the thumbnail analysis result.
        """
        self.logger.info("Starting oai_thumbnails_to_csv.")
        results = {}

        if number_calls > 0 and isinstance(number_calls, int):
            count = 0
            for video_id, data_list in self.video_data.items():
                if count >= number_calls: #Exit out of loop when count hits threshold
                    break

                for data in data_list:
                    if count >= number_calls:
                        break

                    if data.get('thumbnail_url') and "thumbnail_analysis" not in data:
                        # Append analysis to the list for each video_id
                        if video_id not in results:
                            results[video_id] = []
                        analysis = self.oai_thumbnail_analysis(data.get('thumbnail_url'))
                        results[video_id].append(analysis)
                        data["thumbnail_analysis"] = analysis #update video data profile with information
                        count += 1
                        self.logger.info(f"Count is now {count}")

        if count < number_calls:
            self.logger.info("Reached the end of video data before completing the specified number of calls.")

        self.logger.info(f"Writing {len(results.keys())} items to {filename}")

        # Open a file to write
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['Video ID', 'Thumbnail Analysis'])

                # Write the data
                for video_id, analyses in results.items():
                    for analysis in analyses:
                        writer.writerow([video_id, analysis])
        
            self.logger.info(f"File saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error while saving file: {e}")

        if os.path.exists(filename):
            self.logger.info(f"File succesfully saved to {filename}")
        else:
            self.logger.info(f"{filename} unsuccesfully saved.")


    def full_thumbnail_analysis(self, url):
        """
        Generates an analysis of a thumbnail image, including quality metrics and content labels.

        Args:
            url (str): The URL of the thumbnail image to be analyzed.

        Returns:
            dict: A dictionary containing the quality metrics and content labels of the thumbnail.
                The 'quality' key maps to a dictionary of quality metrics such as resolution,
                aspect ratio, and mode. The 'content' key maps to a list of content labels
                extracted from the image.
        """
        quality = self.analyze_thumbnail_quality(url)
        content_labels = self.analyze_thumbnail_content(url)

        return {
            'quality': quality,
            'content': content_labels
        }

    def analyze_thumbnail_content(self, url):
        """
        Analyzes the content of a thumbnail image by detecting labels using Google Vision API.
        Implements retry logic with exponential backoff in case of service unavailability.

        Args:
            url (str): The URL of the thumbnail image to be analyzed.

        Returns:
            list: A list of label descriptions detected in the thumbnail image.

        Raises:
            GoogleAPICallError: If an error occurs in the Google API call that is not related
            to service availability.
            Exception: For any other unexpected errors during label detection
        """
        max_retries = 5  # maximum number of retries
        backoff_factor = 2  # factor by which to multiply the delay with each retry
        initial_delay = 1  # initial delay in seconds

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt}: Starting label detection in analyze_thumbnail_content")
                client = vision.ImageAnnotatorClient()
                image = vision.Image()
                image.source.image_uri = url

                response = client.label_detection(image=image)
                labels = response.label_annotations

                self.logger.info("Label detection completed successfully")
                return [label.description for label in labels]

            except ServiceUnavailable as e:
                self.logger.warning(f"ServiceUnavailable error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    self.logger.error("Max retries reached. Aborting label detection.")
                    raise
                else:
                    delay = initial_delay * (backoff_factor ** (attempt - 1))
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

            except GoogleAPICallError as e:
                self.logger.error(f"Google API call error: {e}")
                raise  # re-raise the exception as it's not related to service availability

            except Exception as e:
                self.logger.error(f"Unexpected error during label detection: {e}")
                raise  # re-raise unexpected exceptions

    def analyze_thumbnail_quality(self, url):
        """
        Retrieves and analyzes the quality of a thumbnail image, including its resolution,
        aspect ratio, and mode.

        Args:
            url (str): The URL of the thumbnail image to be analyzed.

        Returns:
            dict: A dictionary containing the quality metrics of the thumbnail image such as
            'resolution' (tuple), 'aspect_ratio' (float), and 'mode' (str).
        """
        self.logger.info("Starting image retrieval in analyze_thumbnail_quality")
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        quality_metrics = {
            'resolution': img.size,
            'aspect_ratio': img.size[0] / img.size[1],
            'mode': img.mode
        }
        
        self.logger.info("Image retrieved and processed")
        return quality_metrics
    
    def fetch_related_videos(self, video_id, max_results):
        """
        Fetches videos related to the given video ID based on subject and title.
        Args:
            video_id (str): The ID of the video to find related videos for.
            max_results (int): The maximum number of related videos to return.
        Returns:
            list: A list of dictionaries containing details of related videos.
        """
        if video_id not in self.video_data or not self.video_data[video_id]:
            self.logger.error(f"No data found for video ID {video_id}")
            return []

        video_data = self.video_data[video_id][0]
        subject = video_data.get('subject', '')
        title = video_data.get('title', '')

        # Use subject and title as search query
        search_query = f"{subject} {title}"

        # Get related videos based on the search query
        related_video_ids = self.get_top_videos(query=search_query, max_results=max_results)

        # Filter out video IDs already in video_data_profile
        new_video_ids = [vid for vid in related_video_ids if vid not in self.video_data]
        self.load_data_from_ids(new_video_ids)

        # Fetch and store details for these related videos
        related_video_details = []

        for related_video_id in related_video_ids:
            if related_video_id in self.video_data:
                related_video_data = self.video_data.get(related_video_id, [{}])[0]
                if 'video_id' not in related_video_data:
                    self.logger.error(f"Missing 'video_id' in data for related video ID {related_video_id}")
                    continue  # Skip if 'video_id' is not in the data
                related_video_details.append(related_video_data)
            else:
                # Handle the case where related_video_id is not in self.video_data
                self.logger.warning(f"Related video ID {related_video_id} not found in video data profile")

        return related_video_details

    def filter_videos_by_criteria(self, filter_type, min_threshold=None, max_threshold=None):
        """
        Filters videos based on given criteria.

        Args:
            filter_type (str): Possible filters ("view_count," "comment_count," "like_count").
            min_threshold (int): Minimum threshold to filter by.
            max_threshold (int): Maximum threshold to filter by.

        Returns:
            list: Filtered video IDs.
        """
        if filter_type not in ["view_count", "like_count", "comment_count"]:
            self.logger.error(f"filter_type parameter for filter_videos_by_critera()  may only be 'veiw_count', 'like_count', or 'comment_count' - you asked for {filter_type}")
        
        filtered_videos = []

        for video_id, video_data_list in self.video_data.items():
            for video_data in video_data_list:
                metric_value = int(video_data.get(filter_type, 0))
                if (min_threshold is None or metric_value >= min_threshold) and (max_threshold is None or metric_value <= max_threshold):
                    filtered_videos.append(video_id)

    def post_comment(self, video_id, text):
        body = {
            'snippet': {
                'videoId': video_id,
                'topLevelComment': {
                    'snippet': {
                        'textOriginal': text
                    }
                }
            }
        }

        try:
            response = self.youtube.commentThreads().insert(
                part='snippet',
                body=body
            ).execute()
            return response

        except HttpError as e:
            print(f'An HTTP error {e.resp.status} occurred: {e.content}')
            return None
        
    def get_all_comments(self):
        """
        Retrieves comments for all videos stored in self.video_data using the YouTube API.
        Calls get_comments function for each video ID.
        """
        for video_id in self.video_data:
            # Check if comments are already fetched
            if 'comment_log' in self.video_data[video_id]:
                print(f"Comments already loaded for video ID {video_id}. Skipping.")
                continue

            # Call get_comments function for each video
            comments = self.get_comments(video_id)
            if comments:
                self.video_data[video_id].append({'comment_log': comments})
                print(f"Comments loaded for video ID {video_id}")

                # Extract video title and create a valid filename
                video_title = self.video_data[video_id][0]['title']
                safe_title = "".join(x for x in video_title if (x.isalnum() or x in "._- "))
                modified_file_name = f"{safe_title}_comments.txt"

                # Save comments to the file
                with open(modified_file_name, 'w', encoding='utf-8') as f:
                    for comment in comments:
                        f.write(f"{comment}\n")
                print(f"Comments saved to {modified_file_name}")
            else:
                print(f"No comments found or unable to load comments for video ID {video_id}")



    def get_comments(self, video_id, seperate_save=False, filename = ""):
        """get comments for a video_id, if seperate_save save to filename

        Args:
            video_id (_string_): video_id
            seperate_save (bool, optional): Defaults to False - if wanting to save comments - set to True
            filename (str, optional):Defaults to "".
        """
        
        backoff_time = 1
        max_backoff_time = 32
        retry_attempt = 0

        if video_id in self.video_data:
            for each_video_data in self.video_data[video_id]:  # Iterate through the list of dictionaries
                if 'comment_log' in each_video_data:
                    self.logger.info(f"Comments already loaded for video ID: {video_id}. Skipping.")
                    continue

                while retry_attempt < 5:
                    try:
                        request = self.youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=100,  # Adjust as needed
                            textFormat="plainText"
                        )
                        response = request.execute()

                        comments = []
                        for item in response.get('items', []):
                            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                            comments.append(comment)

                        # Update video data with comments
                        each_video_data['comment_log'] = comments
                        self.logger.info(f"Comments loaded for video ID: {video_id}")
                        return comments  # Exit the while loop after successful fetch

                    except errors.HttpError as e:
                        if e.resp.status in [403, 429]:
                            self.logger.warning("Rate limit exceeded. Backing off for %d seconds.", backoff_time)
                            time.sleep(backoff_time)
                            backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                            retry_attempt += 1
                            if retry_attempt > 8:
                                # Extract filename and extension
                                filename, file_extension = os.path.splitext(self.file_name)
                                
                                # Append "_c" to filename and add the extension back
                                modified_file_name = filename + "_comments_failure" + file_extension
                                
                                # Save the data to the modified file name
                                self.save_video_data_to_csv(modified_file_name)
                                print("Exceeded maximum retry attempts and saved to {modified_file_name}")
                                break  # Exit the while loop as maximum retries reached
                        else:
                            self.logger.error(f"An HTTP error occurred: {e}\n*Breaking errors.HttpError except loop*")
                            break  # Exit the while loop as it's not a rate limit error

                    except Exception as e:
                        self.logger.error(f"An unexpected error occurred: {e}")
                        self.logger.warning("Backing off for %d seconds.", backoff_time)
                        time.sleep(backoff_time)
                        backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                        retry_attempt += 1
                        if retry_attempt > 8:
                            # Extract filename and extension
                                filename, file_extension = os.path.splitext(self.file_name)
                                
                                # Append "_c" to filename and add the extension back
                                modified_file_name = filename + "_comments_failure" + file_extension
                                
                                # Save the data to the modified file name
                                self.save_video_data_to_csv(modified_file_name)
                                print("Exceeded maximum retry attempts and saved to {modified_file_name}")
                                break  # Exit the while loop as maximum retries reached
        
        # Save comments to a .txt file
        if seperate_save and filename:
            with open(filename, 'w', encoding='utf-8') as file:
                for comment in comments:
                    file.write(comment + '\n')

        else:
            self.logger.warning(f"Video ID {video_id} not found in video_data_profile.")

    def analyze_comments(self):

        for video_id, video_entries in self.video_data.items():
            for video_entry in video_entries:
                if 'comment_log' in video_entry:
                    comments_text = "\n".join(video_entry['comment_log'])

                    # Prompt to analyze the comments
                    prompt = (f"Based on the following comments:\n{comments_text}\n"
                            "Identify the most popular scenes and what people enjoyed about them. "
                            "Provide a concicse summary around popular topics")

                    
                    response = self.gpt_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                        {"role": "system", "content": "You are expert consultant on youtube content/comments and public opinion"},
                        {"role": "user", "content": prompt}
                    ],
                        max_tokens=2000
                    )

                    
                    analysis = response.choices[0].message.content.strip()
                    print(f"Analysis for video ID {video_id} Completed \n Results: {analysis}")

                    # Append the analysis to the list of dictionaries for this video
                    video_entry['comment_analysis'] = analysis
                    print(f'Added comment analysis succuesfully')

    def ask_gpt(self, question):
        response = self.gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful expert youtube consultant servant"},
                {"role": "user", "content": question}
            ],
            max_tokens=2000
        )

        answer = response.choices[0].message.content.strip()
        return answer
    
    def generate_image(self, prompt):
        response = self.gpt_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            quality="standard",
            size="1024x1024",  # Image size
            n=1,  # Number of images to generate
        )
        return response.data[0].url
    
    def generate_variation(self, image, number=1):
        response = self.gpt_client.images.create_variation(
        model="dall-e-2",
        image=open(image, "rb"),
        n=number,
        size="1024x1024"
        )

        return response.data[0].url

    def save_image(self, image_url, prompt):
        # Create the filename based on the first 8 letters of the prompt
        filename = f"{prompt[:8].replace(' ', '_')}.png"
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Image saved as {filename}")
        else:
            print("Failed to download image")

    def calculate_ratio(self, numerator, denominator):
        """
        Calculates the ratio of two values as a percentage.
        Args:
            numerator (int): The numerator in the ratio.
            denominator (int): The denominator in the ratio.
        Returns:
            float: The ratio as a percentage.
        """
        return (numerator / denominator * 100) if denominator else 0

    def plot_top_topics_by_views(self, autosave = False):
        # Aggregate data by topic
        topic_data = {}
        for video_id, data_list in self.video_data.items():
            for data in data_list:
                topic = data.get("subject", "Unknown")  # Assuming "subject" is a key in the video data
                view_count = data.get("view_count", 0)
                if topic in topic_data:
                    topic_data[topic]["views"] = topic_data[topic].get("views", 0) + view_count
                else:
                    topic_data[topic] = {"views": view_count}

        # Extract top 10 topics based on views
        top_10_topics = sorted(topic_data.items(), key=lambda x: x[1]["views"], reverse=True)[:10]

        # Convert data to DataFrame for plotting
        df = pd.DataFrame(top_10_topics, columns=["Topic", "Data"])
        df["Total Views"] = df["Data"].apply(lambda x: x["views"])

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plotting the data
        fig = plt.figure(figsize=(12, 6))

        # Plot bars for views
        view_plot = sns.barplot(x='Total Views', y='Topic', data=df, color='skyblue', alpha=0.7, label='Total Views')

        # Adding number overlays for views
        for i, p in enumerate(view_plot.patches):
            view_count = df.iloc[i]["Total Views"]
            view_plot.annotate(f'Views: {view_count}', 
                            (p.get_width(), p.get_y() + p.get_height() / 2),
                            ha='center', va='center', fontsize=10, color='black', xytext=(5, 0),
                            textcoords='offset points')


        # Adding labels and title
        plt.xlabel('Counts')
        plt.ylabel('Topic')
        plt.title('Top 10 Topics by Views')
        plt.legend()

        # Show the plot
        plt.tight_layout()
        if autosave:
            return fig, f"Top Topics by Views"
        else:
            plt.show()

    def plot_top_topics_by_likes(self, autosave = False):

        # Aggregate data by topic
        topic_data = {}
        for video_id, data_list in self.video_data.items():
            for data in data_list:
                topic = data.get("subject", "Unknown")
                like_count = data.get("like_count", 0)
                if not pd.isnull(like_count):  # Check for NaN values
                    if topic in topic_data:
                        topic_data[topic]["likes"] = topic_data[topic].get("likes", 0) + like_count
                    else:
                        topic_data[topic] = {"likes": like_count}

        # Extract top 10 topics based on likes
        top_10_topics = sorted(topic_data.items(), key=lambda x: x[1].get("likes", 0), reverse=True)[:10]

        # Convert data to DataFrame for plotting
        df = pd.DataFrame(top_10_topics, columns=["Topic", "Data"])
        df["Total Likes"] = df["Data"].apply(lambda x: x.get("likes", 0))

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plotting the data
        fig = plt.figure(figsize=(12, 6))
        like_plot = sns.barplot(x='Total Likes', y='Topic', data=df, color='skyblue', alpha=0.7, label='Total Likes')

        # Adding number overlays for likes
        for i, p in enumerate(like_plot.patches):
            like_count = df.iloc[i]["Total Likes"]
            like_plot.annotate(f'Likes: {like_count}',
                            (p.get_width(), p.get_y() + p.get_height() / 2),
                            ha='center', va='center', fontsize=10, color='black', xytext=(5, 0),
                            textcoords='offset points')

        # Adding labels and title
        plt.xlabel('Counts')
        plt.ylabel('Topic')
        plt.title('Top 10 Topics by Likes')
        plt.legend()

        # Show the plot
        plt.tight_layout()
        if autosave:
            return fig, "Top Topics by Likes"
        else:
            plt.show()

    def plot_top_topics_by_comments(self, autosave = False):
        topic_data = {}
        for video_id, data_list in self.video_data.items():
            for data in data_list:
                topic = data.get("subject", "Unknown")  # Assuming "subject" is a key in the video data
                comment_count = data.get("comment_count", 0)
                if topic in topic_data:
                    if not pd.isnull(comment_count):  # Check for NaN values
                        topic_data[topic]["comments"] += comment_count
                else:
                    if not pd.isnull(comment_count):  # Check for NaN values
                        topic_data[topic] = {"comments": comment_count}

        print("Topic Data:", topic_data)
        # Extract top 10 topics based on comments
        top_10_topics = sorted(topic_data.items(), key=lambda x: x[1]["comments"], reverse=True)[:10]

        # Convert data to DataFrame for plotting
        df = pd.DataFrame(top_10_topics, columns=["Topic", "Data"])
        df["Total Comments"] = df["Data"].apply(lambda x: x["comments"])
        print("DataFrame:", df)
        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plotting the data
        fig = plt.figure(figsize=(12, 6))

        # Plot bars for comments
        comment_plot = sns.barplot(x='Total Comments', y='Topic', data=df, color='orange', alpha=0.7, label='Total Comments')

        # Adding number overlays for comments
        for i, p in enumerate(comment_plot.patches):
            comment_count = df.iloc[i]["Total Comments"]
            comment_plot.annotate(f'Comments: {comment_count}', 
                            (p.get_width(), p.get_y() + p.get_height() / 2),
                            ha='center', va='center', fontsize=10, color='black', xytext=(5, 0),
                            textcoords='offset points')

        # Adding labels and title
        plt.xlabel('Total Comments')
        plt.ylabel('Topic')
        plt.title('Top 10 Topics by Comments')
        plt.legend()

        # Show the plot
        plt.tight_layout()
        if autosave:
            return fig, 'Top 10 Topics by Comments'
        else:
            plt.show()

    def plot_3d_scatter_view_vs_like(self, df, sample_size=100):
        """
        Plot a 3D scatter plot showing View Count vs Like Count across Subjects.

        Args:
        - df (pd.DataFrame): DataFrame containing 'view_count', 'like_count', and 'subject' columns.
        - sample_size (int): Number of data points to sample for clarity.

        Returns:
        - fig (matplotlib.figure.Figure): Figure object containing the plot.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Selecting subset for demonstration
        subset = df.sample(sample_size)  # Choose a subset of data for clarity

        # Plotting a 3D surface
        x = subset['view_count']
        y = subset['like_count']
        z = subset['subject'].astype('category').cat.codes  # Mapping subjects to numeric categories
        ax.scatter(x, y, z, c=z, cmap='viridis')

        ax.set_xlabel('View Count')
        ax.set_ylabel('Like Count')
        ax.set_zlabel('Subject')

        plt.title('3D Scatter Plot: View Count vs Like Count across Subjects')
        plt.show()
        return fig

    def plot_views_title_length_line_chart(self, subject):
        """
        Create a line chart to compare view counts and title character lengths.

        Parameters:
        subject (str): Subject name.

        Returns:
        fig object of chart, Title using filename (only works if load_video_data called or self.file_name is set)
        """
        video_ids = []
        views = []
        title_lengths = []

        # Iterate through video IDs for the given subject
        for vid, data_list in self.video_data.items():
            for data in data_list:
                if 'subject' in data and data['subject'] == subject:
                    view_count = data.get('view_count')
                    title = data.get('title')

                    # Extract view_count and title data
                    if view_count is not None and title is not None:
                        try:
                            views.append(int(float(view_count)))
                        except ValueError as e:
                            self.logger.error(f"Error converting view_count to int for video {vid}: {view_count}")

                        title_lengths.append(len(title))

        # Sort data based on views for line chart
        views, title_lengths = zip(*sorted(zip(views, title_lengths)))

        # Plotting as line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(views, title_lengths, marker='o', linestyle='-')

        # Setting labels and title
        ax.set_xlabel('Views')
        ax.set_ylabel('Title Length (characters)')
        ax.set_title(f'Comparison of View Counts and Title Length for {subject}')

        # Layout adjustment
        plt.tight_layout()
        plt.show()
        return fig, f'Comparison of View Counts and Title Length - Topic: {subject}'

    def plot_views_to_title_chars_scatter(self, subject, autosave = False):
        """
        Create a scatter plot to compare view counts and title character lengths.
        If autosave = True, return fig object and title, else show plot.

        Parameters:
        subject (str): Subject name.

        Returns:
        fig object of chart, Title using filename (only works if load_video_data called or self.file_name is set)
        """
        video_ids = []
        views = []
        title_lengths = []

        # Iterate through video IDs for the given subject
        for vid, data_list in self.video_data.items():
            for data in data_list:
                if 'subject' in data and data['subject'] == subject:
                    view_count = data.get('view_count')
                    title = data.get('title')

                    # Extract view_count and title data
                    if view_count is not None and title is not None:
                        try:
                            views.append(int(float(view_count)))
                        except ValueError as e:
                            self.logger.error(f"Error converting view_count to int for video {vid}: {view_count}")

                        title_lengths.append(len(title))

        # Create a DataFrame for Seaborn
        data = {
            'Views': views,
            'Title Length': title_lengths
        }
        df = pd.DataFrame(data)

        # Create scatter plot using Seaborn
        sns.set(style="whitegrid")  # Set plot style
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Views', y='Title Length', ax=ax)
        sns.regplot(data=df, x='Views', y='Title Length', ax=ax, scatter=False)  # Add a regression line

        # Setting labels and title
        ax.set_xlabel('Views')
        ax.set_ylabel('Title Length (characters)')
        ax.set_title(f'Comparison of View Counts and Title Length for {subject}')

        # Layout adjustment
        plt.tight_layout()
        if autosave:
            return fig, f'Comparison of View Counts and Title Length - Topic: {subject}'
        else:
            plt.show()
        
    def plot_view_count_distribution(self, dataframe, auto_save=False, save_dir=None):
        """
        Plots the distribution of view counts.

        Args:
            dataframe (pd.DataFrame): DataFrame containing 'view_count' column.
            auto_save (bool): If True, saves the graph; if False, displays the graph.
            save_dir (str): Directory to save the graph if auto_save is True.

        Returns:
            tuple: If auto_save is False, returns title and fig object; else, returns None.
        """
        # Function to format tick labels with commas
        def format_with_commas(x, pos):
            return "{:,}".format(int(x))

        # Plotting the view count distribution
        plt.figure(figsize=(8, 6))
        plt.hist(dataframe['view_count'], bins=20, color='skyblue', edgecolor='black')
        plt.yscale('log')
        plt.xlabel('View Count')
        plt.ylabel('Frequency')
        plt.title('View Count Distribution')
        
        formatter = FuncFormatter(format_with_commas)
        plt.gca().xaxis.set_major_formatter(formatter)

        plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees, align them to the right

        # Adjust layout for better readability
        plt.tight_layout(pad=4)  # Add padding at the bottom

        if auto_save:
            if save_dir:
                file_path = f"{save_dir}/view_count_distribution.png"
                plt.savefig(file_path)
                return None
            else:
                raise ValueError("Please provide a valid save directory.")
        else:
            plt.show()
            return 
    
    def plot_top_tags_pie_chart(self, subject, autosave=False):
        tag_counter = Counter()
        sample_size = 0  # Initialize sample size

        # Aggregate tags for the specified subject
        for video_data in self.video_data.values():
            for data in video_data:
                if data['subject'] == subject:
                    sample_size += 1
                    tags = data['tags'].split(",")  # Assuming tags are comma-separated strings
                    for tag in tags:
                        clean_tag = tag.strip()
                        if clean_tag and clean_tag != "[]":
                            tag_counter.update([clean_tag])

        # Get the top 5 tags
        top_tags = tag_counter.most_common(8)
        self.logger.info(f"Top tags found are: {top_tags}")
        # Calculate the percentage of videos that use each of the top 5 tags
        total_videos = sum(tag_counter.values())
        sizes = [(count / total_videos) * 100 for tag, count in top_tags]
        labels = [tag for tag, count in top_tags]

        # Plotting the pie chart with percentages
        fig = plt.figure(figsize=(10, 8))
        pie_wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        
        # Customize autopct labels (percentages)
        for autotext in autotexts:
            autotext.set_color('white')  # Change color as needed
            autotext.set_fontsize(10)  # Adjust font size as needed
        title = f"Top 8 Tags - Subject: {subject} | by % videos using 'tag' out of {sample_size} videos"
        plt.title(title, pad = 20)
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        plt.tight_layout(pad=4)
        if autosave:
            return fig, title
        else:
            plt.show()

        
    def plot_radar_chart(self, *args, autosave=False, **kwargs):
        """
        Generates a radar chart representing the likes, comments, length of title in
        characters and words, as well as number of tags and number of tag words used in the title
        compared as a percentage to the range of the metrics in the dataset stored in self.video_data

        Output:
        Plots a radar chart measuring above metrics.
        """
        if args and len(args[0]) == 11:
            video_id = args[0]  # Assumes the first argument is the video_id
            data_list = self.video_data[video_id]
            if not autosave:
                # Initial values
                likes, comments, tags, headline_word_count, headline_char_count, headline_tag_count = 0, 0, 0, 0, 0, 0
                title = "init_title"

                # Initialize lists to accumulate metric values for the current dataset
                current_dataset_values = {
                    "likes": [],
                    "comments": [],
                    "tags": [],
                    "headline_word_count": [],
                    "headline_char_count": [],
                    "headline_tag_count": [],
                }
            
                # Iterate through self.video_data, accumulating values for the current dataset
                for video_id, data_list in self.video_data.items():
                    for data in data_list:
                        # Extract metrics from each data point
                        likes_str = data.get('like_count', '0')
                        comments_str = data.get('comment_count', '0')
                        tags_str = data.get('tags', [])
                        title = data.get('title', 'Default Title')

                        #print(f"Processing video ID: {video_id}")
                        #print(f"Likes: {likes_str}, Comments: {comments_str}, Tags: {tags_str}, Title: {title}")

                        # Accumulate headline metrics
                        current_dataset_values["headline_word_count"].append(len(title.split()))
                        current_dataset_values["headline_char_count"].append(len(title))
                        words_in_title = [word for word in title.split()]
                        current_dataset_values["headline_tag_count"].append(sum(1 for word in words_in_title if word in tags_str))

                        # Handling NaN for 'like_count' and 'comment_count'
                        if not math.isnan(float(likes_str)):
                            current_dataset_values["likes"].append(int(float(likes_str)))
                        if not math.isnan(float(comments_str)):
                            current_dataset_values["comments"].append(int(float(comments_str)))

                        #print("\nCurrent dataset values:")
                        #for metric, values in current_dataset_values.items():
                            #print(f"{metric}: {values}")

                        # Check for common words between title and tags (inside the loop)
                        title_words = set(title.split())
                        tag_words = set(data.get('tags', []))
                        common_words = title_words & tag_words
                        # if common_words:
                            #print(f"For video ID {video_id}, common words found in title and tags: {common_words}")
                        #else:
                            #print(f"For video ID {video_id}, no common words found between title and tags")

                # Calculate ranges based on the current dataset's values
                try:
                    ranges = {
                            metric: max(values) - min(values) if values else 0  # Handle empty lists
                            for metric, values in current_dataset_values.items()
                        }
                except ValueError:
                    print("Value Error: Unable to calculate ranges due to empty metric lists. Check data integrity.")
                    return# Or handle the error differently

                # Calculate values for the specific video_id
                values = [
                    min(current_dataset_values["likes"][0] / ranges["likes"], 1),  # Access values for specific video_id
                    min(current_dataset_values["comments"][0] / ranges["comments"], 1),
                    min(current_dataset_values["headline_word_count"][0] / ranges["headline_word_count"], 1),
                    min(current_dataset_values["headline_tag_count"][0] / ranges["headline_tag_count"], 1),
                    min(len(tags_str) / ranges["tags"] if ranges["tags"] else 1, 1),  # Percentage of tag words relative to max tags
                    min(len(title_words) / ranges["headline_char_count"], 1),  # Percentage of title char length relative to max headline chars
                ]

                print("\nValues before percentage calculation:")
                print(values)
                # Convert values to percentages and scale for plotting
                values = [v * 100 for v in values]

                print("\nValues after percentage calculation:")
                print(values)

                # Define categories for the radar chart
                categories = ["Likes", "Comments", "Tags", "Headline Words", "Headline Chars", "Tag Words Used in Headline"]
                N = len(categories)

                # Compute angles for each category
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop

                # Replicate values for loop closure
                values += values[:1]


                # Create the radar chart plot
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                ax.plot(angles, values, linewidth=2, linestyle='solid', label='Video Metrics')
                ax.fill(angles, values, alpha=0.1)
                if common_words:
                    common_words_text = "\nCommon words in title and tags:\n" + "\n".join(common_words)
                    ax.annotate(common_words_text, xy=(0.5, 0), ha='center', va='bottom', fontsize=10)

                # Add labels for each category
                plt.xticks(angles[:-1], categories)

                # Adjust radial ticks to show percentages
                plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=12)
                plt.ylim(0, 100)

                # Add title and adjust label size
                plt.xticks(angles[:-1], categories, size=12)  # Increase font size

                # Adjust title and legend
                plt.title(title, size=15, color='blue', fontweight='bold', y=1.1)  # Reduce font size and adjust position
                legend_text = '100% =(' + ', '.join(f"{metric}: {range_value}" for metric, range_value in ranges.items()) + ')'
                ax.legend([legend_text], loc='upper left', bbox_to_anchor=(-0.3, 1.11),  # Increase horizontal space using the first value
                fontsize=8,  # Reduce font size
                )
                plt.tight_layout()
                plt.show()

        elif autosave:  # Return data for all videos if autosave is True
            data_for_all_videos = []
            unique_videos = set()

            current_dataset_values = self.accumulate_metrics(self.video_data)
            print(f"Current Data Set from accumlate_metrics is {current_dataset_values}")
            ranges = self.calculate_ranges(current_dataset_values)
            common_words = []

            for vid, data_list in self.video_data.items():

                if ranges is None:
                    continue

                title = data_list[0].get('title', 'Default Title')  # Obtain title of current video

                video_values = []

                for vid, video_data in current_dataset_values.items():
                    all_metrics = video_data["all_metrics"]  # Accessing the 'all_metrics' dictionary
                    for metric, value_list in all_metrics.items():
                        video_value = min(value_list[0] / ranges[metric], 1) * 100 if ranges[metric] != 0 else 0
                        video_values.append(video_value)
                        print(f"Adding video_value to video_values. video_value = {video_value}")

                # Define categories for the radar chart
                categories = ["Likes", "Comments", "Tags", "Headline Words", "Headline Chars", "Tag Words Used in Headline"]
                N = len(categories)

                # Compute angles for each category
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop

                # Replicate values for loop closure
                video_values += video_values[:1]

                print("Video Values:", video_values)
                print("Angles:", angles)

                if len(angles) != len(video_values):
                    print(f"Skipping video ID {vid} due to mismatch in data dimensions. Angles length: {len(angles)}, Values length: {len(video_values)}")
                    continue  # Skip this iteration and continue with the next


                # Create the radar chart plot
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                ax.plot(angles, video_values, linewidth=2, linestyle='solid', label='Video Metrics')
                ax.fill(angles, video_values, alpha=0.1)

                # Add annotation for common words if present
                if current_dataset_values["headline_tag_count"][0] > 0:
                    common_words_text = "\nCommon words in title and tags:\n" + "\n".join(current_dataset_values[vid]['common_words'])
                    ax.annotate(common_words_text, xy=(0.5, 0), ha='center', va='bottom', fontsize=10)

                # Add labels for each category
                plt.xticks(angles[:-1], categories)

                # Adjust radial ticks to show percentages
                plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=12)
                plt.ylim(0, 100)

                # Add title and adjust label size
                plt.xticks(angles[:-1], categories, size=12)  # Increase font size
                plt.title(title, size=15, color='blue', fontweight='bold', y=1.1)  # Reduce font size and adjust position

                # Adjust title and legend
                legend_text = '100% =(' + ', '.join(f"{metric}: {range_value}" for metric, range_value in ranges.items()) + ')'
                ax.legend([legend_text], loc='upper left', bbox_to_anchor=(-0.3, 1.11), fontsize=8)
                plt.tight_layout()

                if title not in unique_videos:
                    # Append data to the list only if it's not a duplicate
                    data_for_all_videos.append((fig, title))
                    unique_videos.add(title)  # Add title to processed set

                plt.close(fig)

            return data_for_all_videos
        else:
            print("Invalid arguments provided for plot_radar_charts.")
        
    def accumulate_metrics(self, video_data):
        """
        Accumulates metrics from video data.

        Args:
        - video_data (dict): Dictionary containing video metrics

        Returns:
        - accumulated_metrics (dict): Dictionary with accumulated metrics 
        - common_words (list): List containing common words found
        """
        current_dataset_values = {
            "likes": [],
            "comments": [],
            "headline_word_count": [],
            "headline_char_count": [],
            "headline_tag_count": [],
            "tags": [], # Assuming this refers to tag count in each video's data
            # ... Add other metrics similarly
        }
        accumulated_metrics = {}
        current_video_values = {}

        for vid, data_list in video_data.items():     
            # Store common words separately per video ID
            common_words_per_video = []
            current_video_values = {}
            for data in data_list:
                current_video_values["likes"] = ''
                current_video_values["comments"] = ''
                current_video_values["title"] = ''
                current_video_values["headline_word_count"] = []
                current_video_values["headline_char_count"] = []
                current_video_values["headline_tag_count"] = []
                #reset values back to empty

                likes_str = data.get('like_count', '0')
                comments_str = data.get('comment_count', '0')
                tags_str = data.get('tags', [])
                title = data.get('title', 'Title_Not_Found')
                
                # Handling NaN for 'like_count' and 'comment_count'
                likes_val = int(float(likes_str)) if self.is_numeric(likes_str) else 0
                comments_val = int(float(comments_str)) if self.is_numeric(comments_str) else 0

                # Accumulate headline metrics for each video
                title_words = set(title.split())
                tag_words = set(tags_str)
                common_words = list(title_words & tag_words)
                common_words_per_video.extend(common_words)

                current_dataset_values["headline_word_count"].append(len(title_words))
                current_dataset_values["headline_char_count"].append(len(title))
                current_dataset_values["headline_tag_count"].append(sum(1 for word in title_words if word in tags_str))

                current_dataset_values["likes"].append(likes_val)
                current_dataset_values["comments"].append(comments_val)
                current_dataset_values["tags"].append(len(tags_str))

                current_video_values["likes"] = likes_val
                current_video_values["comments"] = comments_val
                current_video_values["title"] = title
                current_video_values["headline_word_count"] = current_dataset_values["headline_word_count"]
                current_video_values["headline_char_count"] = current_dataset_values["headline_char_count"]
                current_video_values["headline_tag_count"] = current_dataset_values["headline_tag_count"]

            # Record common words per video ID
            accumulated_metrics[vid] = {
                "all_metrics": current_dataset_values,
                "my_metrics": current_video_values,
                "common_words": common_words_per_video
            }

        #self.pp.pprint(f"accumulate_metrics() function is now returnning accumlated_metrics: {accumulated_metrics}")
        return accumulated_metrics

    def is_numeric(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False
        
    def calculate_ranges(self, current_dataset_values):
        try:
            ranges = {}
            for metric, values in current_dataset_values.items():
                # Print the current metric and its values
                print(f"Processing metric: {metric}, values: {values}")

                # Check if the values are numeric and calculate the range
                if all(isinstance(v, (int, float)) for v in values):
                    ranges[metric] = self.calculate_metric_range(values)
                else:
                    ranges[metric] = 0  # Or handle non-numeric metrics differently

            return ranges
        except ValueError as e:
            print(f"Error encountered: {e}")
            return {}

    def calculate_metric_range(self, values):
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if not numeric_values:
            return 0
        return max(numeric_values) - min(numeric_values)
        
    def plot_totalchars_barchart(self, subject, autosave = False):
        """
        Plot video views as a bar chart with a color gradient representing the title length.
        Clips the list of video IDs to the first 20 if it's longer.

        Parameters:
        subject (str): Subject name.

        Returns:
        fig object of chart, Title using filename (only works if load_video_data called or self.file_name is set)
        """
        video_ids = []
        views = []
        title_lengths = []

        # Iterate through video IDs for the given subject
        for vid, data_list in self.video_data.items():
            for data in data_list:
                if 'subject' in data and data['subject'] == subject:
                    video_ids.append(vid)
                    view_count = data.get('view_count')
                    title = data.get('title')

                    # Extract view_count and title data
                    if view_count is not None and title is not None:
                        try:
                            views.append(int(float(view_count)))
                        except ValueError as e:
                            self.logger.error(f"Error converting view_count to int for video {vid}: {view_count}")

                        title_lengths.append(len(title))

        # Clip to the first 20 video IDs
        if len(video_ids) > 20:
            excluded_videos = len(video_ids) - 20
            print(f"List too long. {excluded_videos} videos were excluded from the plot.")
            video_ids = video_ids[:20]
            views = views[:20]
            title_lengths = title_lengths[:20]

        # Normalize title lengths for color mapping
        norm = mcolors.Normalize(vmin=min(title_lengths), vmax=max(title_lengths))
        cmap = plt.cm.viridis
        colors = cmap(norm(title_lengths))

        # Plotting as bar chart with logarithmic scale on y-axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(video_ids)), views, color=colors)
        ax.set_yscale('log')  # Set y-axis to logarithmic scale

        # Adding color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Title Length (characters)')

        # Setting labels and title
        ax.set_xticks(range(len(video_ids)))
        ax.set_xticklabels(video_ids, rotation=90)
        ax.set_ylabel('Views (log scale)')
        ax.set_xlabel('Video IDs')
        ax.set_title(f'Video Views vs. Title Length for {subject}')

        # Layout adjustment
        plt.tight_layout()
        if autosave:
            return fig, f'Video Views vs. Title Length - Topic: {subject}'
        else:
            plt.show()

    def plot_enage_ratios_barchart(self, view_count_min, view_count_max, top_n_videos=50):
        """
        Creates a bar chart for top N videos within a specified view count range, comparing engagement ratios.

        Args:
        view_count_min (int): The minimum view count for filtering videos.
        view_count_max (int): The maximum view count for filtering videos.
        top_n_videos (int): The number of top videos to display based on total engagement.

        Output:
        A bar chart where each set of bars represents the engagement ratios for a video.
        """

        # Filter videos based on view count and sort by total engagement (like + comment counts)
        filtered_videos = {
            vid: data[0] for vid, data in self.video_data.items()
            if data and view_count_min <= self.safe_int_conversion(data[0]['view_count']) <= view_count_max
        }
        sorted_videos = sorted(filtered_videos.items(), key=lambda x: self.safe_int_conversion(x[1]['like_count']) + self.safe_int_conversion(x[1]['comment_count']), reverse=True)[:top_n_videos]

        indices = np.arange(len(sorted_videos))
        like_view_ratios = []
        comment_view_ratios = []
        comment_like_ratios = []

        for video_id, data in sorted_videos:
            like_view_ratio = self.calculate_ratio(self.safe_int_conversion(data['like_count']), self.safe_int_conversion(data['view_count']))
            comment_view_ratio = self.calculate_ratio(self.safe_int_conversion(data['comment_count']), self.safe_int_conversion(data['view_count']))
            comment_like_ratio = self.calculate_ratio(self.safe_int_conversion(data['comment_count']), self.safe_int_conversion(data['like_count']))

            like_view_ratios.append(like_view_ratio)
            comment_view_ratios.append(comment_view_ratio)
            comment_like_ratios.append(comment_like_ratio)

        current_date = datetime.datetime.now()
        video_ages = [current_date - datetime.datetime.fromisoformat(data['upload_date'].replace('Z', '')) for _, data in sorted_videos]

        # Combine ratios and normalize ages for color gradient
        total_ratios = [like_view_ratios[i] + comment_view_ratios[i] for i in range(len(sorted_videos))]
        normalized_ages = [age.days / max(video_ages).days for age in video_ages]  # Normalize to 0-1

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(max(10, len(sorted_videos) // 2), 6))
        bars = ax.bar(indices, total_ratios, color=plt.cm.coolwarm(normalized_ages))

        # Customize the plot
        ax.set_xlabel('Video ID')
        ax.set_ylabel('Total Engagement Ratio (%)')
        ax.set_title(f"Total Engagement (Like/View + Comment/View) for Top {top_n_videos} Videos in View Count Range {view_count_min} to {view_count_max}")
        ax.set_xticks(indices)
        ax.set_xticklabels([video_id for video_id, _ in sorted_videos], rotation=45, ha='right')

        # Color gradient legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, max(normalized_ages)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Video Age (Older -> Newer)')

        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        plt.show()

        return fig, f'(Like/View + Comment/View) in View Count Range {view_count_min} to {view_count_max}'

    def plot_entity_word_cloud(self, subject, autosave = False):
        """
        Creates a word cloud of the most common named entities found in all text
        related to videos of a given subject.

        Args:
            subject (str): string that must be an existing subject key in the self.video_data dictionary
        """
        all_text = ''
        # Concatenate all text data for the specified subject
        for video_data in self.video_data.values():
            for data in video_data:
                if data['subject'] == subject:
                    all_text += ' '.join([str(data.get('title', '')), str(data.get('description', ''))])
                    # Check if transcript is available and convert to string before adding
                    if 'transcript' in data and data['transcript'] is not None:
                        transcript_text = str(data['transcript'])
                        all_text += ' ' + transcript_text

        # Perform NER to extract entities
        doc = self.nlp(all_text)
        entities = ' '.join([ent.text for ent in doc.ents])

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='skyblue',  min_font_size=13, contour_color='black', contour_width=2).generate(entities)

        # Plotting
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Word Cloud of Named Entities for {subject}')
        if autosave:
            return fig, f'Word Cloud of Named Entities for {subject}'
        else:
            plt.show()

        
    def plot_common_words_in_titles(self, subject, autosave = False):
        """
        Creates a bar chart comparing the most common words found in the "title" of a video (x-axis)
        compared to the number of videos that use that word (y-axis). Additionally, the color of the bars 
        represents the average length of the titles containing each word.

        Args:
            subject (str): string that must be an existing subject key in the self.video_data dictionary
        """
        word_counter = Counter()
        title_length_for_word = defaultdict(list)

        # Filter and tokenize titles based on subject
        for video_data in self.video_data.values():
            for data in video_data:
                if 'subject' in data and data['subject'] == subject:
                    title = data['title']
                    title_length = len(title.split())
                    words = [word for word in re.findall(r'\w+', title.lower()) if len(word) >= 4]
                    word_counter.update(words)
                    for word in words:
                        title_length_for_word[word].append(title_length)

        # Get the top 10 words
        top_words = word_counter.most_common(10)

        if top_words:
            words, counts = zip(*top_words)
            # Calculate average title length for each word
            avg_title_length = [np.mean(title_length_for_word[word]) for word in words]

            # Normalize for color mapping
            norm = plt.Normalize(min(avg_title_length), max(avg_title_length))
            colors = plt.cm.viridis(norm(avg_title_length))

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(words, counts, color=colors)
            ax.set_xlabel('Words From Headlines')
            ax.set_ylabel('Number of Videos')
            ax.set_title(f'Common Words in Titles + Avg Title Length| Subject: {subject} | For {len(self.video_data.keys())} videos')
            ax.set_xticks(words)
            ax.set_xticklabels(words, rotation=45)

            plt.subplots_adjust(bottom=0.2)
            # Adding a colorbar to represent the average title length
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)  # Explicitly specify the ax argument
            cbar.ax.set_ylabel('Average Title Length (Words) For Videos Using Word')

            plt.tight_layout()
            if autosave:
                return fig, f'Top 10 Common Words in Titles in {subject} and Average Word Length'
            else:
                plt.show()
        else:
            self.logger.error("No words found after filtering input.")
        
    
    def plot_word_cloud_titles(self, subject, autosave = False):
        """
        Generates and displays a word cloud based on the most common words found in the titles of YouTube videos for a given subject.

        Args:
            subject (str): The subject based on which videos are filtered. Only videos with this subject 
            are considered for generating the word cloud.
        """
        title_text = ''
        video_count = 0
        for video_id, video_list in self.video_data.items():
            for video_data in video_list:
                if video_data['subject'] == subject:
                    video_count += 1
                    title_text += ' ' + video_data['title']

        wordcloud = WordCloud(
            width=1200,  # Consider if this size fits your display needs
            height=800,  # Adjust if a different aspect ratio is desired
            background_color='white',
            max_words=50,  # Increased for more word density
            max_font_size=100,  # Adjusted for larger most frequent words
            min_font_size=10,  # Slightly increased minimum font size
            scale=4,  # Increased for better resolution
            relative_scaling=0.5  # Adjusted for frequency impact on word size
        ).generate(title_text)

        
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = f"Word Cloud: Top 'Title' Words Found From {video_count} '{subject}' Videos"
        plt.title(title, fontsize=14, pad = 25)
        if autosave:
            return fig, title
        else:
            plt.show()

    def plot_word_cloud_tags(self, subject, autosave = False):
        """
        Generates and displays a word cloud based on the most common words found in the tags of YouTube videos for a given subject.

        Args:
            subject (str): The subject based on which videos are filtered. Only videos with this subject 
            are considered for generating the word cloud.
        """
        tag_text = ''
        video_count = 0

        for video_id, video_list in self.video_data.items():
            for video_data in video_list:
                if video_data['subject'] == subject:
                    video_count += 1
                    tags = video_data['tags'].split(',')  # Assuming tags are comma-separated
                    tag_text += ' '.join(tags)

        wordcloud = WordCloud(
            width=1200,  # Consider if this size fits your display needs
            height=800,  # Adjust if a different aspect ratio is desired
            background_color='white',
            max_words=50,  # Increased for more word density
            max_font_size=100,  # Adjusted for larger most frequent words
            min_font_size=10,  # Slightly increased minimum font size
            scale=4,  # Increased for better resolution
            relative_scaling=0.5  # Adjusted for frequency impact on word size
        ).generate(tag_text)

        
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = f"Word Cloud: Top 'Tags' found from {video_count} Videos on '{subject}' "
        plt.title(title, fontsize=14, pad = 25)
        if autosave:
            return fig, title
        else:
            plt.show()

    def plot_word_cloud_all(self, subject, autosave = False):
        """
        Generates and displays a word cloud based on the most common words found in the titles, 
        descriptions, tags, and transcripts (if available) of YouTube videos for a given subject.

        The function aggregates text from the 'title', 'description', 'tags', and 'transcript' (if available) 
        fields of each video in 'self.video_data'. It then generates a word cloud where the size of 
        each word is proportional to its frequency across all aggregated text. This visualization helps in 
        understanding the most prominent words or themes related to the specified subject in the video dataset.

        Args:
            subject (str): The subject based on which videos are filtered. Only videos with this subject 
                are considered for generating the word cloud.

        Displays:
            A matplotlib figure showing the word cloud. Words that are more frequent in the video data 
            appear larger in the word cloud.
        """
        all_text = ''
        video_count = 0
        for video_id, video_list in self.video_data.items():
            for video_data in video_list:
                if video_data['subject'] == subject:
                    video_count += 1
                    try:
                        title = str(video_data['title']) if 'title' in video_data else ''
                        description = str(video_data['description']) if 'description' in video_data else ''
                        tags = ' '.join(map(str, video_data['tags'])) if 'tags' in video_data and isinstance(video_data['tags'], list) else ''
                        combined_text = title + ' ' + description + ' ' + tags
                        if 'transcript' in video_data:
                            combined_text += ' ' + str(video_data['transcript'])
                        if 'comments' in video_data:
                            combined_text += ' ' + str(video_data['comments'])


                    except TypeError as e:
                            combined_text_type = type(combined_text)
                            video_title_type = type(video_data['title'])
                            video_description_type = type(video_data['description'])
                            video_tags_type = type(video_data['tags'])

                            self.logger.info(f"combined_text is not a String. It is {combined_text_type}")
                            if combined_text_type == str:
                                self.logger.info(f"With length of {len(combined_text)}")

                            self.logger.info(f"Type for title: {video_title_type}")
                            if video_title_type == str:
                                self.logger.info(f"and length of {len(video_data['title'])}")

                            self.logger.info(f"Type for description: {video_description_type}")
                            if video_description_type == str:
                                self.logger.info(f"and length of {len(video_data['description'])}")

                            self.logger.info(f"Type for tags: {video_tags_type}")
                            if video_tags_type == list:
                                self.logger.info(f"and length of {len(video_data['tags'])}")

                            print(f"Error with video data: {video_id}")
                            raise e
                    # add transcripts
            if 'transcript' in video_data:
                combined_text += ' ' + str(video_data['transcript'])
            # add comments
            if 'comments' in video_data:
                combined_text += ' ' + str(video_data['comments'])

            all_text += ' ' + combined_text
        # Filter out words with less than four characters
        words = re.findall(r'\b\w{4,}\b', all_text.lower())

        filter_words = {'youtube', 'website', 'https', 'facebook', 'twitter', 'amzn', 'amazon'
                        'instagram', 'youtu', 'video', 'music', 'channel', 'playlist', 'tiktok',
                        'and', 'but', 'or', 'so', 'because', 'like', 'just', 'also', 'really', 'very',
                        'say', 'said', 'see', 'use', 'used', 'try', 'tried', 'go', 'going', 'seem',
                        'seems', 'seemed', 'make', 'makes', 'made', 'take', 'takes', 'took', 'get',
                        'gets', 'got', 'put', 'puts', 'place', 'places', 'placed', 'come', 'comes',
                        'came', 'think', 'thinks', 'thought', 'tell', 'tells', 'told', 'find', 'finds',
                        'found', 'give', 'gives', 'gave', 'look', 'looks', 'looked', 'ask', 'asks', 'asked',
                        'need', 'needs', 'needed', 'feel', 'feels', 'felt', 'seem', 'seems', 'seemed', 
                        'appear', 'appears', 'appeared', 'keep', 'keeps', 'kept', 'begin', 'begins', 'began', 
                        'start', 'starts', 'started', 'show', 'shows', 'showed', 'hear', 'hears', 'heard',
                        'let', 'lets', 'letting', 'continue', 'continues', 'continued', 'set', 'sets', 'setting'
                        'thing', 'will', 'well', 'know', 'yeah', 'okay', 'right', 'want', 'okay', 'thing', 'good'
                        'much', 'thing', 'word', 'things','stuff', 'because', 'good', 'mean', 'many'
                    }
        filter_words.update(self.nlp.Defaults.stop_words)
        # Remove filter words from the list
        filtered_words = [word for word in words if word not in filter_words]

        # Create a word cloud
        wordcloud = WordCloud(
            width=1200,  # Consider if this size fits your display needs
            height=800,  # Adjust if a different aspect ratio is desired
            background_color='white',
            max_words=50,  # Increased for more word density
            max_font_size=100,  # Adjusted for larger most frequent words
            min_font_size=10,  # Slightly increased minimum font size
            scale=4,  # Increased for better resolution
            relative_scaling=0.5  # Adjusted for frequency impact on word size
        ).generate(' '.join(filtered_words))

        # Display the word cloud
        
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = title = f"Word Cloud of All text found from {video_count} Videos on {subject}"
        plt.title(title, fontsize=14, pad=25)
        if autosave:
            return fig, title
        else:
            plt.show()

    def plot_video_metrics(self, video_list, cutoff=35, sort_by='like_count'):
        """
        Plot metrics (view count, like count, comment count) for a list of videos.

        Parameters:
        video_list (list): A list of video IDs for which metrics will be plotted.
        cutoff (int): The maximum number of videos to include in the plot.
        sort_by (str): The metric by which to sort the videos (e.g., 'like_count', 'view_count', 'comment_count').
        """
        if not video_list:
            print("No videos provided for plotting.")
            return

        # Convert video_list to a list if it is not already
        if not isinstance(video_list, list):
            video_list = list(video_list)

        # Ensure sort_by is a valid metric
        valid_metrics = ['view_count', 'like_count', 'comment_count']
        if sort_by not in valid_metrics:
            print(f"Invalid sort_by value: {sort_by}. Valid options are {valid_metrics}.")
            return

        # Sort the video list by the specified metric
        video_list.sort(key=lambda vid: -sum(int(self.safe_int_conversion(data[sort_by])) for data in self.video_data[vid]))

        # Limit the number of videos to the cutoff
        video_list = video_list[:cutoff]

        num_metrics = 3  # View count, like count, comment count
        num_videos = len(video_list)

        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 8 * num_metrics))

        video_titles = []
        video_ids = []

        for i, metric in enumerate(valid_metrics):
            metric_data = [sum(self.safe_int_conversion(data[metric]) for data in self.video_data[vid]) for vid in video_list]

            axes[i].bar(range(num_videos), metric_data, color='b', alpha=0.7)
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} for Top {num_videos} Videos (Sorted by {sort_by.replace("_", " ").capitalize()})')

            # Collect video titles and IDs for the legend
            for vid in video_list:
                for data in self.video_data[vid]:
                    video_titles.append(data['title'])
                    video_ids.append(vid)
            video_ids.extend([vid for vid in video_list])

        # Adjust spacing for the legend
        plt.subplots_adjust(right=0.8)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)
        plt.show()



    def filter_videos_by_criteria(self, filter_type, min_threshold=None, max_threshold=None):
        """
        Filters videos based on given criteria.

        Args:
            filter_type (str): Type of filter. Must be one of the following
                'video_id': item['id'],
                'subject': 
                'title': 
                'description': 
                'tags': 
                'url': 
                'view_count': ,
                'like_count':
                'comment_count': it
                'upload_date': 
                'thumbnail_url':
                'next_page_token'
            min_threshold (int): Minimum threshold to filter by.
            max_threshold (int): Maximum threshold to filter by.

        Returns:
            list: Filtered video IDs.
        """
        filtered_videos = []

        for video_id, video_data_list in self.video_data.items():
            for video_data in video_data_list:
                metric_value = int(video_data.get(filter_type, 0))
                if (min_threshold is None or metric_value >= min_threshold) and (max_threshold is None or metric_value <= max_threshold):
                    filtered_videos.append(video_id)

        return filtered_videos

    def calculate_ratio(self, numerator, denominator):
        """
        Calculates the ratio of two values, safely converting strings to integers.

        Args:
        numerator (str or int): The numerator of the ratio.
        denominator (str or int): The denominator of the ratio.

        Returns:
        float: The calculated ratio, or 0 if the denominator is zero or conversion fails.
        """
        try:
            numerator = int(numerator)
            denominator = int(denominator)
            return numerator / denominator if denominator != 0 else 0
        except ValueError:
            return 0

    def safe_int_conversion(self, value, default=0):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
        
    def extract_named_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        return entities

    def print_video_data(self):
        """
        Prints the contents of self.video_data, including types of objects and the first 5 key-value pairs.
        """
        # Check if video_data_profile exists and is not empty
        if not hasattr(self, 'video_data') or not self.video_data:
            self.logger.info("video_datais empty or not initialized.")
            return

        # Print the type of video_data_profile
        self.logger.info(f"Type of video_data: {type(self.video_data)}")

        # Print the first 5 key-value pairs
        counter = 0
        for video_id, details in self.video_data.items():
            self.logger.info(f"Video ID: {video_id}, Type of details: {type(details)}")
            self.logger.info(f"Details: {details[:5]}")  # Print first 5 details if available
            counter += 1
            if counter >= 5:
                break

        # Print the total number of keys for context
        total_keys = len(self.video_data)
        self.logger.info(f"Total number of video IDs in video_data: {total_keys}")

    # Saving to File \ Reading From File
    #   Functions
    #      Below ---- thumbnails to text, headlines to text, video_data_profile to .csv

    def clean_string(self, file_path):
        # Replace problematic characters with alternatives or remove them
        cleaned_file_path = re.sub(r'[\\/:*?"<>|]', '_', file_path)
        return cleaned_file_path

    def save_graph(self, figure, title, file_dir):
        """
        Saves a CHART matplot fig object to a specific file_path

        Args:
            figure (matplotlib.figure.Figure): The word cloud figure to be saved.
            file_path (str): The file path where the figure will be saved.
        """
        title = title.replace(':', '_').replace(' ', '_').replace("'", '')  # Replace ':' and spaces with underscores
        title = self.clean_string(title)
        #clean the title to ensure able to save

        # Combine the cleaned filename with the directory path
        file_path = os.path.join(file_dir, f'{title}.jpg')

        figure.savefig(file_path)
        plt.close(figure)  # Close the figure after saving

        if os.path.isfile(file_path):
            self.logger.info(f"Chart SAVED to {file_path}")
        else:
            self.logger.info(f"Chart NOT SAVED {file_path}")

    def create_plots_for_directory(self, input_directory, new_directory_path, func, *args, **kwargs):
        """
        Call create_and_save_plots over every folder in a directory.
        REQUIREMENT: .csv of the data within initial file in input_directory
        Populates all folders in the directory with graphs of data.

        Args:
        - input_directory (str): The initial directory to walk through.
        - new_directory_path (str): The name of the new directory to create within each subdirectory.
        """
        self.logger.info(f"Starting create_plots_for_directory for {input_directory} - adding extension {new_directory_path}")
        self.logger.info(f"Function: {func.__name__}")
        self.logger.info(f"Args: {args}")
        self.logger.info(f"Kwargs: {kwargs}")


        # Walk through the directory
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    file_directory = os.path.dirname(file_path)  # Get the directory of the CSV file

                    # Create the new directory in the same location as the CSV file
                    folder_path = os.path.join(file_directory, new_directory_path)

                    # Ensure the directory doesn't exist before creating it
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        print(f"Created directory: {folder_path}")  # Print for debugging purposes

                    print(f"Processing file: {file_path}")

                    if self.video_data: # Reset video_data before loading new video data
                        self.video_data = {}
                    # Load video ids from the CSV file
                    self.load_video_data_from_csv(file_path)
                    video_ids = list(self.video_data.keys())
                    #print(f"Loaded video IDs: {video_ids}")

                    # Call create_and_save_plots and with passed args/kwargs
                    self.create_and_save_plots(video_ids, folder_path, func, *args, **kwargs)
                    print(f"create_and_save_plots called for: {folder_path} | using: {func.__name__} | with args: {args} | and kwargs: {kwargs}")

        self.logger.info("Finished processing directory.")

    def create_and_save_plots(self, video_ids, save_dir, function, *args, **kwargs):
        """
        Iterates through all video IDs in `video_ids`, gathers unique subjects of all video IDs,
        and dynamically calls the specified function for plotting graphs based on the subjects or conditions provided.

        *Functions must return the fig object of the graph and not plot the graph directly.
        Functions must also return a "title" used in the graph, utilized as a filename in `save_graph`.

        Args:
            video_ids (list): List of video IDs to process.
            save_dir (str): The directory where individual graph images will be saved.
            function (function): A function that generates a plot based on specified parameters.

        *args (list): Variable positional arguments to be passed to the `function`.
        **kwargs (dict): Variable keyword arguments to be passed to the `function`.

        Note:
            The behavior of function invocation depends on the arguments provided:
            - If 'subject' is included in *args, the function iterates through each unique subject,
            generating graphs based on subject-specific data.
            - If 'autosave' is provided in **kwargs and 'subject' is not in *args,
            the function generates a graph based on general criteria without subject iteration.
            - If neither 'subject' is in *args nor 'autosave' in **kwargs, an error is logged.

        Returns:
            None: The function saves the generated graphs to the specified `save_dir`.

        Raises:
            Error: Logs an error if the function encounters an exception while generating or saving graphs.
        """
        self.logger.info(f"Function: {function.__name__}, args: {args}, kwargs: {kwargs}")
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            self.logger.info(f'Created {save_dir} directory.')
            os.makedirs(save_dir)

        # Find unique subjects from the given video IDs
        unique_subjects = set()
        
                        #self.logger.info(f"Added {data['subject']} to unique_subjects set")

        # Iterate through each unique subject and create word chart
        if 'subject' in args:  # Check if 'subject' is passed in *args
            self.logger.info(f"Detected: 'subject' in *args")
            for vid in video_ids:
                if vid in self.video_data:
                    for data in self.video_data[vid]:
                        if 'subject' in data:
                            unique_subjects.add(data['subject'])
            for subject in unique_subjects:
            #Activate line above to iterate through subjects
                try:
                    fig, title = function(subject, **kwargs)
                    print(f"Title returned from graph function: {title} | Subject: {subject}")
                except Exception as e:
                    self.logger.error(f"{e} Error occured using title:{title} fig{fig}")
                else:
                    self.save_graph(fig, title, save_dir)
                    
        elif 'autosave' in kwargs and kwargs['autosave'] is True and 'subject' not in args and "radar_chart" not in args:
        
            #Check if 'autosave' is passed as a keyword argument
            self.logger.info("Detected 'autosave' only in **kwargs'")
            try:
                fig, title = function(*args, **kwargs)
            except Exception as e:
                self.logger.error(e)
            else:
                self.logger.info(f"Adding {title} to {save_dir}... Calling save_graph")
                self.save_graph(fig, title, save_dir)
                
        elif 'radar_chart' in args:  # Check if 'video_id' is passed in *args
            self.logger.info(f"Detected: 'radar_chart' in *args")
            try:
            
                figures_and_titles = function(**kwargs)  # Assuming function returns a list of tuples

                if figures_and_titles is not None:  # Check if data is retrieved
                    for fig, title in figures_and_titles:
                        if fig is not None:  # Check if fig is not None
                            print(f"Title returned from graph function: {title} Video ID: {args[0]}")

                            self.logger.info(f"Adding {title} to {save_dir}... Calling save_graph")
                            self.save_graph(fig, title, save_dir)  # Pass full path to save_graph
                        else:
                            self.logger.warning(f"No valid fig object for {vid}. Skipping save_graph.")
            except Exception as e:
                self.logger.error(f"{e} Error occurred trying to save video. Title:{title} Fig{fig}")
        else:
            self.logger.error(f"Invalid arguments. 'subject' or 'video_id' not in *args and 'autosave' not in **kwargs")


    def process_and_save_thumbnail_analysis(self, csv_output_filename, input_file=None):
        """
        Processes thumbnail analysis for all videos in self.video_data that have a thumbnail URL.
        If self.video_data is not already loaded, it loads data from input_file. Then, it updates
        self.video_data with the analysis results and saves these results to a CSV file.

        Parameters:
        csv_output_filename (str): Name of the CSV file to save the thumbnail analysis.
        input_file (str, optional): Name of the input CSV file to load data from. Defaults to None.
        """
        if not self.video_data or (input_file and hasattr(self, 'video_data_profile')):
            self.logger.info(f"Loading input file: {input_file}")
            self.load_video_data_from_csv(input_file)

        fieldnames = ['video_id', 'thumbnail_url', 'resolution', 'aspect_ratio', 'mode', 'content_labels']

        for video_id, video_data in self.video_data.items():
            for data in video_data:
                thumbnail_url = data.get('thumbnail_url')
                if thumbnail_url:
                    thumbnail_analysis = self.full_thumbnail_analysis(thumbnail_url)
                    data['thumbnail_analysis'] = [thumbnail_analysis]  # Replace any existing analysis

        with open(csv_output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for video_id, video_data in self.video_data.items():
                for data in video_data:
                    if 'thumbnail_analysis' in data:
                        for analysis in data['thumbnail_analysis']:
                            row = {
                                'video_id': video_id,
                                'thumbnail_url': data.get('thumbnail_url'),
                                'resolution': f"{analysis['quality']['resolution'][0]}x{analysis['quality']['resolution'][1]}",
                                'aspect_ratio': analysis['quality']['aspect_ratio'],
                                'mode': analysis['quality']['mode'],
                                'content_labels': ', '.join(analysis['content'])
                            }
                            writer.writerow(row)

        self.logger.info(f"Thumbnail analysis completed and saved to '{csv_output_filename}'.")

    def update_and_save(self, videos_to_find, output_filename, input_filename):
        """
        Findes next (videos_to_find) (int) videos by calling self.get_top_videos() 
        according to next page tokens from the .CSV file to create a new 
        self.video_data and save the updated dictionary to a new CSV file (output_filename).

        Parameters:
        videos_to_find (int): Parameter passed to self.get_top_videos(max_results=videos_to_find)
        input_filename (str): Filename from existing CSV file.
        output_filename (str): Filename for the new CSV file.
        """
        # Step 1: Load existing data
        self.logger.info(f"Loading existing video data from {input_filename}")
        self.load_video_data_from_csv(input_filename)
        # Step 2: Merge new data
        self.logger.info(f"Updating video data with the next {videos_to_find} videos")
        self.search_next_page_for_subjects(videos_to_find)
        # Step 3: Save combined data
        self.logger.info(f"Saving updated video data to {output_filename}")
        self.save_video_data_to_csv(output_filename)

    def save_video_data_to_csv(self, filename):
        """
        Saves the video data from self.video_data_profile to a CSV file.

        Parameters:
        filename (str): The path to the CSV file to be written.
        """
        required_fields = ['video_id', 'title', 'view_count', 'like_count', 'comment_count', 
                        'description', 'tags', 'url', 'upload_date', 'thumbnail_url',
                        'next_page_token']

        # Create an empty list to store all rows
        try:
            all_data_rows = []

            # Validate data and prepare rows
            for video_id, data_list in self.video_data.items():
                for data in data_list:
                    if isinstance(data, dict):
                        # Check if all required fields are present
                        if all(field in data for field in required_fields):
                            data_with_id = data.copy()
                            data_with_id.setdefault("video_id", video_id)
                            all_data_rows.append(data_with_id)
                        else:
                            missing_fields = [field for field in required_fields if field not in data]
                            self.logger.warning(f"Missing required fields {missing_fields} for video ID {video_id}")
                    else:
                        self.logger.warning(f"Invalid data format for video ID {video_id}")

            # Convert to DataFrame and save to CSV
            if all_data_rows:
                df = pd.DataFrame(all_data_rows)
                df.to_csv(filename, index=False)
                self.logger.info(f"Data successfully saved to '{filename}'")
            else:
                self.logger.warning("No data to save.")
        except Exception as e:
            self.logger.exception(f"An error occurred while saving data to {filename}: {e}")

    def load_video_data_from_csv(self, filename, chunk_size = 5000):
        """
        Loads video data from a CSV file and merges it into self.video_data while
        avoiding any duplicates.

        Parameters:
        filename (str): The path to the CSV file to be read.
        """
        # Log the start of the loading process
        self.logger.info(f'Loading videos from {filename}')
        
        # Store filename for future reference if needed
        self.file_name = filename
        
        # Define fields that should be converted to numeric types
        numeric_fields = ['view_count', 'like_count', 'comment_count']
        
        # Initialize a counter for skipped videos due to missing video_id
        skipped_videos = 0

        # Read the CSV file in chunks to manage memory usage
        for chunk in pd.read_csv(filename, chunksize=chunk_size, encoding='utf-8', dtype=str):
            # Log the size of the current chunk being processed
            self.logger.info(f"Processing a chunk that is {len(chunk)} long")

            # Iterate through each row in the chunk
            for index, row in chunk.iterrows():
                # Extract the video_id from the row
                video_id = row.get('video_id')

                # Check if the video_id is missing or null
                if not video_id:
                    skipped_videos += 1  # Increment the skipped video counter
                    continue  # Skip to the next row

                try:
                    # Convert numeric fields to integers and handle NaN values
                    new_data = {k: int(float(v)) if k in numeric_fields and pd.notna(v) else v for k, v in row.items()}
                    
                    # Check if the video_id is already in the video_data and if the new data is not a duplicate
                    if video_id not in self.video_data or new_data not in self.video_data[video_id]:
                        # Add the new data to the video_data dictionary, creating a new list if video_id is new
                        self.video_data.setdefault(video_id, []).append(new_data)
                    else:
                        # Log if the data for this video_id is a duplicate and skip adding it
                        self.logger.debug(f"Skipping duplicate data for video_id {video_id}")
                except ValueError as e:
                    # Log any error encountered during the data conversion process
                    self.logger.error(f"Error processing row {index} for video_id {video_id}: {e}")

        # Log the total number of videos processed and how many were skipped
        self.logger.info(f"Loaded data from '{filename}'. Total videos added: {len(self.video_data)}. Skipped videos: {skipped_videos}")

        # Debug statement to print a part of self.video_data
        #sample_keys = list(self.video_data.keys())[:5]  # Adjust the slice as needed
        #for key in sample_keys:
        #self.logger.debug(f"Sample data for video ID {key}: {self.video_data[key][:1]}")  # Print first entry for each sample video ID

    def row_is_empty(self, row): # helper function for load_video_data_from_csv
        """
        Check if a row from CSV is empty or missing essential data.

        :param row: Row of data from CSV
        :return: True if the row is empty or missing essential data, False otherwise
        """
        # Define essential columns that must have data
        essential_columns = ['video_id', 'title', 'view_count']  # Add other essential columns here

        return any(not row.get(col) for col in essential_columns)
    
    def merge_csv_files(self, files):
        """
        Merges multiple CSV files, merges on all columns, and keeps files that have empty fields.
        Removes duplicates, prints the amount of duplicates to terminal.
        Use yt.merg_files([files]).to_csv("output_file.csv") to save file, else returns dataframe object

        Args:
            files (list): List of file paths to be merged.

        Returns:
            DataFrame: Merged DataFrame with empty fields dropped and duplicates removed.
        """
        # Check if files exist
        existing_files = [file for file in files if os.path.exists(file)]
        if not existing_files:
            raise FileNotFoundError("No valid files found.")

        # Read and merge files
        dfs = [pd.read_csv(file) for file in existing_files]
        merged_data = pd.concat(dfs, ignore_index=True)

        # Count rows before removing duplicates
        total_rows_before = len(merged_data)

        # Remove duplicates
        merged_data.drop_duplicates(inplace=True)

        # Count rows after removing duplicates
        total_rows_after = len(merged_data)

        # Count duplicate rows
        duplicate_rows_count = total_rows_before - total_rows_after

        # Print the number of duplicate rows
        print(f"Number of duplicate rows across all files: {duplicate_rows_count}")

        return merged_data
    
    def merge_and_drop_empty(self, files):
        """
        Merges multiple CSV files, merges on all columns, and drops entries with any empty fields.
        Removes Duplicate entires.
        Use yt.merg_files([files]).to_csv("output_file.csv") to save file, else returns dataframe object

        Args:
            files (list): List of file paths to be merged.

        Returns:
            DataFrame: Merged DataFrame with empty fields dropped.
        """
        # Check if files exist
        existing_files = [file for file in files if os.path.exists(file)]
        if not existing_files:
            raise FileNotFoundError("No valid files found.")

        # Read and merge files
        dfs = [pd.read_csv(file) for file in existing_files]
        merged_data = pd.concat(dfs, ignore_index=True)

        # Remove entries with any empty fields not in needed_fields

        needed_fields = ["title", "video_id", "view_count", "subject", "next_page_token"]
        cols_to_drop = [col for col in merged_data.columns if col not in needed_fields]
        merged_data.dropna(subset=cols_to_drop, how='any', inplace=True)
        
        num_duplicates = merged_data.duplicated().sum()
        #Remove dupicates
        merged_data.drop_duplicates(inplace=True)
        print(f"Number of duplicates dropped: {num_duplicates}")

        return merged_data

    def load_data_and_fetch_transcripts(self, csv_filename, num_videos, start_num = None):
        """
        Loads video data from a CSV file, selects a specified number of videos, and fetches their transcripts.
        
        Args:
            csv_filename (str): The path to the CSV file containing video data.
            num_videos (int): The number of videos for which to fetch transcripts.

        Updates:
            self.video_data: A dictionary where each value is a list containing dictionaries of video data, including transcripts.
        """
        # Step 1: Load video data from CSV
        self.load_video_data_from_csv(csv_filename)

        # Step 2: Filter the first 'num_videos' videos
        
        if (start_num):
            filtered_video_ids = list(self.video_data.keys())[start_num: (num_videos+start_num)]
        else:
            filtered_video_ids = list(self.video_data.keys())[:num_videos]

        video_ids_with_transcript = [video_id for video_id in filtered_video_ids if self.is_transcript_available(str(video_id))]

        # Step 3: Fetch transcripts for the filtered videos
        self.get_transcripts(video_ids_with_transcript)

    def get_data_from_column(self, column):
        data_list = []
        if self.video_data:
            for vid_id, data_list in yt.video_data.items():
                for data in data_list:
                    if column in data and data[column]:
                        data_list.append(data[column])
        else:
            print("No data loaded yet. Must call load_data_from_csv('filename')")
            return
        return data_list
    
    def save_data_to_txt(self, data, file_name, directory=None,):
        """Save data from a column/field to a .txt file.

        Args:
            data, preferably a list of strings. 
            file_name: name to save file under
            directory: Optionaly to create new directory to save file
        """
        if directory:
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_path = os.path.join(directory, file_name)
        else:
            file_path = file_name

        with open(file_path, 'w', encoding='utf-8') as file:
            for datas in data:
                if isinstance(datas, dict):
                    print(f"Detected dict within data (save_data_to_txt(data, filename)) - converting to string")
                    datas = json.dumps(datas)  # Convert dict to string
                file.write(datas + "\n")

        print(f"\n Data saved to {file_name} \n")

    def save_ts_report_to_csv(self, filename="analysis_report_00.csv"):
    # Create CSV file and write the data from transcript report from GPT4

        csv_data = []

        prompt_to_header = {
            "Content Type: Classify as educational, entertainment, or a blend.": "Content Type",  # Assuming best fit
            "Target Audience: Identify problems/pain points, interests, and desires.": "Target Audience",
            "Introduction Quality: Evaluate opening lines for engagement and clarity of expectations.": "Introduction Quality",
            "Narrative Structure: Assess structure (problem-solution, chronological, storytelling) for audience interest.": "Narrative Structure",
            "Pacing and Rhythm: Examine content flow and balance between information and engagement.": "Pacing and Rhythm",
            "CTAs: Analyze placement and nature of calls to action.": "CTAs",
            "Language and Tone: Review language appropriateness and tone consistency.": "Language and Tone",
            "Storytelling Elements: Examine anecdotes, metaphors, and personal stories.": "Storytelling Elements",
            "Educational Content Delivery: For educational scripts, assess clarity and structure of information.": "Educational Content Delivery",
            "Entertainment Value: For entertainment scripts, evaluate humor, drama, suspense.": "Entertainment Value",
            "Viewer Engagement Strategies: Analyze interactive elements like questions and community involvement.": "Viewer Engagement Strategies",
            "Use of Keywords: Assess SEO-friendly keywords.": "Use of Keywords",
            "Conclusion Effectiveness: Evaluate conclusion for summarization and viewer direction.": "Conclusion Effectiveness",
            "Length and Segmentation: Review script length and topic segmentation.": "Length and Segmentation",
            "Cross-Platform References: Identify social media and platform integration.": "Cross-Platform References",
            "Data and Fact Accuracy: Verify accuracy of presented information.": "Fact Check"  # Assuming best fit
        }

        header = ["Video ID"] + list(prompt_to_header.values())

        for video_id, video_data_list in self.video_data.items():
            for video_data in video_data_list:
                row = {'Video ID': video_id}
                for prompt, header_key in prompt_to_header.items():
                    row[header_key] = video_data.get(prompt, 'N/A')
                csv_data.append(row)

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

    def jpgs_2_pdf(self, directory, output_pdf='all_in_one.pdf'):
        """
        Compile all .jpg files in a single directory into a single PDF, with each image as a full page.
        Move the output PDF file to the specified directory.

        :param directory: Directory containing .jpg files
        :param output_pdf: Name of the output PDF file
        """
        images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
        if not images:
            print("No .jpg files found in the specified directory.")
            return

        pdf = FPDF()

        for image_path in images:
            fig = plt.figure(figsize=(8.27, 11.69))  # Set figure size to A4 paper size (adjust as needed)
            plt.axis('off')  # Hide axes for a cleaner image
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Hide axes for a cleaner image
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot to remove whitespace
            pdf.add_page()
            pdf.image(image_path, x=0, y=0, w=pdf.w, h=pdf.h)
            plt.close(fig)  # Close the figure after adding it to the PDF

        output_path = os.path.join(directory, output_pdf)
        pdf.output(output_path)
        print(f"PDF created successfully: {output_pdf}")

        # Move the output PDF to the specified directory
        shutil.move(output_path, os.path.join(directory, output_pdf))

    def process_directory_jpg2pdf(self, directory):
        """
        Process all files within the specified directory.
        Compile all jpg's in directory to one pdf directories containing .jpg files, call compile_jpg_to_pdf method.
        For batch processing of all charts as .jpg  within a file.

        :param directory: Parent directory to process
        """
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                dir_path = os.path.join(root, d)
                jpg_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
                if jpg_files:
                    print(f"Processing .jpgs for {dir_path} - now calling jpgs_2_pdf")
                    self.jpgs_2_pdf(dir_path, f'{d}_all_charts.pdf')  # Call compile_jpg_to_pdf method

        
    
    def merge_pdfs(self, root_directory):
        """
        Merge all PDF files found within a specified root directory into a single PDF file.

        Parameters:
        - root_directory (str): The root directory containing PDF files to be merged.

        Output:
        - Creates a new PDF file that contains the merged content of all PDFs found within the root directory.
        The file is saved with a name reflecting the current date in the format 'YYYY-MM-DD_merged.pdf'.
        """
        pdf_writer = PdfWriter()
        date_today = datetime.now().strftime("%Y-%m-%d")
        output_filename = f"{date_today}_merged.pdf"

        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    pdf_reader = PdfReader(file_path)
                    for page in pdf_reader.pages:
                        pdf_writer.add_page(page)

        with open(output_filename, 'wb') as output_file:
            pdf_writer.write(output_file)

        print(f"Merged PDF created successfully: {output_filename}")

    def generate_sample_data(filename, num_rows=100):
        # Generate random data to sample and troubleshoot functions
        data = {
            'video_id': [f'vid_{i}' for i in range(num_rows)],
            'subject': [f'Subject {i % 5}' for i in range(num_rows)],  # 5 different subjects
            'title': [f'Title {i}' for i in range(num_rows)],
            'description': [f'Description for video {i}' for i in range(num_rows)],
            'tags': [f'tag1,tag2,tag{i % 3}' for i in range(num_rows)],  # 3 different sets of tags
            'url': [f'https://youtu.be/{i}' for i in range(num_rows)],
            'view_count': np.random.randint(0, 10000, num_rows).astype(object),
            'like_count': np.random.randint(0, 1000, num_rows).astype(object),
            'comment_count': np.random.randint(0, 500, num_rows).astype(object),
            'upload_date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_rows)],
            'thumbnail_url': [f'https://thumbnail.url/{i}.jpg' for i in range(num_rows)],
            'next_page_token': [f'token_{i}' if i % 10 == 0 else np.nan for i in range(num_rows)],  # NaN for some tokens
            'token_time_stamp': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(num_rows)]
        }

        # Introduce some NaN values
        for key in ['view_count', 'like_count', 'comment_count']:
            data[key][np.random.choice(num_rows, 10, replace=False)] = np.nan

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def count_missing_fields(self, csv_filename):
        df = pd.read_csv(csv_filename)
        
        null_counts = df.isnull().sum()
        
        print(f"Total fields: {len(df)}")
        print("Number of missing fields per column:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"{col}: {count} missing values")
    
    def find_overlapping_rows(self, csv1, csv2):

        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)
        
        merged = df1.merge(df2, indicator=True, how='outer')
        overlaps = merged[merged['_merge'] == 'both']
        c
        print(f"Number of overlapping rows: {len(overlaps)}")
        print(f"\nNumber of rows in csv1: {len(df1)}")
        print(f"\nNumber of rows in csv2: {len(df2)}")

        if len(overlaps) > 0:
            print("\nOverlapping rows:")
            print(overlaps)


yt = YTFunc()


#yt.create_plots_for_directory("C:\Python\YTFunc\Charts", "Word Clouds\Title Words", yt.plot_word_cloud_titles, "subject", autosave=True)
#yt.jpgs_2_pdf("C:\Python\YTFunc\Charts\Virality\Word Clouds\Title Words")



#yt.search_youtube_videos(max_videos = 100, sort_order="likes", channel_id = "UCQ5mWx_XYGRbpcV8zjGhorg")
#yt.save_video_data_to_csv("jeremyminer.csv")


#yt.load_video_data_from_csv("LoL_strat_ts.csv")
#yt.analyze_transcripts()


#yt.load_video_data_from_csv("paddy_gallow_strategy.csv")
#yt.plot_views_to_title_chars_scatter("strategey")
#yt.plot_views_title_length_line_chart("strategey")
#yt.plot_top_tags_pie_chart("strategey")
#yt.plot_word_cloud_titles("strategey")
#yt.plot_common_words_in_titles("strategy")
#yt.plot_word_cloud_all("strategey")
#yt.plot_word_cloud_tags("strategey")
#yt.plot_video_metrics(yt.video_data.keys(), sort_by="view_count")

#
# yt.download_youtube_clip("https://youtu.be/HDzhA_UFrnA?si=gbJOboenmJxBLvXo", "00:30:18", "00:30:35", "ig-ceo-advertising-business-attention.mp4")
#yt.get_top_videos("Viral Video Editing", max_results=5)
#yt.get_all_comments()
#yt.analyze_comments()
#yt.get_transcripts(yt.video_data.keys())
#yt.analyze_transcripts()
#yt.save_video_data_to_csv("viral-script-writing.csv")

#yt.download_youtube_audio_clip("https://www.youtube.com/watch?v=FVHLY0fiKXI", "grateful.mp3")


#yt.load_video_data_from_csv("video_ads_0_ts.csv")
#yt.oai_thumbnails_to_csv(20, "video_ads_0_thumbnail_analysis.csv")

#yt.download_video("https://www.youtube.com/watch?v=Fxe5ImKqBA4", "jack_gordon_tikok_algorithm.mp4")
yt.extract_images("/home/dell/Videos/yt-dl/jack_gordon_tiktok_algorithm.mp4", "video_frames_extraction_10fps", fps = 10, duration = 10)
