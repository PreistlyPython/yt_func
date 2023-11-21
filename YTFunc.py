from googleapiclient.discovery import build
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi, NoTranscriptFound
from collections import Counter
import nltk
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
import random

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
        self.nlp = spacy.load("en_core_web_sm")
        self.video_data_profile = {} # a dictionary of keys = video_ids values = dictionarys containing data relating to video.
        self.yt_v3_key = 'AIzaSyCVeYN4eGlnoSXHOieZcx_nqjy-qEt26_k'
        self.openai_key = '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'
        self.youtube = build('youtube', 'v3', developerKey = self.yt_v3_key)
        openai.api_key = self.openai_key
        self.gpt_client = OpenAI(api_key=self.openai_key)
        self.valid_videos = [] 
        self.pp = pprint.PrettyPrinter(indent=4)
        self.subject_searched = ''
        self.videos_downloadable =[]
        self.next_page_token = ''
        self.script_report = []
        self.script_pre_prompt = "Analyze the following youtube script " 
        self.script_prompts = ["Content Type: Classify as educational, entertainment, or a blend.", "Target Audience: Identify problems/pain points, interests, and desires.", "Introduction Quality: Evaluate opening lines for engagement and clarity of expectations.", "Narrative Structure: Assess structure (problem-solution, chronological, storytelling) for audience interest.", "Pacing and Rhythm: Examine content flow and balance between information and engagement.", "CTAs: Analyze placement and nature of calls to action.", "Language and Tone: Review language appropriateness and tone consistency.", "Storytelling Elements: Examine anecdotes, metaphors, and personal stories.", "Educational Content Delivery: For educational scripts, assess clarity and structure of information.", "Entertainment Value: For entertainment scripts, evaluate humor, drama, suspense.", "Viewer Engagement Strategies: Analyze interactive elements like questions and community involvement.", "Use of Keywords: Assess SEO-friendly keywords.", "Conclusion Effectiveness: Evaluate conclusion for summarization and viewer direction.", "Length and Segmentation: Review script length and topic segmentation.", "Cross-Platform References: Identify social media and platform integration.", "Data and Fact Accuracy: Verify accuracy of presented information."]
        self.script_post_prompt = " | using the LEAST amount of words possible while focusing your analysis ONLY on the topic of - "
        logging.info("YTFunc instance created.\n")
        #nltk.download('punkt')
        #nltk.download('averaged_perceptron_tagger')
        #self.number_of_videos_max = number_of_videos_max


    def ask_gpt4(self, text): # text parameter is the prompt for the GPT 4 turbo -
        """
        Sends a text prompt to the GPT-4 model and returns its response.

        Parameters:
        text (str): The prompt text to be sent to GPT-4.

        Returns:
        str: The response text from GPT-4.
        """
        response = self.gpt_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are capable an analyzing text and providing accurate and reliable infomration about the next quesiton."},
            {"role": "user", "content": text},
        ],
        
            max_tokens = 100,
            temperature = 0
        )
        return response['choices'][0]['message']['content']

        #gpt-4-1106-preview | gpt 4
        #gpt-4-vision-preview
        #
    def get_topics_from_gpt(self, text): #Use OPEN AI API to query GPT 4 and get topics from text
        """
        Queries GPT-4 to identify topics from the provided text.

        Parameters:
        text (str): The text from which topics need to be extracted.

        Returns:
        str: Topics identified by GPT-4 from the given text.
        """

        response = self.gpt_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a world class reporter and summarizer of data with a uncanny ability of identifying topics covered from text."},
            {"role": "user", "content": "Return all topcis covered in the following text. [" + text + "]"},
        ],
        

            temperature = 0,
            max_tokens = 100
        )
        return response['choices'][0]['message']['content']


    def get_audio(self, url): #Return audio file from Youtube Url LInk
        yt = YouTube(url)
        return yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")
    
    def get_transcript_from_audio(self, url, model_size, lang, format): #Return transcript of audio file

        model = whisper.load_model(model_size)

        if lang == "None":
            lang = None
        
        result = model.transcribe(self.get_audio(url), fp16=False, language=lang)

        if format == "None":
            return result["text"]
        elif format == ".srt":
            return self.format_to_srt(result["segments"])

    def format_to_srt(self, segments):
        output = ""
        for i, segment in enumerate(segments):
            output += f"{i + 1}\n"
            output += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            output += f"{segment['text']}\n\n"
        return output

    def get_top_videos(self, query, max_results = 10):  #Lines 121,122 check if TRANSCRIPTS and Downloadable
        valid_videos = []
        self.subject_searched = query
        search_attempt = 0
        next_page_token = None
        # Exponential backoff parameters
        backoff_time = 1  # Initial backoff time in seconds
        max_backoff_time = 32  # Maximum backoff time in seconds
        while len(valid_videos) < max_results and search_attempt < 10:
            try:
                request = self.youtube.search().list(
                    q=query,
                    part='snippet',
                    type='video',
                    maxResults=max_results + search_attempt,
                    order='viewCount',
                    pageToken = next_page_token)
                
                response = request.execute()
                next_page_token = response.get('nextPageToken')
                self.next_page_token - next_page_token

                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    #if self.is_video_valid(video_id):    ENABLE WHEN SEARCHING TRANSCRIPtS
                    valid_videos.append(item)         
                    if len(valid_videos) == max_results:
                        break

                backoff_time = 1
            except Exception as e:
                print(f"Error during search request: {e}")
                time.sleep(backoff_time)
                # Increase backoff time exponentially with some random jitter
                backoff_time = min(backoff_time * 2, max_backoff_time) + random.uniform(0, 1)
                continue  # Continue to next itteration
            finally:
                search_attempt += 1 # Increment to widen the search in the next iteration if needed 

        video_ids = [video['id']['videoId'] for video in valid_videos]
        videos = []
        if video_ids:
            # Split video IDs into smaller chunks if necessary
            video_id_chunks = [valid_videos[i:i + 50] for i in range(0, len(valid_videos), 50)]

            for chunk in video_id_chunks:
                video_ids = [video['id']['videoId'] for video in chunk]
                video_id_str = ','.join(video_ids)

                try:
                    video_request = self.youtube.videos().list(
                        part='snippet,contentDetails,statistics',
                        id=video_id_str
                    )
                    video_response = video_request.execute()

                    for item in video_response['items']:
                        video_id = item.get('id', '')
                        subject = query
                        title = item['snippet'].get('title', '')
                        description = item['snippet'].get('description', '')
                        tags = item['snippet'].get('tags', [])
                        url = f'https://www.youtube.com/watch?v={video_id}' if video_id else ''
                        view_count = item['statistics'].get('viewCount', '0')
                        upload_date = item['snippet'].get('publishedAt', '')
                        thumbnail_url = item['snippet']['thumbnails']['high']['url']                       

                        videos.append({
                            'video_id': video_id,
                            'subject': subject,
                            'title': title,
                            'description': description,
                            'tags': tags,
                            'url': url,
                            'view_count': view_count,
                            'upload_date': upload_date,
                            'thumbnail_url': thumbnail_url,
                            'next_page_token' : next_page_token
                        })

                        if video_id in self.video_data_profile:
                            for video in videos:
                                self.video_data_profile[video_id].append(video)
                        else:
                            self.video_data_profile[video_id] = videos.copy()

                except Exception as e:
                    print(f"Error during video details request: {e}")

        self.pp.pprint(videos)

        return videos

    def load_data_from_ids(self,video_ids): #Function split up get_top_videos
        for video_id in video_ids:
            try:
                video_request = self.youtube.videos().list(
                    part='snippet,contentDetails,statistics',
                    id=video_id
                )
                video_response = video_request.execute()

                for item in video_response['items']:
                    video_detail = {
                        'video_id': item['id'],
                        'subject': self.subject_searched,
                        'title': item['snippet'].get('title', ''),
                        'description': item['snippet'].get('description', ''),
                        'tags': item['snippet'].get('tags', []),
                        'url': f'https://www.youtube.com/watch?v={item['id']}',
                        'view_count': item['statistics'].get('viewCount', '0'),
                        'upload_date': item['snippet'].get('publishedAt', ''),
                        'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                        'next_page_token': self.next_page_token
                    }

                    if video_id in self.video_data_profile:
                        self.video_data_profile[video_id].append(video_detail)
                    else:
                        self.video_data_profile[video_id] = [video_detail]

            except Exception as e:
                print(f"Error during video details request for video ID {video_id}: {e}")

    def find_min_view_count(self, subject):
        """
        Finds the minimum view count among the videos for a given subject.

        Parameters:
        subject (str): The subject to search for in the video data profile.

        Returns:
        int: The minimum view count or None if no videos are found.
        """
        min_views = None
        for video_id, video_data_list in self.video_data_profile.items():
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


    def get_transcripts(self, videos):
        for video in videos:
            video_id = video['video_id']
            print(f"Fetching transcript for video ID: {video_id}")  # Log the video ID being processed
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_generated_transcript(['en']).fetch()
                text = ' '.join([t['text'] for t in transcript])
                video_with_transcript = {}
                video_with_transcript['transcript'] = text  # Add transcript to the copy

                # Append to the list for this video ID in the data profile
                if video_id in self.video_data_profile:
                    self.video_data_profile[video_id].append(video_with_transcript)
                else:
                    self.video_data_profile[video_id] = [video_with_transcript]

                print(f"Transcript for video {video_id} (first 100 chars): {text[:100]}")  # Print first 100 chars
            
            except (NoTranscriptFound, TranscriptsDisabled):
                print(f"No transcript found for video ID: {video_id}")  # Log if no transcript is found
                video_with_no_transcript = video.copy()  # Copy the video dictionary
                video_with_no_transcript['transcript'] = "Transcript NOT Found"  # Add None for transcript
                if video_id in self.video_data_profile:
                    self.video_data_profile[video_id].append(video_with_no_transcript)
                else:
                    self.video_data_profile[video_id] = [video_with_no_transcript]

            

    def analyze_transcripts(self): #Call gpt4 to analyze to comprehensivley analyze the script

        for video_id, video_data_list in self.video_data_profile.items():
            for video_data in video_data_list:
                transcript = video_data.get('transcript', 'value not found')
                if transcript and transcript != 'value not found':
                    logging.info(f"Analyzing transcript from video: {video_data.get('title', 'Unknown Title')}\n")
                    for prompt in self.script_prompts:
                        print("Asking for:  + " + prompt + " ----- * * *------ ")
                        script_response = self.gpt_client.chat.completions.create(
                            model="gpt-4-1106-preview",
                            messages=[
                                {"role": "system", "content": "You are an expert at content creation and a YouTube Script Analyzer"},
                                {"role": "user", "content": self.script_pre_prompt + transcript + self.script_post_prompt + prompt}
                            ]
                        )

                        script_report = script_response.choices[0].message.content
                        self.script_report.append("SCRIPT REPORT FOR | Video id: " + str(video_id) + ": " + script_report + "\n")
                        self.pp.pprint(script_report)

                        # Update the current video_data dictionary with the GPT-4 response
                        video_data[prompt] = script_report
                        print("*** - - ADDED [" + prompt + "] = " + script_report + "\n Above Key/Value pair assigned to " + str(video_id) + " *-*-*-*-")
                        logging.info(f"Analysis completed for video: {video_data.get('title', 'Unknown Title')}")

                else:
                    logging.warning(f"No transcript for video: {video_data.get('title', 'Unknown Title')}\n")

        logging.info("All transcripts have been analyzed.")


    def report(self, subject):
        self.subject_searched = subject
        logging.info(f"Generating report for subject: {subject}\n")
        top_videos = self.get_top_videos(subject)
        self.get_transcripts(top_videos)
        self.analyze_transcripts()
        #relevant_videos = self.apply_heuristic(top_videos, subject)

        self.save_ts_report_to_csv()
                
        self.save_transcripts_to_txt()

        for video_data in self.video_data_profile.values():
            print(f"\n Start of second for loop in report()\n")

            print(f"\nTitle: {video_data['title']}")
            print(f"URL: {video_data['url']}")
            print(f"View Count: {video_data['view_count']}")
            print(f"Upload Date: {video_data['upload_date']}")

        logging.info(f"Report generated for subject: {subject}\n")

    # Saving to File \ Reading From File
    #   Functions
    #      Below ---- thumbnails to text, headlines to text, video_data_profile to .csv

    def save_thumbnails_to_txt(self, directory='thumbnails'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{self.subject_searched}_thumbnails.txt")

        with open(file_path, 'w', encoding='utf-8') as file:
            for video_id, video_data_list in self.video_data_profile.items():
                for video_data in video_data_list:
                    # Extract thumbnail URL only if it exists in the dictionary
                    if 'thumbnail' in video_data:
                        thumbnail_url = video_data['thumbnail']
                        file.write(str(video_id) + ": " + thumbnail_url + "\n")

        print("\n Video Thumbnails Saved to .txt \n")

    def save_video_data_to_csv(self, filename="video_data_profile.csv"):
        """
        Save the video data profile to a CSV file.
        The first column is the video ID, the second is the subject searched,
        and the remaining columns are from the video data profile.
        """

        # Creating a list to store CSV rows
        csv_data = []

        # Define headers for the CSV file
        headers = []

        # Check if there are keys to add to headers from the first video data (if available)
        first_video_id = next(iter(self.video_data_profile), None)
        if first_video_id:
            additional_headers = list(self.video_data_profile[first_video_id][0].keys())
            headers.extend(additional_headers)

        # Iterate through each video data
        for video_id, video_data_list in self.video_data_profile.items():
            for video_data in video_data_list:
                row.update(video_data)
                csv_data.append(row)

        # Write data to CSV file
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

        print(f"Data saved to {filename}")

    def load_video_data_from_csv(self, filename):
        """
        Loads video data from a CSV file and populates self.video_data_profile.

        Parameters:
        filename (str): The path to the CSV file to be read.
        """
        # Reset or initialize the video_data_profile
        self.video_data_profile = {}

        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                video_id = row.get('video_id')
                if video_id:
                    # Remove the 'Video ID' key from the row as it's used as the dictionary key
                    del row['video_id']
                    
                    if video_id in self.video_data_profile:
                        self.video_data_profile[video_id].append(row)
                    else:
                        self.video_data_profile[video_id] = [row]

        print(f"Data loaded from {filename}")

    def save_headlines_to_txt(self, directory='headlines'):
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{self.subject_searched}_headlines.txt")

        with open(file_path, 'w', encoding='utf-8') as file:
            counter = 0
            for video_id, video_data_list in self.video_data_profile.items():
                for video_data in video_data_list:
                    # Extract title only if it exists in the dictionary
                    if 'title' in video_data:
                        headline = video_data['title']
                        file.write(str(video_id) + ": " + headline + "\n")
                        counter += 1
                        break  # Assuming we only want the first title per video ID

        print(f"\n {counter} Videos Headlines Saved to {self.subject_searched}_headlines.txt \n")

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

        for video_id, video_data_list in self.video_data_profile.items():
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



    def apply_heuristic(self, videos, subject): #heurestic to identify/check how relevant video is to search word
        relevant_videos = []
        for video in videos:
            transcript = video.get('transcript', '')
            title = video.get('title', '')
            description = video.get('description', '')
            tags = video.get('tags', [])
            
            if "yes" in self.ask_gpt4("Based on this set of keywords "+ subject +" - Transcript: " +transcript+ "Title: " +title+ "Description: " +description + "Respond yes or no."):
                relevant_videos.append(video)

        return relevant_videos
    


# Example Usage

yt_func = YTFunc()
topics = [
    'Vlogs (Personal Blogs)',
    'Gaming',
    'Educational Content',
    'Technology Reviews',
    'Cooking and Recipes',
    'Fitness and Health',
    'Travel Guides',
    'Beauty and Fashion',
    'Product Reviews',
    'Comedy and Skits',
    'Music Covers and Originals',
    'DIY Projects and Crafts',
    'Unboxing Videos',
    'Movie and TV Reviews',
    'Animation and Short Films',
    'Podcasts and Interviews',
    'Parenting and Family Content',
    'Motivational and Inspirational Content',
    'Science and Technology Experiments',
    'Book Reviews and Literature Discussions',
    'Pet and Animal Videos',
    'Art and Drawing Tutorials',
    'News and Current Events',
    'Cultural and Historical Content',
    'ASMR Videos'
]
topics_expanded = {
    'Vlogs (Personal Blogs)': [
        'Day in the Life', 'Travel Vlogs', 'Q&A Sessions', 'Challenge Videos', 'Life Updates'
    ],
    'Gaming': [
        "Let's Play Videos", 'Game Reviews', 'Speedruns', 'eSports Tournaments', 'Gaming News'
    ],
    'Educational Content': [
        'Language Learning', 'Science Tutorials', 'Math Lessons', 'History Documentaries', 'Coding Tutorials'
    ],
    'Technology Reviews': [
        'Smartphone Reviews', 'Laptop Unboxings', 'Software Tutorials', 'Gadget Comparisons', 'Tech News Updates'
    ],
    'Cooking and Recipes': [
        'Baking Desserts', 'Vegan Cooking', 'Ethnic Cuisine', 'Meal Prep Guides', 'Cooking for Beginners'
    ],
    'Fitness and Health': [
        'Home Workouts', 'Yoga Sessions', 'Nutrition Tips', 'Mental Health Advice', 'Fitness Challenges'
    ],
    'Travel Guides': [
        'City Tours', 'Travel Tips', 'Budget Travel', 'Adventure Travel', 'Cultural Experiences'
    ],
    'Beauty and Fashion': [
        'Makeup Tutorials', 'Fashion Hauls', 'Skincare Routines', 'Hair Styling Tips', 'Seasonal Fashion Trends'
    ],
    'Product Reviews': [
        'Gadget Reviews', 'Beauty Product Reviews', 'Home Appliance Reviews', 'Toy Unboxings', 'Book Reviews'
    ],
    'Comedy and Skits': [
        'Stand-up Comedy', 'Parody Videos', 'Prank Videos', 'Comedy Sketches', 'Satirical News'
    ],
    'Music Covers and Originals': [
        'Cover Songs', 'Original Music Videos', 'Live Performances', 'Music Tutorials', 'Album Reviews'
    ],
    'DIY Projects and Crafts': [
        'Home Decor DIY', 'Scrapbooking', 'Upcycling Projects', 'Knitting and Crochet', 'Paper Crafts'
    ],
    'Unboxing Videos': [
        'Tech Unboxings', 'Toy Unboxings', 'Beauty Box Unboxings', 'Mystery Box Openings', 'Subscription Box Reviews'
    ],
    'Movie and TV Reviews': [
        'Film Analysis', 'TV Show Recaps', 'Actor Interviews', 'Behind-the-Scenes', 'Genre-Specific Reviews'
    ],
    'Animation and Short Films': [
        'Animated Short Stories', 'Character Design Tutorials', 'Animation Software Tutorials', 'Fan-Made Animations', 'Stop Motion Films'
    ],
    'Podcasts and Interviews': [
        'Celebrity Interviews', 'Expert Discussions', 'Political Podcasts', 'True Crime Podcasts', 'Comedy Podcasts'
    ],
    'Parenting and Family Content': [
        'Parenting Tips', 'Family Vlogs', 'Educational Activities for Kids', 'Pregnancy Journals', 'Parent-Child Cooking'
    ],
    'Motivational and Inspirational Content': [
        'Life Coaching', 'Success Stories', 'Motivational Speeches', 'Self-Improvement Tips', 'Inspirational Interviews'
    ],
    'Science and Technology Experiments': [
        'Home Science Experiments', 'Robotics Tutorials', 'Astronomy and Space', 'Ecology and Environment', 'Tech DIY Projects'
    ],
    'Book Reviews and Literature Discussions': [
        'Literary Criticism', 'Author Interviews', 'Book Club Discussions', 'Genre-Specific Reviews', 'Writing Tips'
    ],
    'Pet and Animal Videos': [
        'Pet Care Tips', 'Wildlife Documentaries', 'Funny Pet Videos', 'Animal Rescue Stories', 'Exotic Animal Facts'
    ],
    'Art and Drawing Tutorials': [
        'Digital Art Tutorials', 'Watercolor Painting', 'Sketching Techniques', 'Art Supply Reviews', 'Art History Lessons'
    ],
    'News and Current Events': [
        'Political Analysis', 'Global News Reports', 'Economic News', 'Social Issues Discussions', 'Technology News'
    ],
    'Cultural and Historical Content': [
        'Historical Documentaries', 'Cultural Heritage', 'Travel and Culture', 'Historical Reenactments', 'Cultural Analysis'
    ],
    'ASMR Videos': [
        'Whispering ASMR', 'Eating Sounds', 'Roleplay ASMR', 'Soundscapes', 'Tapping and Scratching'
    ]
}

#for topics in topics_expanded.values():
    #for topic in topics:
        #yt_func.get_top_videos(topics)
for topic in topics:
    yt_func.get_top_videos(topic)
yt_func.save_video_data_to_csv()
#yt_func.save_headlines_to_txt()

#yt_func.get_transcript_from_audio('https://www.youtube.com/watch?v=ORbseYAkzRM', )
#Figure out how to pass parameters to test whisper audio 