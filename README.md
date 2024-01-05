# yt_func
This python program was made to gather data from YouTube using the YouTubeData v3 API, analyze that data with the Open AI API and congregate all data into a single file for a OpenAI's custom "GPTs".

And it does alll of that, plus a few charting functions for visualizing data. 

To run the code you'll have to have an Google API Key and Open AI API Key.

All of the code was generated using gpt-4. Curated and assembled and debugged by me, also with assistance from GPT-4.

example usage

yt = yt_func()

#topics to search videos for
topics = [dogs, rare dogs, hunting dogs]

#get the top videos for each topic (10 videos by default)

for topic in topics:
  yt.get_top_videos(topic)

#save data to file

yt.save_video_data_to_csv("dogs.csv")

After it is saved if you want to harvest data over time you can this function

yt.update_and_save("dogs_2.0.csv", "dogs.csv")

This will add 10 more videos using the same exact subjects from the earlier run.




