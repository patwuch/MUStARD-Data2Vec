import os
import pandas as pd
import numpy as np
import glob
import torch
from transformers import AutoTokenizer, AutoProcessor, Data2VecTextModel, Data2VecAudioModel
import librosa
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Data2VecVisionModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''We start off by loading in the original mustard.csv and correcting errors in the entry'''


df = pd.read_csv('local-dir/mustardtext.csv')
df.at[2596,'KEY'] = '1_S11E03_067_u'
df['text_embeddings'] = None
df['audio_embeddings'] = None
df['keyframe_embeddings'] = None


'''We create textembeddings in this section'''

tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    final_layer = outputs.last_hidden_state  # Get the final hidden state (last layer)
    return final_layer

#Creating embeddings for the sarcastic utterance
df['text_embeddings'] = df.apply(lambda row: get_embeddings(row['SENTENCE']) if 'u' in row['KEY'] else None, axis=1)

#Creating a for loop that concatenates the various context sentences
#into one string and then converts it into an embedding

prev_scene = None
concatenated_text = ""
separator_token = " </s> "
for index, row in df.iterrows():
    if row['SCENE'] != prev_scene:
        # If new scene, preprocess and store concatenated text
        if prev_scene is not None:
            df.at[prev_index, 'text_embeddings'] = get_embeddings(concatenated_text)
            concatenated_text = ""
        prev_scene = row['SCENE']
        prev_index = index
    
    if 'c' in row['KEY']:
        concatenated_text += row['SENTENCE'] + separator_token
    #if choosing no separator token, use:
    #   concatenated_text += row['SENTENCE']

# Preprocess and store last scene's text if any
if concatenated_text:
    df.at[prev_index, 'text_embeddings'] = get_embeddings(concatenated_text)

'''In this section we deal with image embeddings'''
# Firstly we need to single out videos that were too short to have keyframes extracted
import cv2
image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
image_model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base")

def find_matching_row(video_path):
    # Extract the filename without the extension
    file_name = os.path.basename(video_path)
    
    modified_file_name = file_name.replace(".mp4", "")
    # Find the boolean series where the modified filename matches the start of the specified column
    matching_rows = df['KEY'].str.startswith(modified_file_name)
    
    # Extract the index of the first matching row
    matching_row_index = matching_rows[matching_rows].index

    if len(matching_row_index) > 0:
        return matching_row_index[0]
    else:
        return None  # Return None if no match is found
    
def video_to_embeddings(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return []
    
    embeddings_list = []

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop if no frame is returned (end of video)
        
        # Convert the frame (BGR) to RGB color format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(frame_rgb)
        
        inputs = image_processor(images=pil_image, return_tensors="pt")

    # Forward pass through the model
        with torch.no_grad():
            outputs = image_model(**inputs)
            embeddings_list.append(outputs.last_hidden_state) 
    
    #   To extract mean of last four representations use 
    #       embeddings_list.append(torch.mean(outputs.last_hidden_state[-4:], dim=1)) 

    # Release the video capture object
    cap.release()
    
    return embeddings_list


# Iterate through all .mp4 files in the folder (utterance first)
folder_path = '/home/chwu/local-dir/final_utterance_videos_1'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.mp4'):
        video_path = os.path.join(folder_path, file_name)
        
        # Convert the video to embeddings
        tensors = video_to_embeddings(video_path)
        
        # Find the matching row in the dataframe
        matching_row_index = find_matching_row(video_path)
        
        if matching_row_index is not None:
            # Store the embeddings in the 'keyframe_embeddings' column
            mean_embeddings = np.mean(np.array(tensors), axis=0)
            mean_embeddings_tensor = torch.tensor(mean_embeddings)
            
            # Store the embeddings in the 'keyframe_embeddings' column
            df.at[matching_row_index, 'keyframe_embeddings'] = mean_embeddings_tensor
            print(f"Processed {file_name}")
            
        else:
            print(f"No matching row found for video: {file_name}")

#Now moving onto context videos

folder_path = '/home/chwu/local-dir/final_context_videos_1'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.mp4'):
        video_path = os.path.join(folder_path, file_name)
        
        # Convert the video to embeddings
        tensors = video_to_embeddings(video_path)
        
        # Find the matching row in the dataframe
        matching_row_index = find_matching_row(video_path)
        
        if matching_row_index is not None:
            # Store the embeddings in the 'keyframe_embeddings' column
            mean_embeddings = np.mean(np.array(tensors), axis=0)
            mean_embeddings_tensor = torch.tensor(mean_embeddings)
            
            # Store the embeddings in the 'keyframe_embeddings' column
            df.at[matching_row_index, 'keyframe_embeddings'] = mean_embeddings_tensor
            print(f"Processed {file_name}")
        else:
            print(f"No matching row found for video: {file_name}")

'''Here we process the pre-extracted keyframes'''

def find_matching_row(file_path):
    # Extract the filename without the extension
    file_name = os.path.basename(file_path)
    
    # Remove "_0.jpeg" from the filename
    modified_file_name = file_name.replace("_0.jpeg", "")
    
    # Find the index of the row where the modified filename matches the specified column
    matching_row_index = df[df['KEY'].str.startswith(modified_file_name)].index
    
    if not matching_row_index.empty:
        return matching_row_index[0]
    else:
        return None

def process_image(img_path):
    # Load the image from file
    image = Image.open(img_path)

    # Process the image using the image processor
    inputs = image_processor(images=image, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = image_model(**inputs)
        representations = outputs.last_hidden_state

    return representations

#STARTING WITH CONTEXT
image_main_folder = 'local-dir/final_all_keyframes/final_context_keyframes'
non_match_img = []
all_image_files = glob.glob(os.path.join(image_main_folder, '*_0.jpeg'))

with tqdm(total=len(all_image_files), desc="Processing Image Files") as pbar:
    for image_file in all_image_files:
        image_location = image_file
        matching_row_index = find_matching_row(image_location)
        
        if matching_row_index is not None:
            embedding_result = process_image(image_location)
            if embedding_result is not None:
                df.at[matching_row_index, 'keyframe_embeddings'] = embedding_result
            else:
                non_match_img.append(image_location)
        else:
            non_match_img.append(image_location)
        
        pbar.update(1)

print(non_match_img)

def find_matching_row(file_path):
    # Extract the filename without the extension
    file_name = os.path.basename(file_path)
    
    # Remove "_0.jpeg" from the filename
    modified_file_name = file_name.replace("_0.jpeg", "")
    
    # Find the index of the row where the modified filename matches the specified column
    matching_row_index = df[df['KEY'].str.startswith(modified_file_name)].index
    
    if not matching_row_index.empty:
        return matching_row_index[0]
    else:
        return None

#MOVING ONTO UTTERANCE
image_main_folder = 'local-dir/final_all_keyframes/final_utterance_keyframes'
non_match_img = []
all_image_files = glob.glob(os.path.join(image_main_folder, '*_0.jpeg'))

with tqdm(total=len(all_image_files), desc="Processing Image Files") as pbar:
    for image_file in all_image_files:
        image_location = image_file
        matching_row_index = find_matching_row(image_location)
        
        if matching_row_index is not None:
            embedding_result = process_image(image_location)
            if embedding_result is not None:
                df.at[matching_row_index, 'keyframe_embeddings'] = embedding_result
            else:
                non_match_img.append(image_location)
        else:
            non_match_img.append(image_location)
        
        pbar.update(1)

print(len(df[df['keyframe_embeddings'].isna()]))

'''Here we deal with audio files'''

audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
audio_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")

# Function to process a single audio file
def process_audio(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=16000)
    inputs = audio_processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        final_features = audio_model(**inputs).last_hidden_state
        # for mean of last four layers, use 
        # final_features = torch.mean(audio_model(**inputs).last_hidden_state[-4:], dim=1)
    return final_features
    
folder_path = "/home/chwu/local-dir/final_context_audios"

import os
import pandas as pd
import numpy as np
import torch

# Assuming 'examine' DataFrame is already defined and loaded
# Initialize the 'audio_embeddings' column if not already present
if 'audio_embeddings' not in df.columns:
    df['audio_embeddings'] = [np.nan] * len(df)

# Ensure the column is of an object dtype to hold tensors
if df['audio_embeddings'].dtype != 'object':
    df['audio_embeddings'] = df['audio_embeddings'].astype('object')


from tqdm import tqdm

# Assuming tqdm is used to wrap the for loop
for file_name in tqdm(os.listdir(folder_path), desc='Processing audio files'):
    file_prefix = os.path.splitext(file_name)[0]  # Get the filename without extension

    file_path = os.path.join(folder_path, file_name)
    
    # Process the audio file to extract features
    features = process_audio(file_path)

    # Convert features to a tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Find the corresponding index in the dataframe
    index_list = df[df['KEY'].str.startswith(file_prefix)].index.tolist()
    if not index_list:
        print(f"No matching index found for prefix context: {file_prefix}")
        continue
    index = index_list[0]
    df.at[index, 'audio_embeddings'] = features_tensor

folder_path = "/home/chwu/local-dir/final_utterance_audios"

for file_name in tqdm(os.listdir(folder_path), desc='Processing audio files'):
    file_prefix = os.path.splitext(file_name)[0]  # Get the filename without extension

    file_path = os.path.join(folder_path, file_name)
    
    # Process the audio file to extract features
    features = process_audio(file_path)

    # Convert features to a tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Find the corresponding index in the dataframe
    index_list = df[df['KEY'].str.startswith(file_prefix)].index.tolist()
    if not index_list:
        print(f"No matching index found for prefix utterance: {file_prefix}")
        continue
    index = index_list[0]
    df.at[index, 'audio_embeddings'] = features_tensor


'''Here we dump everything into a pickle.'''

import pickle

# Specify the file path where you want to save the pickled dataframe
file_path = 'features.pkl'

# Open a file in binary write mode
with open(file_path, 'wb') as f:
    pickle.dump(df, f)

'''This is for creating layered versions of extracted features'''

# audio_processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
# audio_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")

# # Function to process a single audio file
# def process_audio(file_path):
#     # Load audio file
#     audio, sr = librosa.load(file_path, sr=16000)
    
#     # Define the target length (10 seconds)
#     target_length = 10 * sr
    
#     # List to store features for segmented audio
    
#     segment_features=[]
    
#     if len(audio) < target_length:
#         # Pad audio if it is less than 10 seconds
#         padded_audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
#         inputs = audio_processor(padded_audio, sampling_rate=16000, return_tensors="pt")
#         with torch.no_grad():
#             final_features = audio_model(**inputs).last_hidden_state
#         return final_features
#     else:
#         # Segment audio if it is longer than 10 seconds
#         num_segments = int(np.ceil(len(audio) / target_length))
#         for i in range(num_segments):
#             start_idx = i * target_length
#             end_idx = min((i + 1) * target_length, len(audio))
#             segment = audio[start_idx:end_idx]
#             if len(segment) < target_length:
#                 segment = np.pad(segment, (0, target_length - len(segment)), mode='constant')
#             inputs = audio_processor(segment, sampling_rate=16000, return_tensors="pt")
#             with torch.no_grad():
#                 features = audio_model(**inputs).last_hidden_state
#             segment_features.append(features)
    
#         # Average the segment features
#         final_features = np.mean(segment_features, axis=0)
#         return final_features
