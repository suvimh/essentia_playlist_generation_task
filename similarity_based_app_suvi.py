import os.path
import random
import streamlit as st
import json
import numpy as np


m3u_filepaths_discogs_file = 'playlists/streamlit_discogs.m3u8'
m3u_filepaths_cnn_file = 'playlists/streamlit_cnn.m3u8'
ESSENTIA_ANALYSIS_PATH = 'data/collection_analysis_output.json'

def load_essentia_analysis():
    with open(ESSENTIA_ANALYSIS_PATH, 'r') as file:
        data = json.load(file)
    return data

def get_combined_discogs_dict(styles, vi, dance):
    combined_data = {}
    for filename in set(styles.keys()).intersection(vi.keys()).intersection(dance.keys()):
        combined_data[filename] = styles[filename] + vi[filename] + dance[filename]
    return combined_data


collection_analysis = load_essentia_analysis()
feature_names = ['bpm',
                'key_profiles', 
                'loudness', 
                'music_styles', 
                'voice_instrumental', 
                'danceability', 
                'arousal_valence']
feature_dicts = {feature: 
                {file_name: file_analysis[feature] for file_name, file_analysis in collection_analysis.items()}
                for feature in feature_names}

#get all needed data in correct format, 
#dont need key or tempo info because not based on either embeddings
#music styles, voice-ins and danceability calculated from discogs effnet
DISCOGS_DATA = get_combined_discogs_dict(feature_dicts['music_styles'], feature_dicts['voice_instrumental'], feature_dicts['danceability'])
#arousal_valence calculated from music cnn embeddings, therefore similarity based from that
MUSIC_CNN_DATA = feature_dicts['arousal_valence']

def get_similar_10_files(file_in, data):
    target_values = np.array(data[file_in])
    similarities = {}

    for file, values in data.items():
        if file != file_in:
            similarity = np.dot(target_values, np.array(values))
            similarities[file] = similarity

    mp3s = [file for file in sorted(similarities, key=similarities.get, reverse=True)[:10]]
    return mp3s

#display genre distribution information
st.write('# Audio analysis playlists example')
st.write(f'Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.')

st.write('Loaded audio analysis for', len(collection_analysis), 'tracks.')

st.write('## 🔍 Select a reference song to build similarity playlists')
st.write('### Song ID')
song_select = st.selectbox('Select a song ID:', collection_analysis.keys())

if song_select:
    st.write('Audio preview for the selected file:')
    st.audio(song_select, format="audio/mp3", start_time=0)

shuffle = st.checkbox('Random shuffle')

if st.button("RUN"):
    st.write('## 🔊 Results')
    st.write('### Discogs-effnet based similarity')
    d_mp3s = get_similar_10_files(song_select, DISCOGS_DATA)
    st.write(d_mp3s)

    st.write('### Music CNN based similarity (arousal and valence)')
    cnn_mp3s = get_similar_10_files(song_select, MUSIC_CNN_DATA)
    st.write(cnn_mp3s)  

    if shuffle:
        random.shuffle(cnn_mp3s)
        random.shuffle(d_mp3s)
        st.write('Applied random shuffle to both playlists.')

    #store the M3U8 playlist.
    with open(m3u_filepaths_cnn_file, 'w') as f:
        #modify relative mp3 paths to make them accessible from the playlist folder.
        mp3_paths = [os.path.join('..', mp3) for mp3 in cnn_mp3s]
        f.write('\n'.join(mp3_paths))
        st.write(f'Stored M3U playlist of music CNN playlist (local filepaths) to `{m3u_filepaths_cnn_file}`.')

    with open(m3u_filepaths_discogs_file, 'w') as f:
        #modify relative mp3 paths to make them accessible from the playlist folder.
        mp3_paths = [os.path.join('..', mp3) for mp3 in d_mp3s]
        f.write('\n'.join(mp3_paths))
        st.write(f'Stored M3U playlist of discogs playlist (local filepaths) to `{m3u_filepaths_discogs_file}`.')

    st.write('Audio previews for the Discogs-effnet 10 track similarity playlist:')
    for mp3 in d_mp3s:
        st.audio(mp3, format="audio/mp3", start_time=0)

    st.write('Audio previews for the Music-CNN 10 track similarity playlist:')
    for mp3 in cnn_mp3s:
        st.audio(mp3, format="audio/mp3", start_time=0)
