import os.path
import random
import streamlit as st
import pandas as pd
import json
import numpy as np
from collections import defaultdict


m3u_filepaths_file = 'playlists/streamlit.m3u8'
ESSENTIA_ANALYSIS_PATH = 'data/collection_analysis_output.json'
GENRE_JSON_PATH = 'models/genre_discogs400-discogs-effnet-1.json'
TSV_FILE_PATH = 'data/full_genre_occurrences.tsv'

def load_essentia_analysis():
    with open(ESSENTIA_ANALYSIS_PATH, 'r') as file:
        data = json.load(file)
    return data

def get_genre_activation(music_styles_dict):
    #create a defaultdict to store ordered lists of (file_name, activation_value) tuples
    genre_activation_dict = defaultdict(list)

    #get genres from the json so can match with index of activation values
    with open(GENRE_JSON_PATH, 'r') as file:
        genre_data = json.load(file)
    classes_list = genre_data.get('classes', [])

    for file_name, activation_values in music_styles_dict.items():
        for style_index, activation_value in enumerate(activation_values):
            try:
                full_matched_genre = classes_list[style_index]
                genre_style_key = f"{full_matched_genre}"
                genre_activation_dict[genre_style_key].append((file_name, activation_value))

            except IndexError as e:
                print(f"IndexError searching genre-style: {e}")

    for key, value_list in genre_activation_dict.items():
        genre_activation_dict[key] = sorted(value_list, key=lambda x: x[1], reverse=True)

    return dict(genre_activation_dict)

def get_tempo_dict(bpm_data):
    #get data in tempo-files form 
    tempo_dict = {}
    for file_name, bpm_value in bpm_data.items():
        rounded_tempo = round(bpm_value)
        if rounded_tempo not in tempo_dict:
            tempo_dict[rounded_tempo] = [file_name]
        else:
            tempo_dict[rounded_tempo].append(file_name)
    return tempo_dict

def get_voice_instrumental_dict(vi_data):
    #get data in voice - files and 
    #instrumental - files form
    vi_dict = {
                'Instrumental': [],
                'Voice' : []
               }
    for file_name, vi_values in vi_data.items():
        if vi_values[0] > vi_values[1]:
            vi_dict['Instrumental'].append(file_name)
        if vi_values[1] > vi_values[0]:
            vi_dict['Voice'].append(file_name)
    return vi_dict

def get_danceability_dict(dance_data):
    #get data in voice - files
    #and instrumental - files form
    dance_dict = {}
    for file_name, dance_values in dance_data.items():
        dance_value = round(dance_values[0], 3)
        if dance_value not in dance_dict:
            dance_dict[dance_value] = [file_name]
        else:
            dance_dict[dance_value].append(file_name)
    dance_dict = dict(sorted(dance_dict.items()))
    return dance_dict

def get_arousal_valence_dict(av_data):
    #get data in arousal values - files
    #and valence values - files forms
    a_dict = {}
    v_dict = {}
    for file_name, av_values in av_data.items():
        a_value = round(av_values[1], 3)
        v_value = round(av_values[0], 3)
        if a_value not in a_dict:
            a_dict[a_value] = [file_name]
        else:
            a_dict[a_value].append(file_name)

        if v_value not in v_dict:
            v_dict[v_value] = [file_name]
        else:
            v_dict[v_value].append(file_name)

    a_dict = dict(sorted(a_dict.items()))
    v_dict = dict(sorted(v_dict.items()))

    return a_dict, v_dict

def get_key_scale_data(key_scale_profile):
    #get data in key-scale - files
    #only using edma key profile
    key_dict = {}
    for file, key_profile in key_scale_profile.items():
        key =  key_profile['edma'][0]
        scale = key_profile['edma'][1]
        key_scale = f"{key} {scale}"
        if key_scale not in key_dict:
            key_dict[key_scale] = [file]
        else:
            key_dict[key_scale].append(file)
    return key_dict

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

#get all needed data in correct format
style_activation_data = get_genre_activation(feature_dicts['music_styles'])
audio_analysis_styles = sorted(list(style_activation_data.keys()))

TEMPO_DATA = get_tempo_dict(feature_dicts['bpm'])
VOICE_INS_DATA = get_voice_instrumental_dict(feature_dicts['voice_instrumental'])
DANCEABILITY_DATA = get_danceability_dict(feature_dicts['danceability'])
AROUSAL_DATA, VALENCE_DATA = get_arousal_valence_dict(feature_dicts['arousal_valence'])

KEY_DATA = get_key_scale_data(feature_dicts['key_profiles'])
audio_analysis_keys = sorted(list(KEY_DATA.keys()))

# load data from tsv file
genre_distribution_data = pd.read_csv(TSV_FILE_PATH, sep='\t')
genre_distribution_data = genre_distribution_data.sort_values(by='Occurrences', ascending=False)

def filter_by_tempo(tempo_range, tempo_data=TEMPO_DATA):
    min_t, max_t = tempo_range
    filtered_files_list = [(file_name, tempo) for tempo, file_names in tempo_data.items() for file_name in file_names if min_t <= tempo <= max_t]
    return filtered_files_list

def filter_by_danceability(dance_range, dance_data=DANCEABILITY_DATA):
    min_d, max_d = dance_range
    filtered_files_list = [(file_name, danceability) for danceability, file_names in dance_data.items() for file_name in file_names if min_d <= danceability <= max_d]
    return filtered_files_list

def filter_by_arousal_valence(a_range, v_range, a_data=AROUSAL_DATA, v_data=VALENCE_DATA):
    min_a, max_a = a_range
    filtered_files_a = [(file_name, arousal) for arousal, file_names in a_data.items() for file_name in file_names if min_a <= arousal <= max_a]
    min_v, max_v = v_range
    filtered_files_v = [(file_name, valence) for valence, file_names in v_data.items() for file_name in file_names if min_v <= valence <= max_v]
    
    common_files = set([file_name for (file_name, _) in filtered_files_a]) & set([file_name for (file_name, _) in filtered_files_v])

    filtered_files_list = [
        (file_name, arousal, valence)
        for (file_name, arousal) in filtered_files_a
        for (filename_v, valence) in filtered_files_v
        if file_name == filename_v and file_name in common_files
    ]
    return filtered_files_list

def filter_by_instrumental(selection, voice_ins_data=VOICE_INS_DATA):
    if selection == 'voice':
        voice_instrumental_list = voice_ins_data['Voice']
    if selection == 'ins':
        voice_instrumental_list = voice_ins_data['Instrumental']
    return voice_instrumental_list

def filter_by_key_scale(selected_keys, key_data=KEY_DATA):
    key_scale_list = []
    for key in selected_keys:
        if key in key_data:
            key_scale_list.extend(key_data[key])
    return key_scale_list

#display genre distribution information
st.write('# Audio analysis playlists example')
st.write(f'Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.')

st.write('## Genre Distribution Information')
st.write(genre_distribution_data)

st.write('Loaded audio analysis for', len(collection_analysis), 'tracks.')

st.write('## 🔍 Select')
st.write('### By style')

style_select = st.multiselect('Select by style activations:', audio_analysis_styles)
if style_select:
    selected_data = {}

    for genre in style_select:
        values = style_activation_data[genre]
        selected_data[genre] = [float(val[1]) if isinstance(val, tuple) and len(val) == 2 else float(val) if isinstance(val, (int, float)) else None for val in values]

    audio_analysis = pd.DataFrame.from_dict(selected_data, orient='index')

    st.write(audio_analysis.describe())
    style_select_str = ', '.join(style_select)
    style_select_range = st.slider(f'Select tracks with `{style_select_str}` activations within range:', value=[0.0, 1.])

st.write('## 🔝 Rank')
style_rank = st.multiselect('Rank by style activations (multiplies activations for selected styles):', audio_analysis_styles, [])

st.write('## Tempo Range')
min_tempo = min(TEMPO_DATA.keys())
max_tempo = max(TEMPO_DATA.keys())
tempo_range = st.slider('Select tempo range:', min_value=min_tempo, max_value=max_tempo, value=[min_tempo, max_tempo])

st.write('## Select Voice-Instrumental')
instrumental_only = st.checkbox('Show only instrumental songs')
voice_only = st.checkbox('Show only songs with vocals')

st.write('## Danceability Range')
min_dance = min(DANCEABILITY_DATA.keys())
max_dance = max(DANCEABILITY_DATA.keys())
danceability_range = st.slider('Select danceability range:', min_value=min_dance, max_value=max_dance, value=[min_dance, max_dance])

st.write('## Arousal-Valence Range')
max_a = min(AROUSAL_DATA.keys())
min_a = max(AROUSAL_DATA.keys())
max_v = min(VALENCE_DATA.keys())
min_v = max(VALENCE_DATA.keys())
arousal_range = st.slider('Select arousal range:', min_value=min_a, max_value=max_a, value=[min_a, max_a])
valence_range = st.slider('Select valence range:', min_value=min_v, max_value=max_v, value=[min_v, max_v])

st.write('## Select key/scale')
key_select = st.multiselect('Select key and scale:', audio_analysis_keys, [])

st.write('## 🔀 Post-process')
max_tracks = st.number_input('Maximum number of tracks (0 for all):', value=0)
shuffle = st.checkbox('Random shuffle')

if st.button("RUN"):
    st.write('## 🔊 Results')
    genres = list(style_activation_data.keys())
    mp3s = list(collection_analysis.keys())

    if style_select:
        audio_analysis_query = pd.DataFrame({genre: {file_path: activation_value for file_path, activation_value in style_activation_data.get(genre, [])} for genre in style_select})
        for genre in style_select:
            #if no audio files have activation within given range, no files displayed
            audio_analysis_query.loc[(audio_analysis_query[genre] < style_select_range[0]) | (audio_analysis_query[genre] > style_select_range[1]), genre] = np.nan

        audio_analysis_query.dropna(inplace=True)
        st.write('Files that exist within style select in activation range.')
        mp3s = list(audio_analysis_query.index)
        st.write(audio_analysis_query)
        #st.write(mp3s)

    if style_rank:
        audio_analysis_query_rank = pd.DataFrame({genre: {file_path: activation_value for file_path, activation_value in style_activation_data.get(genre, [])} for genre in style_rank})
        audio_analysis_query_rank['RANK'] = audio_analysis_query_rank[style_rank[0]]

        for style in style_rank[1:]:
            audio_analysis_query_rank['RANK'] *= audio_analysis_query_rank[style]

        ranked = audio_analysis_query_rank.loc[mp3s].sort_values(by='RANK', ascending=False)
        ranked = ranked[['RANK'] + style_rank]
        
        mp3s = list(audio_analysis_query_rank.index)
        st.write('Applied ranking by audio style predictions.')
        st.write(ranked)
        #st.write(mp3s)
    
    if key_select:
        st.write('Applied selected keys.')
        key_filtered_files = filter_by_key_scale(key_select)
        mp3s = [mp3 for mp3 in mp3s if mp3 in key_filtered_files]
        st.write(mp3s)
    
    if instrumental_only:
        st.write('Applied selection of only instrumental files.')
        ins_filtered_files = filter_by_instrumental('ins')
        mp3s = [mp3 for mp3 in mp3s if mp3 in ins_filtered_files]
        st.write(mp3s)

    if voice_only:
        st.write('Applied selection of only files with vocals.')
        voice_filtered_files = filter_by_instrumental('voice')
        mp3s = [mp3 for mp3 in mp3s if mp3 in voice_filtered_files]
        st.write(mp3s)

    st.write('Applied tempo range.')
    tempo_filtered_files = filter_by_tempo(tempo_range)
    tempo_df = pd.DataFrame(tempo_filtered_files, columns=['filename', 'tempo']).sort_values(by='tempo', ascending=False)
    tempo_df_filenames = tempo_df['filename'].tolist()
    mp3s = [mp3 for mp3 in mp3s if mp3 in tempo_df_filenames]
    st.write(tempo_df[tempo_df['filename'].isin(mp3s)])

    st.write('Applied arousal-valence range.') 
    av_filtered_files = filter_by_arousal_valence(a_range=arousal_range, v_range=valence_range)
    av_df = pd.DataFrame(av_filtered_files, columns=['filename', 'arousal', 'valence']).sort_values(by='arousal', ascending=False)
    av_df_filenames= av_df['filename'].tolist()
    mp3s = [mp3 for mp3 in mp3s if mp3 in av_df_filenames]
    st.write(av_df[av_df['filename'].isin(mp3s)])

    st.write('Applied danceability range.')
    danceability_filtered_files = filter_by_danceability(danceability_range)
    dance_df = pd.DataFrame(danceability_filtered_files, columns=['filename', 'danceability']).sort_values(by='danceability', ascending=False)
    dance_df_filenames = dance_df['filename'].tolist()
    mp3s = [mp3 for mp3 in mp3s if mp3 in dance_df_filenames]
    st.write(dance_df[dance_df['filename'].isin(mp3s)])

    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')

    if shuffle:
        random.shuffle(mp3s)
        st.write('Applied random shuffle.')

    #store the M3U8 playlist.
    with open(m3u_filepaths_file, 'w') as f:
        #modify relative mp3 paths to make them accessible from the playlist folder.
        mp3_paths = [os.path.join('..', mp3) for mp3 in mp3s]
        f.write('\n'.join(mp3_paths))
        st.write(f'Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.')

    st.write('Audio previews for the first 10 results:')
    for mp3 in mp3s[:10]:
        st.audio(mp3, format="audio/mp3", start_time=0)
