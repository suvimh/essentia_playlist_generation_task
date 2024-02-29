import numpy as np
import os
import json
import tqdm
from essentia.standard import AudioLoader, MonoMixer, Resample, RhythmExtractor2013, KeyExtractor, LoudnessEBUR128, TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN, TensorflowPredict2D

# !! CHANGE FILE PATH to point to the audio collection location
COLLECTION_FILE_PATH = "musAv"

#location where output json file is saved, default is same location as script location
JSON_FILE_PATH = 'data/collection_analysis_output.json'

#file to store the filepaths of analysed files, if an error occurs
ANALYZED_FILES_PATH = 'data/analyzed_files.txt'
'''
Designing your script, keep in mind that you should be able to run this script on any given 
music collection of any size with any nested folder structure (for example, someone could 
rerun it on their personal music collection). The script should find and analyze all audio 
files located inside a given folder. We should be able to re-run your script by easily 
changing the audio collection path.

It is up to you to decide the format and file structure to store the analysis results 
for the music collection.

Note that you might encounter analysis errors on some files (unlikely but happens when 
running analysis on a very large amount of audio tracks) that would end up with an 
Essentia exception. Your script should be able to skip such errors if they happen and 
recover analysis without recomputing the tracks that have already been analyzed if it 
was interrupted.

It is a nice idea to add a progress bar, for example using tqdm.

'''

def load_audio(filepath):
    '''
    Load audio from file    
    PARAMETERS: path to file.
    RETURN: stereo audio in 44.1kHz and mono audio in 16kHz
    '''
    stereo_audio, sr, num_channels, __, __, __  = AudioLoader(filename=filepath)()

    mono_audio = None

    if num_channels == 2:
        mono_audio = MonoMixer()(stereo_audio, num_channels)
        mono_audio = Resample(inputSampleRate=sr, outputSampleRate=16000)(mono_audio)

    return (stereo_audio, mono_audio)


def analyze_tempo(stereo_audio):
    '''
    Tempo (BPM) extractor using  RhythmExtractor2013.
    PARAMETERS: stereo audio in 44100 Hz sample rate
    RETURN: Tempo in BPM.
    '''
    mono_audio = MonoMixer()(stereo_audio, 2)
    bpm_re2013, _, _, _, _ = RhythmExtractor2013()(mono_audio)

    return bpm_re2013


def analyze_key(mono_audio):
    '''
    Key extractor (`temperley`, `krumhansl`, `edma`) using 
    KeyExtractor algorithm. Limited to major/minor scales.
    PARAMETERS: mono audio with 16kHz sample rate.
    RETURN: Keys estimated by different key estimation profiles.
    '''
    key_temperley, scale_temperley , _ = KeyExtractor(profileType='temperley')(mono_audio)
    key_krumhansl, scale_krumhansl , _ = KeyExtractor(profileType='krumhansl')(mono_audio)
    key_edma, scale_edma , _ = KeyExtractor(profileType='edma')(mono_audio)

    keys = [[key_temperley, scale_temperley], [key_krumhansl, scale_krumhansl],
            [key_edma, scale_edma]]
    return keys


def analyze_loudness(stereo_audio):
    '''
    Loudness calculation using LoudnessEBUR128 to compute 
    PARAMETERS: stedeo audio
    RETURN: Integrated loudness in LUFS.
    '''
    _, _, integrated_loudness, _ = LoudnessEBUR128()(stereo_audio)
    return integrated_loudness


def analyze_embeddings(mono_audio):
    '''
    PARAMETERS: mono audio file.
    RETURN: Embeddings from Discogs-Effnet for the file.
    '''
    discogs_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings_discogs = discogs_model(mono_audio)

    cnn_model = TensorflowPredictMusiCNN(graphFilename='models/msd-musicnn-1.pb',
                                         output="model/dense/BiasAdd")
    embeddings_musicnn = cnn_model(mono_audio)

    return embeddings_discogs, embeddings_musicnn


def analyze_music_styles(embeddings):
    '''
    Music styles estimation based on Discogs-Effnet
    PARAMETERS: discogs-effnet embeddings.
    RETURN: average activation for each genre.
    '''
    model = TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb",
                                input="serving_default_model_Placeholder",
                                output="PartitionedCall:0")
    music_styles_predict = np.array(model(embeddings))
    music_styles_av = np.mean(music_styles_predict, axis=0)

    return music_styles_av


def analyze_voice_instrumental(embeddings):
    '''
    Voice/instrumental classifier based on discogs-effnet.
    PARAMETERS: discogs-effnet embeddings.
    RETURN: average scores for voice_instrumental
            [instrumental, voice]
    '''
    model = TensorflowPredict2D(graphFilename="models/voice_instrumental-discogs-effnet-1.pb",
                                output="model/Softmax")
    voice_instrumental_predictions = np.array(model(embeddings))
    voice_instrumental_av = np.mean(voice_instrumental_predictions, axis=0)

    return voice_instrumental_av


def analyze_danceability(embeddings):
    '''
    Danceability classifier, based on discogs-effnet.
    PARAMETERS: discogs-effnet embeddings.
    RETURN: average scores for danceability.
            [danceable, not_danceable]
    '''
    model = TensorflowPredict2D(graphFilename="models/danceability-discogs-effnet-1.pb",
                                output="model/Softmax")
    predictions = model(embeddings)
    danceability_predictions = np.array(predictions)
    danceability_av = np.mean(danceability_predictions, axis=0)

    return danceability_av


def analyze_arousal_valence(embeddings):
    '''
    Arousal and valence classifier, based on msd-musicnn.
    PARAMETERS: msd-musicnn embeddings.
    RETURN: average scores for arousal and valence
            [valence, arousal] - range [1, 9]
    '''
    model = TensorflowPredict2D(graphFilename="models/muse-msd-musicnn-2.pb",
                                output="model/Identity")
    arousal_valence_predictions = np.array(model(embeddings))
    arousal_valence_av = np.mean(arousal_valence_predictions, axis=0)

    return arousal_valence_av


def analyze_collection(collection_paths):
    '''
    Get audio features for each file of an audio collection.
    PARAMETERS: collection of audio files.
    RETURN: dict containing all audio features computed for each file
    '''
    collection_analysis = {}
    analyzed_files = set()
    if os.path.exists(ANALYZED_FILES_PATH):
        with open(ANALYZED_FILES_PATH, 'r') as file:
            analyzed_files = set(file.read().splitlines())

    try:
        for filepath in tqdm.tqdm(collection_paths, desc="Processing Files", unit="file"):
            if filepath in analyzed_files:
                continue

            try:
                stereo_audio, mono_audio = load_audio(filepath)
                bpm_re2013 = analyze_tempo(stereo_audio)
                temperley, krumhansl, edma = analyze_key(mono_audio)
                key_profiles = {
                    'temperley': temperley,
                    'krumhansl': krumhansl,
                    'edma': edma
                }

                loudness = analyze_loudness(stereo_audio)

                embeddings_discogs, embeddings_musicnn = analyze_embeddings(mono_audio)
                music_styles = analyze_music_styles(embeddings=embeddings_discogs)
                voice_instrumental = analyze_voice_instrumental(embeddings_discogs)
                danceability = analyze_danceability(embeddings_discogs)
                arousal_valence = analyze_arousal_valence(embeddings_musicnn)

                file_analysis = {
                    'bpm': bpm_re2013,
                    'key_profiles': key_profiles,
                    'loudness': loudness,
                    'music_styles': music_styles,
                    'voice_instrumental': voice_instrumental,
                    'danceability': danceability,
                    'arousal_valence': arousal_valence
                }

                collection_analysis[filepath] = file_analysis
                analyzed_files.add(filepath)

            except Exception as e:
                print(f"Error analyzing file '{filepath}': {e}")

    finally:
        with open(ANALYZED_FILES_PATH, 'w') as file:
            file.write('\n'.join(analyzed_files))

    return collection_analysis

def list_files(folder_path, file_paths):
    '''
    Recursive function to create a list of all filepaths for the dataset files.
    Iterates through all folders within the given path until finds file paths.
    PARAMETERS: folder to musAV data
                filepaths list
    '''
    items = os.listdir(folder_path)

    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            list_files(item_path, file_paths)
        else:
            file_paths.append(item_path)


def convert_numpy_arrays_to_lists(obj):
    '''
    Function to make sure data is JSON serializable.
    '''
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_arrays_to_lists(element) for element in obj]
    else:
        return obj

def run_analysis():
    '''
    Function that is run when calling the script. 
    '''
    #get file paths
    collection_file_paths = []
    print('Collecting filepaths.')
    list_files(COLLECTION_FILE_PATH, collection_file_paths)

    #analyse audio features
    print('Starting analysis of files...')
    collection_analysis = analyze_collection(collection_file_paths)
    print('Audio feature extraction complete.')
    #convert np arrays to lists for json output
    print('Saving to JSON...')
    data_out = convert_numpy_arrays_to_lists(collection_analysis)

    #save audio features dict to json
    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump(data_out, json_file)

    print('PROCESS COMPLETE')


if __name__ == "__main__":
    run_analysis()