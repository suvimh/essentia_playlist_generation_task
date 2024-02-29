import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import seaborn as sns

AUDIO_FEATURES_JSON_FILE_PATH = 'data/collection_analysis_output.json'

def save_tsv_file(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Genre', 'Occurrences'])
        for genre, occurrences in data.items():
            writer.writerow([genre, occurrences])


def plot_music_styles_distribution(music_styles_dict):
    ''' 
    Plot parent broad genre distribution. Provide full results on genre distribution as a separate TSV file.
    PARAMETERS: dictionary of genre activation values for all files of the dataset.

    As the model predictions are activations, you need to decide what you consider as the final music style prediction for a track. 
    If you want to consider the possibility of multiple styles per track, define some threshold for activation values. Otherwise, 
    if you want to have a single music style per track, use the one with the maximum activation.
    We have 400 values which may be a challenge to fit in a compact plot. Predicted styles have a parent broad genre category 
    (all style tags have a format `genre—style`). Therefore you can instead report distribution for parent broad genres. In any case, 
    also  (similar to how we report genre distribution in MTG-Jamendo).
    '''

    full_top_genre_occurrences = {}
    main_top_genre_occurrences = {}
    top3_full_genre_occurrences = {}
    top3_main_genre_occurrences = {}

    # get genres from the json so can match with index of activation values
    with open('models/genre_discogs400-discogs-effnet-1.json', 'r') as file:
        genre_data = json.load(file)
    classes_list = genre_data.get('classes', [])

    for file in music_styles_dict:
        # option1: top music style for each file
        index_of_max_activation = np.argmax(music_styles_dict[file])
        try:
            full_matched_genre = classes_list[index_of_max_activation]
            main_genre = full_matched_genre.split('---')[0]
        except IndexError as e:
            print(f"IndexError searching top genre: {e}")

        full_top_genre_occurrences[full_matched_genre] = full_top_genre_occurrences.get(full_matched_genre, 0) + 1
        main_top_genre_occurrences[main_genre] = main_top_genre_occurrences.get(main_genre, 0) + 1

        # option2: top 3 music styles for each file
        try:
            indices_of_top_3_activations = np.argsort(music_styles_dict[file])[-3:][::-1]
            top_3_matched_genres = [classes_list[index] for index in indices_of_top_3_activations]

            for top_genre in top_3_matched_genres:
                main_genre_top3 = top_genre.split('---')[0]
                top3_full_genre_occurrences[top_genre] = top3_full_genre_occurrences.get(top_genre, 0) + 1
                top3_main_genre_occurrences[main_genre_top3] = top3_main_genre_occurrences.get(main_genre_top3, 0) + 1
        except IndexError as e:
            print(f"IndexError searching top 3 genres: {e}")

    # save top genre occurrences to separate TSV files
    save_tsv_file(full_top_genre_occurrences, 'data/full_genre_occurrences.tsv')

    pastel_colors = [sns.color_palette("pastel", len(main_top_genre_occurrences))[i] for i in range(len(main_top_genre_occurrences))]

    # option1: plot top music styles
    plt.figure(figsize=(12, 6))
    plt.bar(main_top_genre_occurrences.keys(), main_top_genre_occurrences.values(), color=pastel_colors)
    plt.title('Top Genre Distribution')
    plt.xlabel('Main Genre')
    plt.ylabel('Occurrences')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig('graphs/main_genre_distribution.png', bbox_inches='tight')
    plt.show()

    # option2: plot top 3 music styles for each
    plt.figure(figsize=(12, 6))
    plt.bar(top3_main_genre_occurrences.keys(), top3_main_genre_occurrences.values(), color=pastel_colors)
    plt.title('Genre Distribution (with main 3 genres per file)')
    plt.xlabel('Main Genre')
    plt.ylabel('Occurrences')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig('graphs/3genres_distribution.png', bbox_inches='tight')
    plt.show()

def plot_tempo_distribution(bpm_dict):
    '''
    Plot tempo distribution in 10 BPM bins.
    PARAMETERS: dict containing bpm values for all files 
    '''
    # bpm bins with 10 BPM step to make clearer graph
    bin_edges = np.arange(0, max(bpm_dict.values()) + 10, 10)
    bpm_occurrences, bin_edges = np.histogram(list(bpm_dict.values()), bins=bin_edges)

    plt.figure(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    pastel_colors = [sns.color_palette("pastel", len(bpm_occurrences))[i] for i in range(len(bpm_occurrences))]

    plt.bar(bin_centers, bpm_occurrences, width=8, color=pastel_colors, align='center')  
    plt.title('Tempo (BPM) Distribution in 10 BPM Bins')
    plt.xlabel('Tempo Bin Range (BPM)')
    plt.ylabel('Occurrences')

    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    plt.xticks(bin_centers, bin_labels, rotation=45, ha="right")

    plt.savefig('graphs/tempo_distribution.png', bbox_inches='tight')
    plt.show()


def plot_danceability_distribution(danceability_dict):
    '''
    Plot danceability distribution.
    PARAMETERS: dict containing danceability values for all files
    '''
    #option 1: danceable/not labelling
    total_files = len(danceability_dict)
    danceable_percentage = sum(1 for percentages in danceability_dict.values() if percentages[0] > percentages[1]) / total_files * 100
    not_danceable_percentage = 100 - danceable_percentage

    danceability_percentages = {'Danceable': danceable_percentage, 'Not Danceable': not_danceable_percentage}

    plt.figure()
    keys = list(danceability_percentages.keys())
    values = list(danceability_percentages.values())

    plt.bar(keys, values, color=['lightgreen', 'lightsalmon'])
    plt.title('Danceability Distribution (binary label for each file)')
    plt.ylabel('Percentage')
    plt.savefig('graphs/danceability_distribution.png', bbox_inches='tight')
    plt.show()

    # #option 2: overall danceability distribution
    # overall_danceable_percentage = 0
    # overall_not_danceable_percentage = 0

    # total_files = len(danceability_dict)

    # for percentages in danceability_dict.values():
    #     overall_danceable_percentage += percentages[0] * 100
    #     overall_not_danceable_percentage += percentages[1] * 100

    # overall_danceable_percentage /= total_files
    # overall_not_danceable_percentage /= total_files

    # plt.figure(figsize=(6, 4))
    # plt.bar(['Danceable', 'Not Danceable'], [overall_danceable_percentage, overall_not_danceable_percentage], color=['lightgreen', 'lightcoral'])
    # plt.title('Danceability Distribution (overall prediction values)')
    # plt.ylabel('Percentage')
    # plt.show()


def plot_key_scale_distribution(key_profiles_dict):
    '''
    Plot distribution according to three profiles (temperley, krumhansl, edma).
    PARAMETERS: dict containing key profile values for all files 
    '''
    occurrences = {'temperley': {}, 'krumhansl': {}, 'edma': {}}
    key_agreed_count = 0  
    
    for file in key_profiles_dict:
        temperley = key_profiles_dict[file]['temperley']
        krumhansl = key_profiles_dict[file]['krumhansl']
        edma = key_profiles_dict[file]['edma']

        temperley_key = ' '.join(temperley)
        krumhansl_key = ' '.join(krumhansl)
        edma_key = ' '.join(edma)

        occurrences['temperley'][temperley_key] = occurrences['temperley'].get(temperley_key, 0) + 1
        occurrences['krumhansl'][krumhansl_key] = occurrences['krumhansl'].get(krumhansl_key, 0) + 1
        occurrences['edma'][edma_key] = occurrences['edma'].get(edma_key, 0) + 1
        
        if temperley_key == krumhansl_key == edma_key:
            key_agreed_count += 1
    
    total_tracks = len(key_profiles_dict)
    agreement_percentage = (key_agreed_count / total_tracks) * 100
    
    # sort keys for consistent order
    all_keys = sorted(set(occurrences['temperley'].keys()) | set(occurrences['krumhansl'].keys()) | set(occurrences['edma'].keys()))

    plt.figure(figsize=(10, 15))
    for i, profile in enumerate(['temperley', 'krumhansl', 'edma']):
        keys = all_keys
        values = [occurrences[profile].get(key, 0) for key in all_keys]

        pastel_colors = [sns.color_palette("pastel", len(all_keys))[i] for i in range(len(all_keys))]

        plt.subplot(3, 1, i+1)
        plt.bar(keys, values, color=pastel_colors)
        plt.title(f'{profile.capitalize()} Profile')
        plt.xlabel('Key-scale')
        plt.ylabel('Occurrences')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('graphs/key_scale_distribution.png', bbox_inches='tight')
    plt.show()

    # plot how much the profiles agreed on the key
    plt.figure(figsize=(6, 6))
    plt.bar(['Agreed on key', 'Disagreed on key'], [agreement_percentage, 100 - agreement_percentage], color=['lightgreen', 'lightsalmon'])
    plt.title('Agreement vs Disagreement')
    plt.xlabel('Agreement Status')
    plt.ylabel('Percentage')
    plt.savefig('graphs/agreement_vs_disagreement.png', bbox_inches='tight')
    plt.show()

def plot_loudness_distribution(loudness_dict):
    '''
    Plot integrated loudness LUFS distribution.
    PARAMETERS: dict containing loudness values in LUFS for all files 
    '''
    loudness_occurrences = {}

    for _, loudness in loudness_dict.items():
        loudness_key = str(round(loudness))
        loudness_occurrences[loudness_key] = loudness_occurrences.get(loudness_key, 0) + 1
    
    sorted_keys = sorted(loudness_occurrences.keys(), key=lambda x: float(x))
    pastel_colors = [sns.color_palette("pastel", len(loudness_occurrences))[i] for i in range(len(loudness_occurrences))]

    plt.figure()
    keys = sorted_keys  
    values = [loudness_occurrences[key] for key in keys]

    plt.bar(keys, values, color=pastel_colors)
    plt.title('Loudness Distribution')
    plt.xlabel('Loudness (LUFS)')
    plt.ylabel('Occurrences')

    plt.savefig('graphs/LUFS_distribution.png', bbox_inches='tight')
    plt.show()

def plot_arousal_valence_distribution(arousal_valence_dict):
    '''
    Plot a 2D distribution of arousal/valence emotion space.
    PARAMETERS: dict containing arousal/valence values for all files 
    '''
    arousal_values = []
    valence_values = []

    for _, values in arousal_valence_dict.items():
        arousal_values.append(values[1])
        valence_values.append(values[0])

    pastel_colors = sns.color_palette("pastel", len(arousal_valence_dict))

    plt.figure()
    plt.scatter(arousal_values, valence_values, color=pastel_colors, alpha=0.5)
    plt.title('Arousal/Valence Distribution')
    plt.xlabel('Arousal')
    plt.ylabel('Valence')
    plt.grid(True)
    plt.savefig('graphs/arousal_valence_distribution.png', bbox_inches='tight')
    plt.show()

def plot_vocal_instrumental_distribution(voice_ins_dict):
    '''
    Plot vocal-instrumental distribution.
    PARAMETERS: dict containing voice-instrumental values for all files  
    '''
    #option 1: instrumental/vocal labelling
    binary_voice_ins_occurrences = {'Instrumental': 0, 'Voice': 0}
    total_occurrences = len(voice_ins_dict)

    for _, percentages in voice_ins_dict.items():
        if percentages[0] > percentages[1]:
            binary_voice_ins_occurrences['Instrumental'] += 1
        else:
            binary_voice_ins_occurrences['Voice'] += 1

    for key in binary_voice_ins_occurrences:
        binary_voice_ins_occurrences[key] = (binary_voice_ins_occurrences[key] / total_occurrences) * 100
    
    plt.figure(figsize=(6, 4))
    keys = list(binary_voice_ins_occurrences.keys())
    values = list(binary_voice_ins_occurrences.values())

    plt.bar(keys, values, color=['lavender', 'thistle'])
    plt.title('Instrumental/Vocal Distribution (binary label for each file)')
    plt.ylabel('Percentage of Occurrences')
    plt.savefig('graphs/voice_instrumental_distribution.png', bbox_inches='tight')
    plt.show()

    #option 2: overall instrumental/vocal distribution
    # overall_ins_percentage = 0
    # overall_vocal_percentage = 0

    # total_files = len(voice_ins_dict)

    # for percentages in voice_ins_dict.values():
    #     overall_ins_percentage += percentages[0] * 100
    #     overall_vocal_percentage += percentages[1] * 100

    # overall_ins_percentage /= total_files
    # overall_vocal_percentage /= total_files

    # plt.figure(figsize=(6, 4))
    # plt.bar(['Instrumental', 'Vocal'], [overall_ins_percentage, overall_vocal_percentage], color=['green', 'lightcoral'])
    # plt.title('Instrumental/Vocal Distribution (overall prediction values)')
    # plt.ylabel('Percentage')
    # plt.show()


def run_distribution_plots():
    '''
    Function that is run when calling the script. 
    '''
    # load audio features dict from json
    with open(AUDIO_FEATURES_JSON_FILE_PATH, 'r') as json_file:
        collection_analysis = json.load(json_file)
    
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

    plot_music_styles_distribution(feature_dicts['music_styles'])
    plot_tempo_distribution(feature_dicts['bpm'])
    plot_danceability_distribution(feature_dicts['danceability'])
    plot_key_scale_distribution(feature_dicts['key_profiles'])
    plot_loudness_distribution(feature_dicts['loudness'])
    plot_arousal_valence_distribution(feature_dicts['arousal_valence'])
    plot_vocal_instrumental_distribution(feature_dicts['voice_instrumental'])


if __name__ == "__main__":
    run_distribution_plots()