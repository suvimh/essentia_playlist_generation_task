This project aimed to develop tools for the analysis and exploration of a music collection, leveraging audio content-based descriptors. We used the MusAV dataset, consisting of 2,092 30-second track previews across 400 genres to do this. The primary tasks included audio analysis using Essentia (Bogdanov, 2013), generating statistical reports on the music collection, and creating two simple user interfaces for playlist generation using Streamlit: one based on various descriptors and another based on track similarity. The generated playlist is saved in a m3u8 file, which can be used to listen to the playlist via e.g. VCL Media Player. A maximum of 10 audio files can be listened to within the UI.


Bogdanov, D., Wack, N., Emilia, .G.G., Gulati, S., Perfecto, .H.B., Mayor, O., Trepat, G.R., Salamon, J., Gonzalez, J.R.Z., Serra, X.: Essentia: an audio analysis library for music information retrieval (2013), http://repositori.upf.edu/handle/10230/322528


Create and activate a Python virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

To run audio feature extraction for collection
(takes a long time, more advisable to use already 
extracted features from data/collection_analysis_output.json):
run audio_analysis_with_essentia_suvi.py

To plot data distribution plots:
```
python plot_distributions_suvi.py
``` 

To run descriptor query based application:
```
streamlit run descriptor_based_app_suvi.py
```

To run similarity based application:
```
streamlit run similarity_based_app_suvi.py
```
