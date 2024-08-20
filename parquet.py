import pandas as pd
import os
from pydub import AudioSegment
import io

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().split('|') for line in file]

def extract_audio_id(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]

def load_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

# Read the text file
data = read_text_file('audio_text_pairs.txt')

# Process the data
processed_data = []
for audio_path, raw_text in data:
    audio_id = extract_audio_id(audio_path)
    audio = load_audio(audio_path)
    
    processed_data.append({
        'audio_id': audio_id,
        'language': 0,
        'audio': audio,
        'raw_text': raw_text,
        'gender': 'female',
        'speaker_id': 4848,
        'is_gold_transcript': True,
        'accent': None
    })

# Create a DataFrame
df = pd.DataFrame(processed_data)

# Save as Parquet
df.to_parquet('yunatts.parquet')