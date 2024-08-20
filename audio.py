import librosa
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import pickle
import json
import os

def prepare_audio(file_path, target_sr=44100):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # Resample to target_sr if necessary
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Normalize audio data
    y = y / np.max(np.abs(y))
    
    return y, target_sr

def extract_mfcc(audio_data, sr, n_mfcc=13, hop_length=512):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfccs.T

def create_tokenizer(mfcc_data, n_tokens=2048):
    # Use K-means clustering to create tokens
    kmeans = KMeans(n_clusters=n_tokens, random_state=42)
    kmeans.fit(mfcc_data)
    
    return kmeans

def tokenize_audio(mfcc_data, tokenizer):
    # Predict tokens
    tokens = tokenizer.predict(mfcc_data)
    
    return tokens

def detokenize_audio(tokens, tokenizer, n_mfcc=13, sr=44100, hop_length=512):
    # Get centroids from the tokenizer
    centroids = tokenizer.cluster_centers_
    
    # Map tokens back to MFCC values
    reconstructed_mfcc = centroids[tokens]
    
    # Inverse MFCC to audio
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(reconstructed_mfcc.T, sr=sr, hop_length=hop_length)
    
    return reconstructed_audio

def test_create_tokenizer(mfcc_data, n_tokens):
    tokenizer = create_tokenizer(mfcc_data, n_tokens)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

def test_detokenize_audio(tokens, tokenizer, n_mfcc, sr, hop_length):
    # Detokenize audio
    reconstructed_audio = detokenize_audio(tokens, tokenizer, n_mfcc, sr, hop_length)

    # Save the reconstructed audio
    sf.write('reconstructed_audio.wav', reconstructed_audio, sr)

def test_tokenize_audio(file_path, target_sr, n_mfcc, hop_length, tokenizer):
    # Prepare audio
    audio_data, sr = prepare_audio(file_path, target_sr)

    # Extract MFCC features
    mfcc_data = extract_mfcc(audio_data, sr, n_mfcc, hop_length)

    # Tokenize audio
    tokens = tokenize_audio(mfcc_data, tokenizer)
    return tokens

# Main process
file_path = 'wavs/all-combined.wav'
target_sr = 44100
n_mfcc = 3000
n_tokens = 8192
hop_length = 256

tokens = [5478,5636]

# Test create_tokenizer
#audio_data, sr = prepare_audio(file_path, target_sr)
#mfcc_data = extract_mfcc(audio_data, target_sr, n_mfcc, hop_length)
#test_create_tokenizer(mfcc_data, n_tokens)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Test tokenize_audio
#tokens = test_tokenize_audio(file_path, target_sr, n_mfcc, hop_length, tokenizer)

#print("Tokens amount: ", len(tokens))

# Test detokenize_audio
test_detokenize_audio(tokens, tokenizer, n_mfcc, target_sr, hop_length)

def process_audio_file(file_path, target_sr, n_mfcc, hop_length, tokenizer):
    # Prepare audio
    print(file_path)

    audio_data, sr = prepare_audio(file_path, target_sr)

    # Extract MFCC features
    mfcc_data = extract_mfcc(audio_data, sr, n_mfcc, hop_length)

    # Tokenize audio
    tokens = tokenize_audio(mfcc_data, tokenizer)
    
    return tokens

def process_and_save(input_file, output_file, target_sr, n_mfcc, hop_length, tokenizer):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Skipping processing.")
        return

    with open(input_file, 'r') as f:
        for line in f:
            audio_file, text = line.strip().split('|')
            if not os.path.exists(audio_file):
                print(f"Audio file {audio_file} does not exist. Skipping.")
                continue

            tokens = process_audio_file(audio_file, target_sr, n_mfcc, hop_length, tokenizer)
            result = {
                'text': text,
                'tokens': tokens.tolist()  # Convert numpy array to list for JSON serialization
            }
            
            # Save the current result to the output file
            with open(output_file, 'a') as out_f:
                json.dump(result, out_f)
                out_f.write('\n')  # Write each JSON object on a new line

# Main process
input_file = 'audio_text_pairs.txt'
output_file = 'tokenized_audio.json'

# Process and save the audio files and their corresponding texts
#process_and_save(input_file, output_file, target_sr, n_mfcc, hop_length, tokenizer)