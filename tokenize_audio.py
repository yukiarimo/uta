import json

def find_largest_strings_and_token_lengths(jsonl_file):
    largest_strings = []
    max_length = 0

    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # Check if 'text' and 'tokens' keys exist
            if 'text' in data and 'audio' in data:
                text = data['text']
                tokens = data['audio']
                
                if isinstance(text, str) and isinstance(tokens, list):
                    length = len(text)
                    if length > max_length:
                        largest_strings = [(text, len(tokens))]
                        max_length = length
                    elif length == max_length:
                        largest_strings.append((text, len(tokens)))
    
    # Print the largest strings and their token lengths
    for string, token_length in largest_strings:
        print(f"String: {string}")
        print(f"Token Length: {token_length}")

# Replace 'tokenized_audio.jsonl' with the path to your JSONL file
find_largest_strings_and_token_lengths('tokenized_audio.json')