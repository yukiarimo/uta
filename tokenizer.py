import json

# Create a list of numbers from 0 to 8192
numbers = list(range(8193))

# Create a list of characters including all English letters, spaces, commas, periods, and apostrophes
characters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.'?!-_…:")

# Add special tokens
special_tokens = ["<pad>", "<text>", "<audio>"]

# Combine the lists
tokens = numbers + characters + special_tokens

# Create a mapping from characters to integers
stoi = {str(token): idx for idx, token in enumerate(tokens)}

# Create a mapping from integers to characters
itos = {idx: str(token) for idx, token in enumerate(tokens)}

# Combine both mappings into a single dictionary
tokenizer_mapping = {
    "stoi": stoi,
    "itos": itos
}

# Save the mapping to a JSON file
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_mapping, f, ensure_ascii=False, indent=4)

print("Tokenizer mapping has been saved to tokenizer_mapping.json")