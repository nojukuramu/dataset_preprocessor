import os
import nltk
import re
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import fasttext
import json

model_path = 'C:/Users/almae/lid.176.bin'
model = fasttext.load_model(model_path)

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Function to read and split text into chunks
# def read_and_split_text(file_path, chunk_size=1024*1024):  # Default chunk size: 1MB
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read()
    
#     # Split text into chunks
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
#     print(f"{file_path} is loaded.")
#     return chunks

def read_and_split_text(file_path, chunk_size=2048*2048):  # Default chunk size: 2MB Adjust this based on your available memory
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            # You can process each chunk here if needed
            # For example: process_chunk(chunk)

    print(f"{file_path} is loaded in chunks.")
    return chunks


# Function to detect if a word is in Tagalog
def is_tagalog_word(word):
    try:
        lang = detect(word)
        return lang == 'tl'  # 'tl' is the code for Tagalog
    except:
        return False
    
def is_tagalog_word_fasttext(word):
    prediction = model.predict(word, k=1)  # k=1 means we only get the top predicted language
    lang = prediction[0][0].replace("__label__", "")  # Extract the language code
    return lang == 'tl'  # 'tl' is the FastText code for Tagalog

# Function to check if a sentence is at least 50% Tagalog
def is_mostly_tagalog(sentence, threshold=0.5):
    words = nltk.word_tokenize(sentence)
    if len(words) == 0:
        return False
    
    tagalog_word_count = sum(1 for word in words if is_tagalog_word(word))
    tagalog_percentage = tagalog_word_count / len(words)
    
    return tagalog_percentage >= threshold

# Function to check if a sentence is at least 50% Tagalog
def is_mostly_tagalog_fasttext(sentence, threshold=0.5):
    words = nltk.word_tokenize(sentence)
    if len(words) == 0:
        return False

    tagalog_word_count = sum(1 for word in words if is_tagalog_word_fasttext(word))
    tagalog_percentage = tagalog_word_count / len(words)

    return tagalog_percentage >= threshold

def normalize_text(text):
    print("Normalizing text...")
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Convert to lowercase
    text = text.lower()
    # Remove symbols and punctuation, excluding sentence-ending punctuation
    text = re.sub(r'[^\w\s\.\?!]', '', text)  # Keep alphanumeric, spaces, and sentence-ending punctuation
    # Remove specific unwanted patterns
    text = re.sub(r'\[deleted\]', '', text)  # Remove [deleted]
    text = re.sub(r'\[removed\]', '', text)  # Remove [removed]
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove links
    text = re.sub(r'\*|_|-|/|\"', '', text)  # Remove *, _, -, /, and "

    # Replace multiple consecutive newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# Function to count tokens and check the size
def count_tokens(tokens):
    print("Counting tokens...")
    return len(tokens)

# Function to process a batch of text
def process_batch(batch):
    sentences = nltk.sent_tokenize(batch)
    tagalog_sentences = []
    total_sentences = len(sentences)
    for idx, sent in enumerate(sentences):
        if is_mostly_tagalog_fasttext(sent, threshold=0.5): #You can Adjust threshold here
            tagalog_sentences.append(sent)
        
        # Calculate and print the progress percentage
        progress = (idx + 1) / total_sentences * 100
        print(f"Progress: {progress:.2f}%", end="\r")

    tagalog_text = ' '.join(tagalog_sentences)
    normalized_text = normalize_text(tagalog_text)
    tokens = normalized_text #you can use nltk to tokenize this if you want
    return tokens

CHECKPOINT_FILE = 'checkpoint.json'
TOKENS_FILE = 'tokens.json'

def save_checkpoint(index, tokens):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'last_index': index, 'tokens': tokens}, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint['last_index'], checkpoint['tokens']
    return -1, []

if __name__ == "__main__":
    # Step 1: Read and split the text into batches
    file_path = "comments.txt"
    chunks = read_and_split_text(file_path)
    total_chunks = len(chunks)
    all_tokens = []

    # Load checkpoint if available
    last_index, checkpoint_tokens = load_checkpoint()
    all_tokens.extend(checkpoint_tokens)
    
    # Step 2: Process each batch
    
    for i in range(last_index + 1, total_chunks):
        print(f"Processing Batch {i+1} of {total_chunks}...")
        tokens = process_batch(chunks[i])
        all_tokens.extend(tokens)
        save_checkpoint(i, all_tokens)
        print(f"Batch {i+1} Processed. Proceeding to the next batch.")
        progress = (i + 1) / total_chunks * 100
        print(f"Progress: {progress:.2f}%")

    
    # Step 3: Count the tokens
    token_count = count_tokens(all_tokens)
    print(f"Total tokens: {token_count}")
    
    # Step 4: If the token count is around 500,000, save the corpus
    with open("comments_preproccesed.txt", 'w', encoding='utf-8') as f:
        f.write(''.join(all_tokens[:500_000]))


    print(f"Corpus has only {token_count} tokens, consider adding more Tagalog text.")
