import nltk
import re
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import fasttext

model_path = '/YOUR/PATH/TO/lid.176.bin' #change this depending where is your fastext model
model = fasttext.load_model(model_path)

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Function to read and split text into chunks
def read_and_split_text(file_path, chunk_size=1024*1024):  # Default chunk size: 1MB
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"{file_path} is loaded.")
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
    
    # Remove extra spaces and trim
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
    print(tokens)
    return tokens

if __name__ == "__main__":
    # Step 1: Read and split the text into batches
    file_path = "submissions.txt"
    chunks = read_and_split_text(file_path)
    
    all_tokens = []
    
    # Step 2: Process each batch
    for chunk in chunks:
        print("Processing Batch...")
        tokens = process_batch(chunk)
        all_tokens.extend(tokens)
        print("Batch Processed. Proceeding to the next batch.")
    
    # Step 3: Count the tokens
    token_count = count_tokens(all_tokens)
    print(f"Total tokens: {token_count}")
    
    # Step 4: If the token count is around 500,000, save the corpus
    with open("submission_preproccesed.txt", 'w', encoding='utf-8') as f:
        f.write('/n'.join(all_tokens[:500_000]))


    print(f"Corpus has only {token_count} tokens, consider adding more Tagalog text.")
