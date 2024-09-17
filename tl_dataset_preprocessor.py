import nltk #for tokenization 
import re #regex for removing symbols
from bs4 import BeautifulSoup #to extract texts on html. doesnt have to remove it if not dealing with html files.
from langdetect import detect, DetectorFactory #language detection

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

# Function to check if a sentence is at least 50% Tagalog
def is_mostly_tagalog(sentence, threshold=0.5):
    words = nltk.word_tokenize(sentence)
    if len(words) == 0:
        return False
    
    tagalog_word_count = sum(1 for word in words if is_tagalog_word(word))
    tagalog_percentage = tagalog_word_count / len(words)
    
    return tagalog_percentage >= threshold

# Function to normalize text
def normalize_text(text):
    print("Normalizing text...")
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
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
        if is_mostly_tagalog(sent, threshold=0.5): #You can Adjust threshold here
            tagalog_sentences.append(sent)
        
        # Calculate and print the progress percentage
        progress = (idx + 1) / total_sentences * 100
        print(f"Progress: {progress:.2f}%", end="\r")

    tagalog_text = ' '.join(tagalog_sentences)
    normalized_text = normalize_text(tagalog_text)
    tokens = normalized_text #you can use nltk to tokenize this if you want
    return tokens

# Example usage
if __name__ == "__main__":
    # Step 1: Read and split the text into batches
    file_path = "output.txt"
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
    with open("preprocessed_tagalog_corpus.txt", 'w', encoding='utf-8') as f:
        f.write(' '.join(all_tokens[:500_000]))


    print(f"Corpus has only {token_count} tokens, consider adding more Tagalog text.")
