import re
import string
import logging
import argparse
from collections import Counter
from hazm import Normalizer, Lemmatizer, word_tokenize
from src.utils import save_vocab

# Setup basic logging for standalone script execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def load_stopwords(file_path):
    """Loads stopwords from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Remove BOM and newline characters
            stop_words = [line.strip('\ufeff').strip() for line in f]
        return set(stop_words)
    except FileNotFoundError:
        logging.error(f"Stopwords file not found: {file_path}")
        return set()

def clean_and_tokenize(text, stopwords_set):
    """Cleans text by removing noise and stopwords."""
    # 1. Remove non-Persian characters (keeping Persian alphabet and spaces)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # 2. Remove digits
    text = re.sub(r'\d+', '', text)
    # 3. Remove extra punctuation (though most is gone from step 1)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Tokenize
    tokens = word_tokenize(text)
    # 5. Remove stopwords and empty tokens
    cleaned_tokens = [word for word in tokens if word not in stopwords_set and word.strip()]
    return cleaned_tokens

def normalize_and_lemmatize(tokens, normalizer, lemmatizer):
    """Applies hazm normalization and lemmatization."""
    normalized_text = normalizer.normalize(' '.join(tokens))
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(normalized_text)]
    # Handle '#' in lemmatized output, e.g., 'رفت#رو' -> 'رفت'
    lemmatized_tokens = [token.split('#')[0] for token in lemmatized_tokens]
    return lemmatized_tokens

def build_vocab(tokens, min_freq=5):
    """Builds a vocabulary from tokens."""
    word_counts = Counter(tokens)
    # Filter words by minimum frequency
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create vocab with special tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(filtered_words):
        vocab[word] = i + 2
        
    logging.info(f"Vocabulary built. Total size: {len(vocab)} (filtered from {len(word_counts)} unique tokens)")
    return vocab

def main(args):
    """Main preprocessing pipeline."""
    logging.info("Starting preprocessing...")
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        logging.error(f"Input data file not found: {args.input_file}")
        return

    stopwords_set = load_stopwords(args.stop_words_file)
    if not stopwords_set:
        logging.warning("Stopwords set is empty. Proceeding without stopword removal.")

    logging.info("Cleaning and tokenizing text...")
    tokens = clean_and_tokenize(data, stopwords_set)
    
    logging.info("Normalizing and lemmatizing text...")
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    processed_tokens = normalize_and_lemmatize(tokens, normalizer, lemmatizer)
    
    # Save processed text
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(processed_tokens))
        logging.info(f"Processed text saved to {args.output_file}")
    except IOError as e:
        logging.error(f"Error saving processed text: {e}")

    # Build and save vocabulary
    vocab = build_vocab(processed_tokens, args.min_freq)
    save_vocab(vocab, args.vocab_file)

    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Persian text data.")
    parser.add_argument('--input_file', type=str, default='data/raw/Persian-WikiText-1.txt',
                        help='Path to the raw input text file.')
    parser.add_argument('--stop_words_file', type=str, default='data/raw/Persian_Stop_Words.txt',
                        help='Path to the stop words file.')
    parser.add_argument('--output_file', type=str, default='data/processed/processed_text.txt',
                        help='Path to save the processed text.')
    parser.add_argument('--vocab_file', type=str, default='data/processed/vocab.json',
                        help='Path to save the vocabulary JSON.')
    parser.add_argument('--min_freq', type=int, default=5,
                        help='Minimum frequency for a word to be included in the vocab.')
    
    args = parser.parse_args()
    main(args)
