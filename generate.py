import torch
import torch.nn.functional as F
import argparse
import logging
import re
import string
from hazm import Normalizer, Lemmatizer, word_tokenize

from src.models.rnn import TextGenerationRNN
from src.models.transformer import TextGenTransformer
from src.utils import load_vocab, load_model, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    
    parser.add_argument('--model_type', type=str, required=True, choices=['rnn', 'transformer'])
    parser.add_-Hargument('--model_path', type=str, required=True, help='Path to the trained model (.pth) file.')
    parser.add_argument('--vocab_file', type=str, default='data/processed/vocab.json')
    parser.add_argument('--seed_text', type=str, default='ایران کشوری در')
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--n_gram', type=int, default=2, help='Context window size (must match training).')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling. k=0 means greedy (argmax).')
    
    # Model Hyperparameters (must match trained model)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    return parser.parse_args()

def preprocess_seed(text, vocab, n, normalizer, lemmatizer):
    """Preprocesses the seed text and returns the last n token indices."""
    unk_idx = vocab.get('<unk>', 1)
    pad_idx = vocab.get('<pad>', 0)
    
    # Clean, normalize, and lemmatize
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    normalized_text = normalizer.normalize(text)
    tokens = word_tokenize(normalized_text)
    lemmatized_tokens = [lemmatizer.lemmatize(word).split('#')[0] for word in tokens]
    
    # Convert to indices
    indices = [vocab.get(token, unk_idx) for token in lemmatized_tokens]
    
    # Get the last n tokens, padding if necessary
    context = indices[-n:]
    if len(context) < n:
        context = [pad_idx] * (n - len(context)) + context
        
    return torch.tensor(context, dtype=torch.long)

def sample_top_k(logits, k):
    """Performs Top-K sampling."""
    if k == 0:
        # Greedy decoding (argmax)
        return torch.argmax(logits, dim=-1)
    
    # Get top k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Apply softmax to the top k logits
    probabilities = F.softmax(top_k_logits, dim=-1)
    
    # Sample from the distribution
    sampled_index_in_top_k = torch.multinomial(probabilities, 1)
    
    # Get the actual token index
    sampled_token_index = top_k_indices.gather(-1, sampled_index_in_top_k)
    return sampled_token_index.squeeze(-1)

def generate_rnn(model, seed_indices, vocab, n, max_length, top_k, device):
    """Generates text using the RNN model."""
    model.eval()
    idx_to_word = {i: w for w, i in vocab.items()}
    generated_indices = seed_indices.tolist()
    
    # Input shape: (batch_size=1, seq_len=n)
    inputs = seed_indices.unsqueeze(0).to(device)
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(inputs, hidden)
            
            # output shape: (1, vocab_size)
            logits = output.squeeze(0)
            
            # Sample next token
            next_token_idx = sample_top_k(logits, top_k)
            
            if next_token_idx.item() == vocab.get('<pad>', 0):
                break
                
            generated_indices.append(next_token_idx.item())
            
            # Update inputs: sliding window
            # (1, n) -> (1, n-1) -> (1, n)
            inputs = torch.cat((inputs[:, 1:], next_token_idx.unsqueeze(0).unsqueeze(0)), dim=1)

    return ' '.join([idx_to_word.get(i, '<unk>') for i in generated_indices])

def generate_transformer(model, seed_indices, vocab, max_length, top_k, device):
    """Generates text using the Transformer model."""
    model.eval()
    idx_to_word = {i: w for w, i in vocab.items()}
    generated_indices = seed_indices.tolist()
    
    # Encoder input (context) shape: (batch_size=1, seq_len=n)
    encoder_input = seed_indices.unsqueeze(0).to(device)
    
    # Decoder input starts with the last token of the seed
    # (or a start token, but here we'll use the last context token)
    decoder_input_indices = [seed_indices[-1].item()]

    with torch.no_grad():
        for _ in range(max_length):
            # decoder_input shape: (1, current_len)
            decoder_input = torch.tensor(decoder_input_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            # output shape: (1, current_len, vocab_size)
            logits = model(encoder_input, decoder_input)
            
            # Get logits for the *last* token only
            # shape: (vocab_size)
            last_token_logits = logits[:, -1, :].squeeze(0)
            
            # Sample next token
            next_token_idx = sample_top_k(last_token_logits, top_k)
            
            if next_token_idx.item() == vocab.get('<pad>', 0):
                break
                
            generated_indices.append(next_token_idx.item())
            decoder_input_indices.append(next_token_idx.item())

    return ' '.join([idx_to_word.get(i, '<unk>') for i in generated_indices])

def main():
    setup_logging('logs/generate.log')
    args = parse_args()
    logging.info(f"Arguments: {vars(args)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    vocab = load_vocab(args.vocab_file)
    if not vocab:
        return
        
    vocab_size = len(vocab)
    
    # Load model
    if args.model_type == 'rnn':
        model = TextGenerationRNN(vocab_size, args.embed_size, args.hidden_size, args.num_layers, args.dropout)
    else:
        # Use embed_size for d_model
        if args.d_model == 128: args.d_model = args.embed_size
        model = TextGenTransformer(vocab_size, args.d_model, args.num_heads, args.num_layers, args.d_ff, args.dropout)
        
    model = load_model(model, args.model_path, device)
    if not model:
        return
        
    # Prepare seed
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    seed_indices = preprocess_seed(args.seed_text, vocab, args.n_gram, normalizer, lemmatizer)
    
    logging.info(f"Original seed text: '{args.seed_text}'")
    logging.info(f"Processed seed indices: {seed_indices.tolist()}")
    
    # Generate
    if args.model_type == 'rnn':
        generated_text = generate_rnn(model, seed_indices, vocab, args.n_gram, args.max_length, args.top_k, device)
    else:
        generated_text = generate_transformer(model, seed_indices, vocab, args.max_length, args.top_k, device)
        
    print("\n--- Generated Text ---")
    print(generated_text)
    print("------------------------")
    logging.info(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
