import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.utils import load_vocab
import logging

def create_ngrams_indices(processed_text, vocab, n):
    """Creates n-gram (context, target) pairs from text and converts to indices."""
    logging.info(f"Creating {n}-grams...")
    tokens = processed_text.split()
    
    contexts = []
    targets = []
    
    unk_idx = vocab.get('<unk>', 1)
    
    # Convert all tokens to indices first
    indices = [vocab.get(token, unk_idx) for token in tokens]
    
    if len(indices) <= n:
        logging.warning(f"Text length ({len(indices)}) is not greater than n-gram size ({n}). No n-grams created.")
        return [], []
        
    for i in range(len(indices) - n):
        context_indices = indices[i:i+n]
        target_index = indices[i+n]
        contexts.append(context_indices)
        targets.append(target_index)
        
    logging.info(f"Created {len(contexts)} n-gram samples.")
    return contexts, targets

def prepare_dataloaders(text_file_path, vocab_path, n, batch_size, test_split=0.1, val_split=0.1):
    """Loads data, creates n-grams, and prepares dataloaders."""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            processed_text = f.read()
    except FileNotFoundError:
        logging.error(f"Processed text file not found: {text_file_path}")
        return None, None, None

    vocab = load_vocab(vocab_path)
    if not vocab:
        return None, None, None

    contexts, targets = create_ngrams_indices(processed_text, vocab, n)
    
    if not contexts:
        logging.error("No n-grams were created. Check data and n-gram size.")
        return None, None, None

    # Convert to tensors
    try:
        inputs_tensor = torch.tensor(contexts, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
    except Exception as e:
        logging.error(f"Error converting n-grams to tensors: {e}")
        return None, None, None

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        logging.error("Dataset size is too small to create train/val/test splits.")
        return None, None, None

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    logging.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    logging.info(f"DataLoaders created with batch size {batch_size}.")
    
    return train_loader, val_loader, test_loader, len(vocab)
