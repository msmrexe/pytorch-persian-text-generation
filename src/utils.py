import logging
import json
import torch
import matplotlib.pyplot as plt
import math
import os

def setup_logging(log_file='logs/main.log'):
    """Configures logging to both console and file."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def save_vocab(vocab, file_path):
    """Saves vocabulary (dict) to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
        logging.info(f"Vocabulary saved to {file_path}")
    except IOError as e:
        logging.error(f"Error saving vocab to {file_path}: {e}")

def load_vocab(file_path):
    """Loads vocabulary (dict) from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        logging.info(f"Vocabulary loaded from {file_path}")
        return vocab
    except FileNotFoundError:
        logging.error(f"Vocab file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from vocab file: {file_path}")
        return None

def save_plot(train_losses, val_losses, title, save_path):
    """Saves a plot of training and validation losses."""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_dir = os.path.dirname(save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        plt.savefig(save_path)
        logging.info(f"Loss plot saved to {save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Error saving plot: {e}")

def calculate_perplexity(loss):
    """Calculates perplexity from cross-entropy loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        logging.warning("OverflowError calculating perplexity. Loss might be too high.")
        return float('inf')

def load_model(model, filepath, device):
    """Loads a model's state dict."""
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"Model loaded from {filepath} and moved to {device}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
