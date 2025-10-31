import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from tqdm import tqdm
import time

from src.dataset import prepare_dataloaders
from src.models.rnn import TextGenerationRNN
from src.models.transformer import TextGenTransformer
from src.utils import setup_logging, save_plot, calculate_perplexity


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text generation model (RNN or Transformer).")
    
    # Paths
    parser.add_argument('--data_file', type=str, default='data/processed/processed_text.txt')
    parser.add_argument('--vocab_file', type=str, default='data/processed/vocab.json')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models/')
    parser.add_argument('--plot_save_dir', type=str, default='outputs/plots/')
    parser.add_argument('--log_file', type=str, default='logs/train.log')
    
    # Model Choice
    parser.add_argument('--model_type', type=str, required=True, choices=['rnn', 'transformer'])
    
    # Data & Training
    parser.add_argument('--n_gram', type=int, default=2, help='Context window size (n-gram).')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value.')
    
    # Model Hyperparameters
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size (for RNN).')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Transformer-specific
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension (for Transformer). Use embed_size.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (for Transformer).')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dim (for Transformer).')

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, clip, device, model_type):
    model.train()
    total_loss = 0
    
    # Initialize hidden state for RNN
    if model_type == 'rnn':
        hidden = model.init_hidden(args.batch_size, device)

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'rnn':
            # Detach hidden state
            hidden = hidden.detach()
            outputs, hidden = model(inputs, hidden)
            # RNN outputs (batch_size, vocab_size), targets (batch_size)
            loss = criterion(outputs, targets)
        
        elif model_type == 'transformer':
            # Transformer model expects target as (batch_size, 1)
            # We use the target as the decoder input (a sequence of 1)
            decoder_input = targets.unsqueeze(-1)
            
            # outputs shape: (batch_size, 1, vocab_size)
            outputs = model(inputs, decoder_input)
            
            # Reshape for loss: (batch_size, vocab_size)
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            # Reshape targets: (batch_size)
            targets_flat = targets.view(-1)
            
            loss = criterion(outputs_flat, targets_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, model_type):
    model.eval()
    total_loss = 0
    
    if model_type == 'rnn':
        hidden = model.init_hidden(args.batch_size, device)
        
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if model_type == 'rnn':
                hidden = hidden.detach()
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, targets)
                
            elif model_type == 'transformer':
                decoder_input = targets.unsqueeze(-1)
                outputs = model(inputs, decoder_input)
                outputs_flat = outputs.view(-1, outputs.shape[-1])
                targets_flat = targets.view(-1)
                loss = criterion(outputs_flat, targets_flat)
                
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


def main(args):
    setup_logging(args.log_file)
    logging.info("Starting training process...")
    logging.info(f"Arguments: {vars(args)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Prepare data
    logging.info("Preparing dataloaders...")
    train_loader, val_loader, test_loader, vocab_size = prepare_dataloaders(
        args.data_file, args.vocab_file, args.n_gram, args.batch_size
    )
    if not train_loader:
        logging.error("Failed to create dataloaders. Exiting.")
        return
    
    logging.info(f"Vocabulary size: {vocab_size}")

    # Initialize model
    if args.model_type == 'rnn':
        logging.info("Initializing RNN (GRU) model...")
        model = TextGenerationRNN(
            vocab_size, args.embed_size, args.hidden_size, 
            args.num_layers, args.dropout
        ).to(device)
    elif args.model_type == 'transformer':
        logging.info("Initializing Transformer model...")
        # Use embed_size for d_model
        model = TextGenTransformer(
            vocab_size, args.embed_size, args.num_heads, 
            args.num_layers, args.d_ff, args.dropout
        ).to(device)
    
    logging.info(model)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <pad> token
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    model_save_path = os.path.join(args.model_save_dir, f"{args.model_type}_best.pth")
    plot_save_path = os.path.join(args.plot_save_dir, f"{args.model_type}_loss.png")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.clip, device, args.model_type)
        val_loss = evaluate(model, val_loader, criterion, device, args.model_type)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        logging.info(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | "
                     f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                     f"Val PPL: {calculate_perplexity(val_loss):.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved to {model_save_path}")

    # Final evaluation on test set
    logging.info("Training complete. Loading best model for test evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss = evaluate(model, test_loader, criterion, device, args.model_type)
    test_ppl = calculate_perplexity(test_loss)
    
    logging.info(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.4f}")

    # Save loss plot
    save_plot(train_losses, val_losses, f"{args.model_type.upper()} Model Loss", plot_save_path)
    
    logging.info("Training process finished.")


if __name__ == "__main__":
    args = parse_args()
    
    # Use embed_size for d_model if d_model is not explicitly set
    if args.model_type == 'transformer' and args.d_model == 128:
        args.d_model = args.embed_size

    main(args)
