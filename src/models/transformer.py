import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Injects positional information into the embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Implements Multi-Head Attention from scratch."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

    def split_heads(self, x, batch_size):
        """Split d_model into num_heads * d_k"""
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x, batch_size):
        """Combine num_heads * d_k back to d_model"""
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Split heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # 3. Scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Combine heads
        attn_output = self.combine_heads(attn_output, batch_size) # (batch_size, seq_len_q, d_model)
        
        # 5. Output linear layer
        output = self.W_o(attn_output)
        return output

class PositionwiseFeedForward(nn.Module):
    """Implements the FFN layer."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """A single encoder layer."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Self-attention
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        return src

class DecoderLayer(nn.Module):
    """A single decoder layer."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Masked self-attention
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        
        # Cross-attention (Query: tgt, Key/Value: memory)
        attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        return tgt

class TextGenTransformer(nn.Module):
    """
    A from-scratch Encoder-Decoder Transformer for n-gram language modeling.
    The 'source' (src) is the n-gram context (e.g., 2 words).
    The 'target' (tgt) is the word to be predicted (e.g., 3rd word).
    
    During training, tgt is the ground truth (shifted right).
    This implementation uses the n-gram context as ENCODER input
    and the target-to-be-predicted as DECODER input, as in the original notebook.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(TextGenTransformer, self).__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout) # Using (seq, batch, embed)
        
        # Encoder
        encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Decoder
        decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Final output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        """Initializes weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, sz, device):
        """Generates a causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src, tgt):
        """
        src shape: (batch_size, src_seq_len) - e.g., (64, 2)
        tgt shape: (batch_size, tgt_seq_len) - e.g., (64, 1)
        """
        device = src.device
        
        # We don't need masks for the n-gram context (src)
        src_mask = None 
        
        # We need a causal mask for the decoder target (tgt)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device) # Shape: (tgt_seq_len, tgt_seq_len)
        
        # We don't use padding masks for this n-gram task, but could be added
        src_padding_mask = None
        tgt_padding_mask = None
        memory_key_padding_mask = None

        # --- Process Inputs ---
        # Note: Transformer expects (seq_len, batch_size, d_model)
        # 1. Embed and add positional encoding
        # src: (batch_size, src_seq_len) -> (src_seq_len, batch_size, d_model)
        src_emb = self.pos_encoder(self.embedding(src).transpose(0, 1) * math.sqrt(self.d_model))
        # tgt: (batch_size, tgt_seq_len) -> (tgt_seq_len, batch_size, d_model)
        tgt_emb = self.pos_encoder(self.embedding(tgt).transpose(0, 1) * math.sqrt(self.d_model))

        # --- Encoder ---
        # memory shape: (src_seq_len, batch_size, d_model)
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask)

        # --- Decoder ---
        # output shape: (tgt_seq_len, batch_size, d_model)
        output = tgt_emb
        for layer in self.decoder:
            output = layer(output, memory, tgt_mask, memory_key_padding_mask)
            
        # --- Final Output ---
        # (tgt_seq_len, batch_size, d_model) -> (batch_size, tgt_seq_len, d_model)
        output = output.transpose(0, 1)
        
        # (batch_size, tgt_seq_len, d_model) -> (batch_size, tgt_seq_len, vocab_size)
        logits = self.fc_out(output)
        
        return logits
