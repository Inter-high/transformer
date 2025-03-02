"""
This module implements the Transformer architecture, including scaled dot-product attention,
multi-head attention, position-wise feed-forward networks, positional encoding, and the full
Transformer encoder-decoder model.

Author: yumemonzo@gmail.com
Date: 2025-03-02
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Implements the scaled dot-product attention mechanism.
    """
    def __init__(self, d_k: int) -> None:
        """
        Initialize with the dimension of the key vectors.

        Args:
            d_k (int): Dimension of the key vectors.
        """
        super().__init__()
        self.d_k = d_k

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch, num_heads, seq_len, d_k).
            K (torch.Tensor): Key tensor of shape (batch, num_heads, seq_len, d_k).
            V (torch.Tensor): Value tensor of shape (batch, num_heads, seq_len, d_k).
            mask (Optional[torch.Tensor]): Mask tensor to avoid attention on certain positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor after applying attention and the attention probabilities.
        """
        # Compute raw attention scores and scale them
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply the mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Normalize the scores to probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Multiply probabilities with the value vectors
        output = torch.matmul(attn_probs, V)
        return output, attn_probs


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention by projecting inputs into multiple subspaces,
    applying scaled dot-product attention, and then combining the results.
    """
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Initialize the multi-head attention module.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Linear projections for queries, keys, and values
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # Final linear projection after concatenation
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch, seq_len, d_model).
            K (torch.Tensor): Key tensor of shape (batch, seq_len, d_model).
            V (torch.Tensor): Value tensor of shape (batch, seq_len, d_model).
            mask (Optional[torch.Tensor]): Optional mask tensor.

        Returns:
            torch.Tensor: Output tensor after multi-head attention.
        """
        batch_size = Q.size(0)
        # Linear projection and reshape to (batch, num_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask dimensions for compatibility if mask is provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Compute attention on all heads in parallel
        out, _ = self.attention(Q, K, V, mask)
        # Concatenate the attention output from all heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        # Final linear projection
        return self.W_O(out)


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    """
    def __init__(self, d_model: int, d_ff: int) -> None:
        """
        Initialize the feed-forward network.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed-forward layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network with a ReLU activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feed-forward network.
        """
        return self.fc2(F.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding to inject position information into token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Initialize the positional encoding.

        Args:
            d_model (int): Dimension of the model.
            max_len (int): Maximum length of sequences.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register pe as a buffer so it moves with the model device
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """
    Implements the Transformer encoder.
    """
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        vocab_size: int, 
        max_len: int, 
        dropout: float = 0.1
    ) -> None:
        """
        Initialize the Transformer encoder.

        Args:
            num_layers (int): Number of encoder layers.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
            vocab_size (int): Vocabulary size.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # Create a list of Transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the input sequence.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch, seq_len).
            mask (Optional[torch.Tensor]): Optional mask tensor.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch, seq_len, d_model).
        """
        # Scale embeddings and add positional encoding
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoding(x)
        # Pass through each Transformer layer
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerLayer(nn.Module):
    """
    Implements a single Transformer layer, combining self-attention and feed-forward sub-layers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Initialize a Transformer layer.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            mask (Optional[torch.Tensor]): Optional mask tensor.

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Apply self-attention with residual connection
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # Apply feed-forward network with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer decoder, including self-attention, cross-attention,
    and feed-forward sub-layers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Initialize the Transformer decoder layer.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_output: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer decoder layer.

        Args:
            x (torch.Tensor): Decoder input tensor of shape (batch, seq_len, d_model).
            enc_output (torch.Tensor): Encoder output tensor.
            tgt_mask (Optional[torch.Tensor]): Mask for the target sequence.
            src_mask (Optional[torch.Tensor]): Mask for the source sequence.

        Returns:
            torch.Tensor: Output tensor of the decoder layer.
        """
        # Self-attention with residual connection
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        # Cross-attention with residual connection (attending to encoder outputs)
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask))
        # Feed-forward network with residual connection
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class TransformerDecoder(nn.Module):
    """
    Implements the Transformer decoder by stacking multiple decoder layers.
    """
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        vocab_size: int, 
        max_len: int, 
        dropout: float = 0.1
    ) -> None:
        """
        Initialize the Transformer decoder.

        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
            vocab_size (int): Target vocabulary size.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_output: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer decoder.

        Args:
            x (torch.Tensor): Decoder input tensor of shape (batch, seq_len, d_model).
            enc_output (torch.Tensor): Encoder output tensor.
            tgt_mask (Optional[torch.Tensor]): Mask for the target sequence.
            src_mask (Optional[torch.Tensor]): Mask for the source sequence.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        # Scale embeddings and add positional encoding
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoding(x)
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Implements the full Transformer model, integrating the encoder and decoder.
    """
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        src_vocab: int, 
        tgt_vocab: int, 
        max_len: int, 
        dropout: float = 0.1
    ) -> None:
        """
        Initialize the Transformer model.

        Args:
            num_layers (int): Number of layers for both encoder and decoder.
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward layer.
            src_vocab (int): Source vocabulary size.
            tgt_vocab (int): Target vocabulary size.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, src_vocab, max_len, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, tgt_vocab, max_len, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab)

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the full Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch, src_seq_len).
            tgt (torch.Tensor): Target input tensor of shape (batch, tgt_seq_len).
            src_mask (Optional[torch.Tensor]): Mask for the source sequence.
            tgt_mask (Optional[torch.Tensor]): Mask for the target sequence.

        Returns:
            torch.Tensor: Output logits of shape (batch, tgt_seq_len, tgt_vocab).
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return self.output_layer(dec_output)
