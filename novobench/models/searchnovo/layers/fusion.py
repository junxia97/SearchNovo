import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionLayer(nn.Module):
    def __init__(self, embed_dim, head_num=4, dropout = 0.0):
        super(FusionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, head_num, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, head_num, dropout=dropout)  # Assume same embed_dim for y and y_r
        self.norm = nn.LayerNorm(embed_dim)
  
    def forward(self, y, y_r, y_mask, y_ref_mask, tgt_mask, similarity_score=None):
        """
        Forward pass for the FusionLayer with padding masks.

        :param y: Input tensor y of shape (batch_size, seq_len1, embed_dim).
        :param y_r: Input tensor y_r of shape (batch_size, seq_len2, embed_dim).
        :param y_mask: Padding mask for y of shape (batch_size, seq_len1).
        :param y_ref_mask: Padding mask for y_r of shape (batch_size, seq_len2).
        :returns: Output tensor of shape (batch_size, seq_len1, embed_dim).
        """
        # Transpose y and y_r to fit the expected input shape for nn.MultiheadAttention
        y_transposed = y.transpose(0, 1)  # Shape: (seq_len1, batch_size, embed_dim)
        y_r_transposed = y_r.transpose(0, 1)  # Shape: (seq_len2, batch_size, embed_dim)

        # Perform self-attention on y with y_mask
        self_attn_output, _ = self.self_attn(
            y_transposed, y_transposed, y_transposed,
            key_padding_mask=y_mask,
            attn_mask=tgt_mask
        )

        # Perform cross-attention between y and y_r with y_ref_mask
        cross_attn_output, _ = self.cross_attn(
            y_transposed, y_r_transposed, y_r_transposed,
            key_padding_mask=y_ref_mask
        )

        # Transpose attention outputs back to original shape
        self_attn_output = self_attn_output.transpose(0, 1)
        cross_attn_output = cross_attn_output.transpose(0, 1)

        # Combine self-attention and cross-attention outputs
        attn_output = self_attn_output + cross_attn_output

        # Apply residual connection
        attn_output = attn_output + y

        # Normalize the output
        attn_output = self.norm(attn_output)

        return attn_output
