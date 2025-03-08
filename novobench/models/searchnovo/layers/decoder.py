import torch

from .base import _PeptideTransformer, FloatEncoder, listify, generate_tgt_mask
from .fusion import FusionLayer

class PeptideSearchDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    reverse : bool, optional
        Sequence peptides from c-terminus to n-terminus.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=True,
        residues="canonical",
        max_charge=5,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse

        # Additional model components
        self.mass_encoder = FloatEncoder(dim_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.fusion = FusionLayer(dim_model, n_head, dropout)
        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)

    def forward(self, sequences, precursors, memory, memory_key_padding_mask, sequences_ref, sequence_ref_sim):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """
        # Prepare sequences
        tokens = self.prepare_tokens(sequences)
        tokens_ref = self.prepare_tokens(sequences_ref)

        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        #  We concatenate the precursor with the encoded amino acids first to ensure that the positional encoding is applied consistently
        tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1) if sequences is not None else precursors
        tgt_ref = torch.cat([precursors, self.aa_encoder(tokens_ref)], dim=1)
 
        # Create padding masks
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt_key_padding_mask_ref = tgt_ref.sum(axis=2) == 0

        # Apply positional encoding
        tgt = self.pos_encoder(tgt)
        tgt_ref = self.pos_encoder(tgt_ref)

        # Slice the amino acid parts for ref sequence
        precursor_length = precursors.size(1)
        tgt_ref_aa = tgt_ref[:, precursor_length:, :]
        tgt_key_padding_mask_ref_aa = tgt_key_padding_mask_ref[:, precursor_length:]

        sequence_ref_sim = sequence_ref_sim.unsqueeze(-1)
        tgt_ref_aa = tgt_ref_aa * sequence_ref_sim

        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(self.device)

        # Apply fusion
        tgt = self.fusion(tgt, tgt_ref_aa, tgt_key_padding_mask, tgt_key_padding_mask_ref_aa, tgt_mask, sequence_ref_sim)


        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )
        return self.final(preds), tokens


    def prepare_tokens(self, sequences):
        """
        Prepares tokens from the input sequences by tokenizing and padding them.

        Parameters
        ----------
        sequences : list or None
            A list of sequences to be tokenized.

        Returns
        -------
        torch.Tensor
            A padded tensor of tokens.
        """
        if sequences is not None:
            sequences = listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)

        return tokens
