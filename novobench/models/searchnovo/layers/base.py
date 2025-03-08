
# model components and helper classes/functions adapted from depthcharge v0.2.3 https://github.com/wfondrie/depthcharge/releases/tag/v0.2.3

"""Base Transformer models for working with mass spectra and peptides"""
import re

import torch
import einops
import math
import numpy as np


class PeptideMass:
    """A simple class for calculating peptide masses

    Parameters
    ----------
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    canonical = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C+57.021": 103.009184505 + 57.02146,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        # "U": 150.953633405,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        # "O": 237.147726925,
    }

    # Modfications found in MassIVE-KB
    massivekb = {
        # N-terminal mods:
        "+42.011": 42.010565,  # Acetylation
        "+43.006": 43.005814,  # Carbamylation
        "-17.027": -17.026549,  # NH3 loss
        "+43.006-17.027": (43.006814 - 17.026549),
        # AA mods:
        "M+15.995": canonical["M"] + 15.994915,  # Met Oxidation
        "N+0.984": canonical["N"] + 0.984016,  # Asn Deamidation
        "Q+0.984": canonical["Q"] + 0.984016,  # Gln Deamidation
    }

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        """Initialize the PeptideMass object"""
        if residues == "canonical":
            self.masses = self.canonical
        elif residues == "massivekb":
            self.masses = self.canonical
            self.masses.update(self.massivekb)
        else:
            self.masses = residues

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass


class FloatEncoder(torch.nn.Module):
    """Encode floating point values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        # Get dimensions for equations:
        d_sin = math.ceil(dim_model / 2)
        d_cos = dim_model - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, dim_model).float() - d_sin) / (
            d_cos - 1
        )
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PositionalEncoder(FloatEncoder):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float, optional
        The shortest wavelength in the geometric progression.
    max_wavelength : float, optional
        The longest wavelength in the geometric progression.
    """

    def __init__(self, dim_model, min_wavelength=1, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__(
            dim_model=dim_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    pos_encoder : bool
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum charge to embed.
    """

    def __init__(
        self,
        dim_model,
        pos_encoder,
        residues,
        max_charge,
    ):
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )

    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            sequence = list(reversed(sequence))

        if not partial:
            if sequence == ['']: #NOTE new added for searchnovo
                sequence = ["$"]
            else:
                sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


def generate_tgt_mask(sz):
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)



def listify(obj):
    """Turn an object into a list, but don't split strings."""
    try:
        assert not isinstance(obj, str)
        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)
