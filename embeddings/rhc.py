"""Residue Hyperdimensional Computing embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchhd
import torchhd.functional as functional


class RHC(nn.Module):
    """Residue Hyperdimensional Computing (RHC) that forms hypervectors using the methods described in `Computing with Residue Numbers in High-Dimensional Representation <https://arxiv.org/abs/2311.04872>`_.
    
    Args:
        in_features (int): The dimensionality of the input feature vector.
        out_features (out): The dimensionality of the output vectors.
        vsa (VSAOptions | None): Defaults to ``"FHRR"``, and can only be ``"FHRR"``.
        dtype (torch.dtype | None): the desired data type of the returned tensor. Default: ``None``, uses the datatype of "FHRR".
        device (torch.device | None): the desired device of the returned tensor. Default: ``None``, uses the current device for the default tensor type.
        requires_grad (bool | None): If autograd should record operations on the returned tensor. Default: ``False``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        moduli: torch.Tensor | list[int] | tuple[int]=torch.tensor([3, 5, 7, 11]),
        vsa: torchhd.types.VSAOptions="FHRR",
        dtype: torch.dtype | None=None,
        device: torch.types.Device | None=None,
        requires_grad: bool = False,
    ) -> None:
        super(RHC, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.moduli = moduli

        if vsa not in {"FHRR"}:
            raise ValueError(
                f"RHC embedding only supports FHRR but provided: {vsa}"
            )

        self.vsa_tensor = functional.get_vsa_tensor_class(vsa)

        if dtype is not None and dtype not in self.vsa_tensor.supported_dtypes:
            raise ValueError(f"dtype {dtype} not supported by {vsa}")

        factory_args = { "device": device, "dtype": dtype }

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, **factory_args)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Generate the basis hypervectors for the moduli computing."""