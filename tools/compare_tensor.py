import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger


def compare_tensors(
    a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6, verbose: bool = False, rtol: float = 0.0
) -> int:
    """
    Compare two PyTorch tensors and count differing elements.

    Finite values match when abs(a - b) <= tol + rtol * abs(b), with b as
    the reference. NaN always counts as a difference, including two NaNs at
    the same position. Infinities match only when they have the same sign;
    finite/infinite pairs and opposite-sign infinities always differ.

    Args:
        a (torch.Tensor): The first tensor to compare.
        b (torch.Tensor): The reference tensor to compare against.
        tol (float, optional): The absolute tolerance threshold. Defaults to 1e-6.
        verbose (bool, optional): If True, print the indices and values of differing elements. Defaults to False.
        rtol (float, optional): The relative tolerance applied to abs(b). Defaults to 0.0,
            preserving absolute-only comparison for existing calls.

    Returns:
        int: The number of differing elements under the tolerance and non-finite policies.

    Raises:
        ValueError: If the shapes of the input tensors do not match.
    """
    # Check if tensor shapes are the same
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Separate non-finite values so NaN arithmetic cannot hide differences.
    finite_mask = torch.isfinite(a) & torch.isfinite(b)
    difference = a - b
    # Retain the original scalar threshold for absolute-only comparisons.
    tolerance = tol
    if rtol != 0.0:
        reference = b.to(dtype=difference.dtype)
        tolerance = tol + rtol * reference.abs()
    finite_diff = difference.abs() > tolerance
    diff_mask = (finite_mask & finite_diff) | (~finite_mask & (a != b))

    # Count the number of differing elements
    diff_count = int(diff_mask.sum().item())

    # If verbose, log details of differing elements
    if verbose and diff_count > 0:
        indices = torch.nonzero(diff_mask, as_tuple=False)
        for idx in indices:
            coordinate = tuple(idx.tolist())
            coordinate_text = ",".join(str(value) for value in coordinate)
            a_value = a[coordinate].item()
            b_value = b[coordinate].item()
            difference = (a[coordinate] - b[coordinate]).item()
            logger.info(f"diff at {coordinate_text}: {a_value:.6f} - {b_value:.6f} = {difference:.6f}")

    return diff_count


if __name__ == "__main__":
    # example:
    # a = torch.load("/path/to/a.pt")
    # b = torch.load("/path/to/b.pt")
    # diff_count = compare_tensors(a, b)
    # print(f"diff count: {diff_count}")
    pass
