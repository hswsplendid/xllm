import cmath
import math
import os
import sys
from collections.abc import Iterator
from fractions import Fraction

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger


def _surd_is_positive(constant: Fraction, coefficient: Fraction, radicand: Fraction) -> bool:
    """Determine whether constant + coefficient * sqrt(radicand) is positive exactly."""
    if coefficient == 0 or radicand == 0:
        return constant > 0
    if coefficient > 0:
        return constant >= 0 or coefficient**2 * radicand > constant**2
    return constant > 0 and constant**2 > coefficient**2 * radicand


def _exact_complex_difference(
    actual: complex | float | int, reference: complex | float | int, tol: float, rtol: float
) -> bool:
    """Compare finite complex magnitudes using exact squared components and thresholds."""
    actual_real, actual_imag = Fraction(actual.real), Fraction(actual.imag)
    reference_real, reference_imag = Fraction(reference.real), Fraction(reference.imag)
    difference_squared = (actual_real - reference_real) ** 2 + (actual_imag - reference_imag) ** 2
    reference_squared = reference_real**2 + reference_imag**2
    absolute_tol, relative_tol = Fraction(tol), Fraction(rtol)
    # A negative threshold is exceeded even by an exact match.
    if _surd_is_positive(-absolute_tol, -relative_tol, reference_squared):
        return True
    return _surd_is_positive(
        difference_squared - absolute_tol**2 - relative_tol**2 * reference_squared,
        -2 * absolute_tol * relative_tol,
        reference_squared,
    )


def _exact_difference_mask(a: torch.Tensor, b: torch.Tensor, tol: float, rtol: float) -> torch.Tensor:
    """Resolve precision-sensitive candidates with exact arithmetic on host scalars."""
    finite_tolerance = math.isfinite(tol) and math.isfinite(rtol)
    if finite_tolerance:
        absolute_tol = Fraction(tol)
        relative_tol = Fraction(rtol)
        denominator = absolute_tol.denominator * relative_tol.denominator
        absolute_numerator = absolute_tol.numerator * relative_tol.denominator
        relative_numerator = relative_tol.numerator * absolute_tol.denominator
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    diff_mask = torch.empty(flat_a.shape, dtype=torch.bool, device=a.device)
    # Bound temporary Python objects and host transfers when inspecting large dumps.
    chunk_size = 16384
    for start in range(0, flat_a.numel(), chunk_size):
        stop = min(start + chunk_size, flat_a.numel())
        differences = []
        for actual, reference in zip(flat_a[start:stop].tolist(), flat_b[start:stop].tolist()):
            if not (cmath.isfinite(actual) and cmath.isfinite(reference)):
                differences.append(actual != reference)
                continue
            if not finite_tolerance:
                # A finite delta exceeds only a negative infinite threshold. For non-finite
                # tolerances, only whether the reference magnitude is zero affects its sign.
                threshold = tol + rtol * (0.0 if reference == 0 else 1.0)
                differences.append(threshold < 0.0)
            elif isinstance(actual, complex) or isinstance(reference, complex):
                differences.append(_exact_complex_difference(actual, reference, tol, rtol))
            else:
                actual_numerator, actual_denominator = actual.as_integer_ratio()
                reference_numerator, reference_denominator = reference.as_integer_ratio()
                difference_numerator = abs(
                    actual_numerator * reference_denominator - reference_numerator * actual_denominator
                )
                tolerance_numerator = absolute_numerator * reference_denominator + relative_numerator * abs(
                    reference_numerator
                )
                # Cross-multiply exact ratios; the common reference denominator cancels.
                differences.append(difference_numerator * denominator > tolerance_numerator * actual_denominator)
        diff_mask[start:stop] = torch.tensor(differences, dtype=torch.bool, device=a.device)
    return diff_mask.reshape(a.shape)


def _absolute_integral_difference_mask(a: torch.Tensor, b: torch.Tensor, tol: float) -> torch.Tensor:
    """Compare signed-representable integral deltas without overflowing int64 subtraction."""
    if math.isnan(tol) or tol >= 2**64 - 1:
        return torch.zeros_like(a, dtype=torch.bool)
    if tol < 0:
        return torch.ones_like(a, dtype=torch.bool)
    if tol < 1:
        return a != b
    actual, reference = a.to(torch.int64), b.to(torch.int64)
    high, low = torch.maximum(actual, reference), torch.minimum(actual, reference)
    threshold = math.floor(tol)
    largest = torch.iinfo(torch.int64).max
    if threshold <= largest:
        # Clamp before adding so even rejected lanes cannot overflow.
        limit = largest - threshold
        return (low <= limit) & (high > low.clamp(max=limit) + threshold)
    # A threshold above INT64_MAX can only be exceeded by opposite-sign operands.
    offset = threshold - 2**63
    shifted_low = low.clamp(max=-1) - torch.iinfo(torch.int64).min
    limit = largest - offset
    return (low < 0) & (shifted_low <= limit) & (high > shifted_low.clamp(max=limit) + offset)


def _non_finite_tolerance_mask(reference: torch.Tensor, tol: float, rtol: float) -> torch.Tensor:
    """Classify non-finite thresholds for finite operands without computing their magnitudes."""
    return tol + rtol * (reference != 0).to(torch.float64) < 0


def _integral_difference_mask(a: torch.Tensor, b: torch.Tensor, tol: float, rtol: float) -> torch.Tensor:
    """Vectorize safe real comparisons involving integers, resolving uncertain candidates exactly."""
    # uint64 mixed promotion/arithmetic is not supported by all PyTorch backends.
    if a.dtype == torch.uint64 or b.dtype == torch.uint64:
        if a.dtype == b.dtype and rtol == 0 and 0 <= tol < 1:
            return a != b
        return _exact_difference_mask(a, b, tol, rtol)
    # These unsigned types lack mixed promotion and Boolean indexing kernels.
    a = a.to(torch.int64) if a.dtype in (torch.uint16, torch.uint32) else a
    b = b.to(torch.int64) if b.dtype in (torch.uint16, torch.uint32) else b
    a_integral, b_integral = not a.is_floating_point(), not b.is_floating_point()
    integral_pair = a_integral and b_integral
    if rtol == 0 and integral_pair:
        return _absolute_integral_difference_mask(a, b, tol)
    if not (math.isfinite(tol) and math.isfinite(rtol)):
        return _non_finite_tolerance_mask(b, tol, rtol)

    actual, reference = a.to(torch.float64), b.to(torch.float64)
    difference = (actual - reference).abs()
    relative = rtol * reference.abs()
    threshold = tol + relative
    # Small integer conversions are exact; two such integers also have an exact delta.
    sensitive = torch.zeros_like(a, dtype=torch.bool)
    if a_integral:
        sensitive |= actual.abs() > 2**52
    if b_integral:
        sensitive |= reference.abs() > 2**52
    if integral_pair and tol == 0 and abs(math.frexp(rtol)[0]) == 0.5:
        # Scaling integers by a binary power is exact unless it overflows; the sign of
        # an overflowing threshold still determines the comparison with a small delta.
        uncertain = sensitive
    else:
        # Enclose both rounded threshold operations with adjacent float64 values.
        negative_inf = torch.full_like(relative, -math.inf)
        positive_inf = torch.full_like(relative, math.inf)
        lower = torch.nextafter(tol + torch.nextafter(relative, negative_inf), negative_inf)
        upper = torch.nextafter(tol + torch.nextafter(relative, positive_inf), positive_inf)
        if integral_pair:
            uncertain = sensitive | ((difference > lower) & (difference <= upper))
        else:
            # A mixed real delta may round too, so enclose its subtraction as well.
            delta_lower = torch.nextafter(difference, negative_inf)
            delta_upper = torch.nextafter(difference, positive_inf)
            uncertain = sensitive | ~((delta_upper <= lower) | (delta_lower > upper))
    differences = difference > threshold
    if tol >= 0 and rtol >= 0:
        equal = (a == b) if integral_pair else ((actual == reference) & ~sensitive)
        uncertain &= ~equal
        differences &= ~equal
    if not integral_pair:
        uncertain &= torch.isfinite(actual) & torch.isfinite(reference)
    if uncertain.any():
        differences[uncertain] = _exact_difference_mask(a[uncertain], b[uncertain], tol, rtol)
    return differences


def _relative_difference_mask(a: torch.Tensor, b: torch.Tensor, tol: float, rtol: float) -> torch.Tensor:
    """Widen one floating chunk and compare finite overflow candidates with exact arithmetic."""
    comparison_dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), torch.float64)
    actual = a.to(dtype=comparison_dtype)
    reference = b.to(dtype=comparison_dtype)
    if not (math.isfinite(tol) and math.isfinite(rtol)):
        return _non_finite_tolerance_mask(reference, tol, rtol)
    difference = (actual - reference).abs()
    reference_magnitude = reference.abs()
    relative_threshold = rtol * reference_magnitude
    differences = difference > tol + relative_threshold
    # Low-precision inputs usually cannot overflow float64. Proving that from their
    # dtype limits avoids synchronizing CUDA to inspect an empty candidate mask.
    largest = max(torch.finfo(a.dtype).max, torch.finfo(b.dtype).max)
    magnitude_bound = largest * (2 if a.is_complex() or b.is_complex() else 1)
    if math.isfinite(2 * magnitude_bound) and math.isfinite(rtol * magnitude_bound):
        return differences
    overflow = (
        torch.isfinite(actual)
        & torch.isfinite(reference)
        & (torch.isinf(difference) | torch.isinf(reference_magnitude) | torch.isinf(relative_threshold))
    )
    if overflow.any():
        differences[overflow] = _exact_difference_mask(a[overflow], b[overflow], tol, rtol)
    return differences


def _iter_tensor_chunks(
    a: torch.Tensor, b: torch.Tensor, chunk_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield bounded row-major chunks, including strided inputs without whole-tensor copies."""
    if a.numel() <= chunk_size:
        yield a.reshape(-1), b.reshape(-1)
    elif a.is_contiguous() and b.is_contiguous():
        flat_a, flat_b = a.view(-1), b.view(-1)
        for start in range(0, a.numel(), chunk_size):
            yield flat_a[start : start + chunk_size], flat_b[start : start + chunk_size]
    elif a[0].numel() > chunk_size:
        for index in range(a.shape[0]):
            yield from _iter_tensor_chunks(a[index], b[index], chunk_size)
    else:
        rows = max(1, chunk_size // a[0].numel())
        for start in range(0, a.shape[0], rows):
            yield a[start : start + rows].reshape(-1), b[start : start + rows].reshape(-1)


def _chunk_difference_mask(a: torch.Tensor, b: torch.Tensor, tol: float, rtol: float) -> torch.Tensor:
    """Compare one chunk while retaining absolute-only floating and non-finite policies."""
    device = a.device
    a_integral = not (a.is_floating_point() or a.is_complex())
    b_integral = not (b.is_floating_point() or b.is_complex())
    if device.type not in ("cpu", "cuda") and rtol != 0:
        a, b = a.cpu(), b.cpu()
    if a_integral and b_integral:
        differences = _integral_difference_mask(a, b, tol, rtol)
    elif rtol != 0 and (a_integral or b_integral):
        if a.is_complex() or b.is_complex():
            differences = _exact_difference_mask(a, b, tol, rtol)
        else:
            finite = torch.isfinite(a) & torch.isfinite(b)
            differences = ~finite | (finite & _integral_difference_mask(a, b, tol, rtol))
    else:
        finite = torch.isfinite(a) & torch.isfinite(b)
        same_infinity = torch.isinf(a) & torch.isinf(b) & (a == b)
        finite_diff = (a - b).abs() > tol if rtol == 0 else _relative_difference_mask(a, b, tol, rtol)
        differences = (finite & finite_diff) | (~finite & ~same_infinity)
    return differences.to(device=device)


def _format_scalar(value: complex | float | int | Fraction) -> str:
    """Format six decimal places without converting exact integers or ratios to float."""
    if isinstance(value, int):
        return f"{int(value)}.000000"
    if isinstance(value, Fraction):
        whole, fractional = divmod(round(abs(value) * 1_000_000), 1_000_000)
        sign = "-" if value < 0 else ""
        return f"{sign}{whole}.{fractional:06d}"
    return f"{value:.6f}"


def compare_tensors(
    a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6, verbose: bool = False, rtol: float = 0.0
) -> int:
    """
    Compare two PyTorch tensors and count differing elements.

    Finite values match when abs(a - b) <= tol + rtol * abs(b), with b as
    the reference. NaN always counts as a difference, including two NaNs at
    the same position. Infinities match only when they have the same sign;
    finite/infinite pairs and opposite-sign infinities always differ.
    Comparisons run in bounded chunks. Relative floating comparisons use at
    least float64, with exact arithmetic for finite overflow candidates. Integral
    pairs use vectorized overflow-safe comparisons, with exact host arithmetic
    for precision-sensitive relative candidates, including mixed real/integral
    pairs. Mixed complex/integral pairs use exact host arithmetic when rtol is
    enabled. Widened comparisons run on host for backends other than CPU and CUDA.

    Args:
        a (torch.Tensor): The first tensor to compare.
        b (torch.Tensor): The reference tensor to compare against.
        tol (float, optional): The absolute tolerance threshold. Defaults to 1e-6.
        verbose (bool, optional): If True, log the indices and values of differing elements. Defaults to False.
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

    a_integral = not (a.is_floating_point() or a.is_complex())
    b_integral = not (b.is_floating_point() or b.is_complex())
    exact_integral = (a_integral and b_integral) or (rtol != 0.0 and (a_integral or b_integral))
    # Small CPU chunks avoid thread-pool overhead; larger CUDA chunks amortize launches.
    chunk_size = 262144 if a.device.type == "cuda" else 16384
    total = torch.zeros((), dtype=torch.int64, device=a.device)
    offset = 0
    for actual, reference in _iter_tensor_chunks(a, b, chunk_size):
        diff_mask = _chunk_difference_mask(actual, reference, tol, rtol)
        total.add_(diff_mask.sum())
        indices = torch.nonzero(diff_mask, as_tuple=False).flatten().tolist() if verbose else ()
        for index in indices:
            flat_index = offset + index
            reversed_coordinate = []
            for size in reversed(a.shape):
                reversed_coordinate.append(flat_index % size)
                flat_index //= size
            coordinate = tuple(reversed(reversed_coordinate))
            coordinate_text = ",".join(str(value) for value in coordinate)
            a_value = actual[index].item()
            b_value = reference[index].item()
            if exact_integral and cmath.isfinite(a_value) and cmath.isfinite(b_value):
                if isinstance(a_value, complex) or isinstance(b_value, complex):
                    real_delta = Fraction(a_value.real) - Fraction(b_value.real)
                    imag_delta = Fraction(a_value.imag) - Fraction(b_value.imag)
                    imag_sign = "-" if imag_delta < 0 else "+"
                    difference_text = f"{_format_scalar(real_delta)}{imag_sign}{_format_scalar(abs(imag_delta))}j"
                else:
                    difference_text = _format_scalar(Fraction(a_value) - Fraction(b_value))
            else:
                difference_text = _format_scalar(a_value - b_value)
            logger.info(
                f"diff at {coordinate_text}: {_format_scalar(a_value)} - {_format_scalar(b_value)} = {difference_text}"
            )
        offset += actual.numel()

    return int(total.item())


if __name__ == "__main__":
    # example:
    # a = torch.load("/path/to/a.pt")
    # b = torch.load("/path/to/b.pt")
    # diff_count = compare_tensors(a, b)
    # print(f"diff count: {diff_count}")
    pass
