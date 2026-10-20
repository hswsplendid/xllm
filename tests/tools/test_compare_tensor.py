# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU regression tests for tensor tolerances, non-finite values and verbose output."""

import math
from fractions import Fraction
from unittest.mock import patch

import pytest
import torch

from tools import compare_tensor


@pytest.mark.parametrize(
    ("shape", "coordinate", "expected_coordinate"),
    [
        pytest.param((3,), (2,), "2", id="1d"),
        pytest.param((2, 3), (1, 2), "1,2", id="2d"),
        pytest.param((2, 3, 4), (1, 2, 3), "1,2,3", id="3d"),
        pytest.param((2, 2, 3, 4), (1, 1, 2, 3), "1,1,2,3", id="4d"),
    ],
)
def test_verbose_reports_complete_coordinates_and_values(
    shape: tuple[int, ...], coordinate: tuple[int, ...], expected_coordinate: str
) -> None:
    a = torch.zeros(shape, dtype=torch.float32)
    b = a.clone()
    a[coordinate] = 1.5
    b[coordinate] = 7.25

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 1
    log_info.assert_called_once_with(f"diff at {expected_coordinate}: 1.500000 - 7.250000 = -5.750000")


def test_verbose_reports_every_difference_with_its_own_values() -> None:
    a = torch.zeros((2, 3, 4))
    b = a.clone()
    a[0, 1, 2] = 2.0
    b[0, 1, 2] = 0.5
    a[1, 2, 3] = -1.0
    b[1, 2, 3] = 3.0

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 2
    assert log_info.call_count == 2
    assert {entry.args[0] for entry in log_info.call_args_list} == {
        "diff at 0,1,2: 2.000000 - 0.500000 = 1.500000",
        "diff at 1,2,3: -1.000000 - 3.000000 = -4.000000",
    }


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_does_not_change_difference_count(verbose: bool) -> None:
    a = torch.tensor([0.0, 1.0, 2.0, 3.0])
    b = torch.tensor([0.0, 4.0, 2.0, 8.0])

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=verbose)

    assert type(count) is int
    assert count == 2
    assert log_info.call_count == (2 if verbose else 0)


def test_default_comparison_does_not_log_differences() -> None:
    a = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    b = torch.tensor([[4.0, 1.0], [2.0, 5.0]])

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b)

    assert type(count) is int
    assert count == 2
    log_info.assert_not_called()


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)], ids=["1d", "2d", "3d"])
def test_equal_tensors_return_zero_without_logging(shape: tuple[int, ...]) -> None:
    a = torch.ones(shape)
    b = a.clone()

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 0
    log_info.assert_not_called()


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        pytest.param((2,), (3,), id="different-sizes"),
        pytest.param((2, 3), (3,), id="broadcastable-shapes"),
    ],
)
def test_shape_mismatch_is_rejected(shape_a: tuple[int, ...], shape_b: tuple[int, ...], verbose: bool) -> None:
    a = torch.zeros(shape_a)
    b = torch.zeros(shape_b)

    with (
        patch.object(compare_tensor.logger, "info") as log_info,
        pytest.raises(ValueError, match="Shape mismatch") as error,
    ):
        compare_tensor.compare_tensors(a, b, verbose=verbose)

    assert str(a.shape) in str(error.value)
    assert str(b.shape) in str(error.value)
    log_info.assert_not_called()


def test_only_differences_strictly_above_tolerance_are_logged() -> None:
    a = torch.zeros(3, dtype=torch.float64)
    b = torch.tensor([0.0625, 0.125, 0.25], dtype=torch.float64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.125, verbose=True)

    assert count == 1
    log_info.assert_called_once_with("diff at 2: 0.000000 - 0.250000 = -0.250000")


def test_default_tolerance_is_preserved() -> None:
    a = torch.zeros(3, dtype=torch.float64)
    b = torch.tensor([0.0, 0.5e-6, 2e-6], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b) == 1


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)], ids=["empty-1d", "empty-3d"])
def test_empty_tensors_return_zero_without_logging(shape: tuple[int, ...]) -> None:
    a = torch.empty(shape)
    b = torch.empty(shape)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 0
    log_info.assert_not_called()


def test_verbose_supports_non_contiguous_tensors_without_modifying_inputs() -> None:
    """Log complete coordinates for non-contiguous inputs without modifying them."""
    a = torch.zeros((2, 3, 4)).transpose(0, 2)
    b = a.clone()
    a[3, 2, 1] = 1.5
    b[3, 2, 1] = 7.25
    original_a = a.clone()
    original_b = b.clone()
    assert not a.is_contiguous()
    assert not b.is_contiguous()

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert count == 1
    log_info.assert_called_once_with("diff at 3,2,1: 1.500000 - 7.250000 = -5.750000")
    assert torch.equal(a, original_a)
    assert torch.equal(b, original_b)


def test_default_relative_tolerance_is_zero() -> None:
    """Keep relative tolerance disabled unless the caller explicitly enables it."""
    a = torch.tensor([1_000_000.25], dtype=torch.float64)
    b = torch.tensor([1_000_000.0], dtype=torch.float64)

    # A nonzero default rtol could hide this absolute error at a large reference.
    assert compare_tensor.compare_tensors(a, b) == 1
    assert compare_tensor.compare_tensors(a, b, tol=1e-6, rtol=0.0) == 1


def test_legacy_positional_tolerance_and_verbose_are_preserved() -> None:
    """Preserve the original positional tolerance and verbose arguments."""
    a = torch.zeros(2, dtype=torch.float64)
    b = torch.tensor([0.125, 0.25], dtype=torch.float64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, 0.125, True)

    assert type(count) is int
    assert count == 1
    log_info.assert_called_once_with("diff at 1: 0.000000 - 0.250000 = -0.250000")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_combined_tolerance_includes_boundary_and_uses_absolute_reference(dtype: torch.dtype) -> None:
    """Accept equality at the combined threshold for positive, negative and zero references."""
    # Binary fractions keep below/equal/above cases exact in every tested dtype.
    a = torch.tensor([8.75, 9.0, 9.25, -8.75, -9.0, -9.25, 0.25, 0.5, 0.75], dtype=dtype)
    b = torch.tensor([8.0, 8.0, 8.0, -8.0, -8.0, -8.0, 0.0, 0.0, 0.0], dtype=dtype)

    count = compare_tensor.compare_tensors(a, b, tol=0.5, rtol=0.0625)

    assert type(count) is int
    assert count == 3


def test_relative_tolerance_can_be_used_without_absolute_tolerance() -> None:
    """Apply a purely relative threshold while still detecting errors against zero."""
    a = torch.tensor([9.0, -9.0, 0.125], dtype=torch.float64)
    b = torch.tensor([8.0, -8.0, 0.0], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=0.125) == 1


def test_second_tensor_is_the_relative_tolerance_reference() -> None:
    """Use the second tensor as the reference for asymmetric relative comparisons."""
    a = torch.tensor([1.0], dtype=torch.float64)
    b = torch.tensor([2.0], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=0.5) == 0
    assert compare_tensor.compare_tensors(b, a, tol=0.0, rtol=0.5) == 1


@pytest.mark.parametrize("rtol", [0.0, 0.5], ids=["absolute-only", "relative-enabled"])
@pytest.mark.parametrize(
    ("a_value", "b_value", "expected_count"),
    [
        pytest.param(float("nan"), float("nan"), 1, id="both-nan"),
        pytest.param(float("nan"), 1.0, 1, id="nan-finite"),
        pytest.param(1.0, float("nan"), 1, id="finite-nan"),
        pytest.param(float("nan"), float("inf"), 1, id="nan-positive-inf"),
        pytest.param(float("inf"), float("nan"), 1, id="positive-inf-nan"),
        pytest.param(float("nan"), -float("inf"), 1, id="nan-negative-inf"),
        pytest.param(-float("inf"), float("nan"), 1, id="negative-inf-nan"),
        pytest.param(float("inf"), float("inf"), 0, id="same-positive-inf"),
        pytest.param(-float("inf"), -float("inf"), 0, id="same-negative-inf"),
        pytest.param(float("inf"), -float("inf"), 1, id="positive-negative-inf"),
        pytest.param(-float("inf"), float("inf"), 1, id="negative-positive-inf"),
        pytest.param(1.0, float("inf"), 1, id="finite-positive-inf"),
        pytest.param(float("inf"), 1.0, 1, id="positive-inf-finite"),
        pytest.param(1.0, -float("inf"), 1, id="finite-negative-inf"),
        pytest.param(-float("inf"), 1.0, 1, id="negative-inf-finite"),
    ],
)
def test_non_finite_comparison_policy(a_value: float, b_value: float, expected_count: int, rtol: float) -> None:
    """Count NaNs and unequal infinities independently of the relative tolerance."""
    a = torch.tensor([a_value], dtype=torch.float64)
    b = torch.tensor([b_value], dtype=torch.float64)

    count = compare_tensor.compare_tensors(a, b, rtol=rtol)

    assert type(count) is int
    assert count == expected_count


@pytest.mark.parametrize("verbose", [False, True])
def test_default_non_finite_policy_counts_and_logs_only_differences(verbose: bool) -> None:
    """Count and optionally log only mismatches under the default non-finite policy."""
    a = torch.tensor([[float("nan"), float("inf")], [-float("inf"), 1.0]])
    b = torch.tensor([[float("nan"), float("inf")], [float("inf"), float("nan")]])

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=verbose)

    assert type(count) is int
    assert count == 3
    assert log_info.call_count == (3 if verbose else 0)
    if verbose:
        assert {entry.args[0] for entry in log_info.call_args_list} == {
            "diff at 0,0: nan - nan = nan",
            "diff at 1,0: -inf - inf = -inf",
            "diff at 1,1: 1.000000 - nan = nan",
        }


def test_relative_and_non_finite_differences_are_counted_together_without_modifying_inputs() -> None:
    """Combine relative and non-finite mismatches without changing either input tensor."""
    a = torch.tensor([[[9.0, 9.25, float("nan"), float("inf"), 1.0, -float("inf")]]], dtype=torch.float64)
    b = torch.tensor([[[8.0, 8.0, float("nan"), float("inf"), float("inf"), float("inf")]]], dtype=torch.float64)
    original_a = a.clone()
    original_b = b.clone()

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.5, verbose=True, rtol=0.0625)

    assert type(count) is int
    assert count == 4
    assert {entry.args[0] for entry in log_info.call_args_list} == {
        "diff at 0,0,1: 9.250000 - 8.000000 = 1.250000",
        "diff at 0,0,2: nan - nan = nan",
        "diff at 0,0,4: 1.000000 - inf = -inf",
        "diff at 0,0,5: -inf - inf = -inf",
    }
    torch.testing.assert_close(a, original_a, rtol=0.0, atol=0.0, equal_nan=True)
    torch.testing.assert_close(b, original_b, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("reference_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mixed_dtypes_preserve_default_absolute_tolerance(reference_dtype: torch.dtype) -> None:
    """Preserve the absolute threshold when the reference has a lower precision."""
    a = torch.tensor([0.999e-6, 1e-6, 1.001e-6], dtype=torch.float64)
    b = torch.zeros(3, dtype=reference_dtype)

    assert compare_tensor.compare_tensors(a, b) == 1
    assert compare_tensor.compare_tensors(a, b, tol=1e-6, rtol=0.0) == 1


@pytest.mark.parametrize("reference_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mixed_dtypes_compute_relative_tolerance_in_promoted_dtype(reference_dtype: torch.dtype) -> None:
    """Retain small absolute contributions in mixed-dtype relative thresholds."""
    # Keep the tiny absolute contribution when adding it to the relative term.
    a = torch.tensor([8.5 + 2**-31, 8.5 + 2**-30, 8.5 + 2**-29], dtype=torch.float64)
    b = torch.full((3,), 8.0, dtype=reference_dtype)

    assert compare_tensor.compare_tensors(a, b, tol=2**-30, rtol=0.0625) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "reference_value",
    [1_000_000, -1_000_000, torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max],
    ids=["positive-large", "negative-large", "int64-min", "int64-max"],
)
@pytest.mark.parametrize(
    ("rtol", "expected_count"),
    [(1e-6, 1), (0.5, 1), (1.0, 0), (2.0, 0)],
    ids=["small-tolerance", "below-boundary", "at-boundary", "above-boundary"],
)
def test_large_integral_references_use_finite_relative_thresholds(
    dtype: torch.dtype, reference_value: int, rtol: float, expected_count: int
) -> None:
    """Compare large signed integers without overflowing the difference or threshold."""
    a = torch.zeros(1, dtype=dtype)
    b = torch.tensor([reference_value], dtype=torch.int64)
    original_a = a.clone()
    original_b = b.clone()

    count = compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol)

    assert type(count) is int
    assert count == expected_count
    assert torch.equal(a, original_a)
    assert torch.equal(b, original_b)


def test_verbose_reports_finite_differences_for_large_integral_references() -> None:
    """Log promoted finite differences for every large integral reference mismatch."""
    a = torch.zeros(2, dtype=torch.float16)
    b = torch.tensor([1_000_000, -1_000_000], dtype=torch.int64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True, rtol=1e-6)

    assert count == 2
    assert [entry.args[0] for entry in log_info.call_args_list] == [
        "diff at 0: 0.000000 - 1000000.000000 = -1000000.000000",
        "diff at 1: 0.000000 - -1000000.000000 = 1000000.000000",
    ]


@pytest.mark.parametrize("rtol", [0.0, 1e-6, 2.0])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_dtype_infinities_never_match_large_finite_integers(rtol: float, sign: int, reverse: bool) -> None:
    """Classify infinity before promotion can turn a finite integral value into infinity."""
    a = torch.tensor([sign * float("inf")], dtype=torch.float16)
    b = torch.tensor([sign * 1_000_000], dtype=torch.int64)
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, rtol=rtol) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(("rtol", "expected_count"), [(1.5, 2), (2.0, 0), (2.5, 0)])
def test_relative_comparison_does_not_overflow_at_floating_limits(
    dtype: torch.dtype, rtol: float, expected_count: int
) -> None:
    """Compare opposite finite extrema below, at and above the exact relative boundary."""
    largest = torch.finfo(dtype).max
    a = torch.tensor([largest, -largest], dtype=dtype)
    b = -a

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == expected_count


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("reverse", [False, True])
def test_integral_subtraction_preserves_unit_deltas_near_dtype_limits(dtype: torch.dtype, reverse: bool) -> None:
    """Retain unit differences at signed integer limits without rounding operands."""
    limits = torch.iinfo(dtype)
    a = torch.tensor([limits.min, limits.max], dtype=dtype)
    b = torch.tensor([limits.min + 1, limits.max - 1], dtype=dtype)
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=2.0 ** -(limits.bits + 1)) == 2


@pytest.mark.parametrize("rtol", [0.0, 2.0**-54])
def test_integral_operands_preserve_exact_deltas_above_float64_precision(rtol: float) -> None:
    """Detect one-unit errors above 2**53 when the absolute or relative threshold is smaller."""
    a = torch.tensor([2**53, -(2**53)], dtype=torch.int64)
    b = torch.tensor([2**53 + 1, -(2**53 + 1)], dtype=torch.int64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == 2


@pytest.mark.parametrize(("rtol", "expected_count"), [(0.5, 2), (1.0, 2), (2.0, 1), (3.0, 0)])
def test_integral_comparison_avoids_subtraction_and_absolute_value_overflow(rtol: float, expected_count: int) -> None:
    """Compare full-range signed deltas and thresholds without int64 or float64 overflow."""
    limits = torch.iinfo(torch.int64)
    a = torch.tensor([limits.min, limits.max], dtype=torch.int64)
    b = torch.tensor([limits.max, limits.min], dtype=torch.int64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == expected_count


@pytest.mark.parametrize(("rtol", "expected_count"), [(2.0**-54, 1), (2.0**-53, 0)])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_float_and_integer_operands_preserve_small_exact_deltas(
    rtol: float, expected_count: int, reverse: bool
) -> None:
    """Keep integer precision when the other operand is a nearby exactly represented float."""
    a = torch.tensor([2**53], dtype=torch.float64)
    b = torch.tensor([2**53 + 1], dtype=torch.int64)
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == expected_count


def test_integral_relative_boundary_is_exact_for_large_odd_reference() -> None:
    """Distinguish exact equality from one unit above an unrounded integral threshold."""
    largest = torch.iinfo(torch.int64).max
    a = torch.tensor([0, -1], dtype=torch.int64)
    b = torch.tensor([largest, largest], dtype=torch.int64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1.0) == 1


def test_verbose_preserves_exact_integral_deltas_with_non_contiguous_inputs() -> None:
    """Report exact unit deltas for non-contiguous integer and mixed inputs without mutations."""
    a = torch.full((2, 3), 2**53, dtype=torch.int64).transpose(0, 1)
    b = a.clone()
    b[2, 1] += 1
    original_a = a.clone()
    original_b = b.clone()

    for first in [a, a.to(dtype=torch.float64)]:
        with patch.object(compare_tensor.logger, "info") as log_info:
            count = compare_tensor.compare_tensors(first, b, tol=0.0, verbose=True, rtol=2.0**-54)

        assert count == 1
        log_info.assert_called_once()
        assert log_info.call_args.args[0].endswith("= -1.000000")

    assert torch.equal(a, original_a)
    assert torch.equal(b, original_b)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(("rtol", "expected_count"), [(0.5, 1), (1.0, 0), (1.5, 0)])
def test_relative_comparison_does_not_overflow_complex_magnitudes(
    dtype: torch.dtype, rtol: float, expected_count: int
) -> None:
    """Rescale complex magnitudes when finite components would produce infinite absolute values."""
    largest = torch.finfo(dtype).max
    a = torch.zeros(1, dtype=dtype)
    b = torch.tensor([complex(largest, largest)], dtype=dtype)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == expected_count


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)], ids=["scalar", "empty-1d", "empty-3d"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_scalar_and_empty_tensors_support_relative_arithmetic(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Preserve scalar and empty shapes in exact integral and overflow-safe floating comparisons."""
    value = torch.iinfo(dtype).max if dtype == torch.int64 else torch.finfo(dtype).max
    a = torch.full(shape, value, dtype=dtype)
    b = -a
    expected = 1 if shape == () else 0

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1.5) == expected


@pytest.mark.parametrize("rtol", [0.0, 1e-6])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bool])
def test_unsigned_and_boolean_comparisons_keep_integral_deltas(dtype: torch.dtype, rtol: float) -> None:
    """Subtract unsigned and boolean values without wrapping or unsupported tensor subtraction."""
    largest = torch.iinfo(dtype).max if dtype != torch.bool else True
    a = torch.tensor([0, largest], dtype=dtype)
    b = torch.tensor([largest, 0], dtype=dtype)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == 2


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
@pytest.mark.parametrize(("rtol", "expected_count"), [(2.0**-54, 1), (2.0**-52, 0)])
def test_complex_and_integral_operands_preserve_exact_deltas(
    reverse: bool, imaginary: float, rtol: float, expected_count: int
) -> None:
    """Preserve both components of a complex delta when mixed promotion would round its integer part."""
    a = torch.tensor([complex(2**53, imaginary)], dtype=torch.complex128)
    b = torch.tensor([2**53 + 1], dtype=torch.int64)
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == expected_count


@pytest.mark.parametrize(
    ("tol", "rtol", "expected_count"),
    [(0.0, 1.0, 1), (1.0, 1.0, 0), (2.0, 1.0, 0), (3.0, 1.0, 0), (-2.0, 2.0, 0), (2.0, -1.0, 1)],
)
def test_complex_integral_threshold_preserves_magnitude_and_boundary(
    tol: float, rtol: float, expected_count: int
) -> None:
    """Compare complex magnitudes against combined tolerances without changing legacy signed thresholds."""
    a = torch.tensor([3 + 4j], dtype=torch.complex128)
    b = torch.tensor([3], dtype=torch.int64)

    assert compare_tensor.compare_tensors(a, b, tol=tol, rtol=rtol) == expected_count


def test_verbose_reports_exact_complex_integral_delta() -> None:
    """Log the retained real and imaginary differences for complex/integer comparisons."""
    a = torch.tensor([complex(2**53, 1.0)], dtype=torch.complex128)
    b = torch.tensor([2**53 + 1], dtype=torch.int64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.0, rtol=2.0**-54, verbose=True)

    assert count == 1
    log_info.assert_called_once()
    assert log_info.call_args.args[0].endswith("= -1.000000+1.000000j")


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_large_integral_comparisons_count_all_differences_without_mutation(dtype: torch.dtype) -> None:
    """Retain exact mismatches throughout large dumps without modifying either operand."""
    a = torch.full((20000,), 2**53, dtype=dtype)
    b = torch.full((20000,), 2**53, dtype=torch.int64)
    b[0] += 1
    b[16000] += 1
    b[19999] += 1
    original_a, original_b = a.clone(), b.clone()

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=2.0**-54) == 3
    assert torch.equal(a, original_a)
    assert torch.equal(b, original_b)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize(("actual", "expected_count"), [(0.0, 1), (8e307, 0), (1e308, 0)])
def test_relative_threshold_overflow_does_not_hide_finite_cancellation(
    dtype: torch.dtype, actual: float, expected_count: int
) -> None:
    """Rescale overflowing relative terms before adding a cancelling absolute tolerance."""
    a = torch.tensor([actual], dtype=dtype)
    b = torch.tensor([1e308], dtype=dtype)

    # The exact threshold is about 5e307, although the relative term alone exceeds float64.
    assert compare_tensor.compare_tensors(a, b, tol=-1.5e308, rtol=2.0) == expected_count


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("tol", "rtol", "expected_forward", "expected_reverse"),
    [
        (float("inf"), 1.0, 0, 0),
        (-float("inf"), 1.0, 1, 1),
        (0.0, float("inf"), 0, 0),
        (0.0, -float("inf"), 0, 1),
        (float("nan"), 1.0, 0, 0),
        (0.0, float("nan"), 0, 0),
        (float("inf"), -float("inf"), 0, 0),
        (-float("inf"), float("inf"), 0, 0),
    ],
)
def test_complex_integral_extrema_support_non_finite_tolerances(
    reverse: bool, tol: float, rtol: float, expected_forward: int, expected_reverse: int
) -> None:
    """Classify infinite and NaN thresholds without overflowing finite complex magnitudes."""
    largest = torch.finfo(torch.float64).max
    a = torch.tensor([complex(largest, largest)], dtype=torch.complex128)
    b = torch.zeros(1, dtype=torch.int64)
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, tol=tol, rtol=rtol) == (
        expected_reverse if reverse else expected_forward
    )


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64, torch.complex128])
def test_verbose_preserves_large_integral_reference_values(dtype: torch.dtype) -> None:
    """Display the original integer reference instead of rounding it through float formatting."""
    a = torch.tensor([2**53], dtype=dtype)
    b = torch.tensor([2**53 + 1], dtype=torch.int64)
    actual_text = "9007199254740992.000000" + ("+0.000000j" if dtype.is_complex else "")
    delta_text = "-1.000000" + ("+0.000000j" if dtype.is_complex else "")

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.0, rtol=2.0**-54, verbose=True)

    assert count == 1
    log_info.assert_called_once_with(f"diff at 0: {actual_text} - 9007199254740993.000000 = {delta_text}")


@pytest.mark.parametrize(
    ("dtype", "actual", "actual_text", "delta_text"),
    [
        (torch.int64, -(2**63), "-9223372036854775808.000000", "-18446744073709551615.000000"),
        (torch.float64, 0.5, "0.500000", "-9223372036854775806.500000"),
        (torch.complex128, 0.5 + 0.25j, "0.500000+0.250000j", "-9223372036854775806.500000+0.250000j"),
    ],
)
def test_verbose_preserves_full_integral_and_mixed_deltas(
    dtype: torch.dtype, actual: int | float | complex, actual_text: str, delta_text: str
) -> None:
    """Keep integer units and fractional components when formatting deltas larger than 2**53."""
    a = torch.tensor([actual], dtype=dtype)
    b = torch.tensor([2**63 - 1], dtype=torch.int64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.0, rtol=2.0**-64, verbose=True)

    assert count == 1
    log_info.assert_called_once_with(f"diff at 0: {actual_text} - 9223372036854775807.000000 = {delta_text}")


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(("direction", "expected_count"), [(-1, 0), (0, 0), (1, 1)])
def test_overflow_comparison_preserves_exact_boundary_and_adjacent_values(
    dtype: torch.dtype, sign: int, direction: int, expected_count: int
) -> None:
    """Accept the exact binary tolerance equality and reject the next larger floating value."""
    magnitude = 1.7976931348623153e308
    if direction:
        magnitude = math.nextafter(magnitude, math.inf if direction > 0 else -math.inf)
    a = torch.tensor([sign * magnitude], dtype=dtype)
    b = torch.tensor([-sign * 1.7976931348623155e308], dtype=dtype)

    assert compare_tensor.compare_tensors(a, b, tol=8.988465674311576e307, rtol=1.5) == expected_count


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("actual", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("rtol", [math.nextafter(1.0, 0.0), 1.0, math.nextafter(1.0, math.inf)])
def test_float64_rounded_deltas_preserve_exact_relative_boundaries(sign: int, actual: float, rtol: float) -> None:
    """Retain finite discrepancies rounded away by float64 subtraction near a relative boundary."""
    actual *= sign
    reference = float(sign * 2**53)
    expected = int(abs(Fraction(actual) - Fraction(reference)) > Fraction(rtol) * abs(Fraction(reference)))
    a = torch.tensor([actual], dtype=torch.float64)
    b = torch.tensor([reference], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("actual", "reference", "tol", "rtol"),
    [
        (0.0, 3.0, 1.0, 2.0 / 3.0),
        (0.0, 1.0, math.nextafter(1.0, 0.0), 2.0**-54),
        (0.0, 1.0, math.nextafter(1.0, math.inf), -(2.0**-53)),
        (1.0, 1.0, -(2.0**-54), 2.0**-55),
    ],
    ids=["rounded-product", "rounded-addition", "negative-relative", "negative-threshold"],
)
def test_rounded_relative_thresholds_agree_with_exact_fractions(
    dtype: torch.dtype, actual: float, reference: float, tol: float, rtol: float
) -> None:
    """Account for multiplication and addition rounding even when low-precision deltas are exact."""
    expected = int(
        abs(Fraction(actual) - Fraction(reference)) > Fraction(tol) + Fraction(rtol) * abs(Fraction(reference))
    )
    a, b = torch.tensor([actual], dtype=dtype), torch.tensor([reference], dtype=dtype)

    assert compare_tensor.compare_tensors(a, b, tol=tol, rtol=rtol) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_floating_dtypes_preserve_deltas_below_float64_resolution(dtype: torch.dtype, reverse: bool) -> None:
    """Detect a unit discrepancy at a large reference across both floating operand orders."""
    a = torch.tensor([-1.0], dtype=dtype)
    b = torch.tensor([2.0**60], dtype=torch.float64)
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1.0) == 1


@pytest.mark.parametrize("sign", [-1, 1])
def test_subnormal_relative_boundary_uses_exact_values(sign: int) -> None:
    """Retain a discrepancy when an underflowing relative product rounds up to the exact delta."""
    smallest = math.ulp(0.0)
    a = torch.tensor([-sign * smallest], dtype=torch.float64)
    b = torch.tensor([sign * smallest], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1.5) == 1


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("rotation", [1.0, -1.0, 1j, -1j])
@pytest.mark.parametrize(("direction", "expected"), [(-1, 1), (0, 0), (1, 0)])
def test_complex_rounded_magnitudes_preserve_exact_relative_boundaries(
    dtype: torch.dtype, rotation: complex, direction: int, expected: int
) -> None:
    """Classify both sides of an exact complex norm boundary even when the rounded norms coincide."""
    a = torch.tensor([rotation * direction * 2.0**-80], dtype=dtype)
    b = torch.tensor([rotation * (1 + 1j)], dtype=dtype)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1.0) == expected


@pytest.mark.parametrize("rotation", [1.0, -1.0, 1j, -1j])
def test_complex_norm_rounding_does_not_report_spurious_differences(rotation: complex) -> None:
    """Accept an exact in-tolerance complex pair whose independently rounded norms suggest a difference."""
    actual = rotation * complex(-5.218365018639276e-31, 1.4574018762342288e-30)
    reference = rotation * complex(-2.0873460074557104e-31, 5.829607504936915e-31)
    delta_squared = (Fraction(actual.real) - Fraction(reference.real)) ** 2 + (
        Fraction(actual.imag) - Fraction(reference.imag)
    ) ** 2
    reference_squared = Fraction(reference.real) ** 2 + Fraction(reference.imag) ** 2
    assert delta_squared <= Fraction(1.5) ** 2 * reference_squared
    a, b = torch.tensor([actual], dtype=torch.complex128), torch.tensor([reference], dtype=torch.complex128)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1.5) == 0


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool])
@pytest.mark.parametrize(
    "tol", [0.0, 0.5, 1.0, 1.5, 2**63 - 1024, 2**63, 2**64 - 2048, -1.0, math.inf, -math.inf, math.nan]
)
def test_integral_absolute_tolerances_preserve_exact_full_range_counts(dtype: torch.dtype, tol: float) -> None:
    """Compare full-range integer differences and scalar thresholds without subtraction overflow."""
    low, high = (0, 1) if dtype == torch.bool else (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
    actual = [low, high, low, high, 0, 1, 0]
    reference = [high, low, low + 1, high - 1, 0, 0, 1]
    a, b = torch.tensor(actual, dtype=dtype), torch.tensor(reference, dtype=dtype)
    expected = sum(abs(first - second) > tol for first, second in zip(actual, reference))

    assert compare_tensor.compare_tensors(a, b, tol=tol) == expected


@pytest.mark.parametrize("dtype", [torch.int8, torch.int64, torch.uint64])
@pytest.mark.parametrize(
    ("tol", "rtol"),
    [(0.0, 1 - 2**-52), (0.0, 1.0), (0.0, 1 + 2**-52), (0.0, 2**-54), (1e-6, 0.1), (-0.5, 0.5), (0.5, -0.5)],
)
def test_integral_relative_thresholds_agree_with_exact_fraction_arithmetic(
    dtype: torch.dtype, tol: float, rtol: float
) -> None:
    """Cover exact and neighboring relative thresholds, signed tolerances and integer extremes."""
    low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    actual = [low, high, 0, 1, low, high, 0]
    reference = [high, low, high, 0, low + 1, high - 1, 0]
    a, b = torch.tensor(actual, dtype=dtype), torch.tensor(reference, dtype=dtype)
    expected = sum(
        abs(first - second) > Fraction(tol) + Fraction(rtol) * abs(second) for first, second in zip(actual, reference)
    )

    assert compare_tensor.compare_tensors(a, b, tol=tol, rtol=rtol) == expected


@pytest.mark.parametrize("rtol", [0.0, 1e-6])
def test_large_common_integral_comparisons_count_sparse_differences(rtol: float) -> None:
    """Compare large ordinary integer dumps with sparse errors using both tolerance modes."""
    a = torch.full((5_000_000,), 7, dtype=torch.int8)
    b = a.clone()
    b[0] = 8
    b[2_500_000] = 8
    b[-1] = 8

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol) == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_large_floating_comparisons_preserve_counts_across_chunks(dtype: torch.dtype) -> None:
    """Retain mismatches across a large low-precision tensor without modifying its contents."""
    a = torch.zeros((3, 400_000), dtype=dtype)
    b = a.clone()
    b[0, 0] = 1
    b[1, 150_000] = 1
    b[2, -1] = 1

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=0.5) == 3
    assert a.count_nonzero().item() == 0
    assert b.count_nonzero().item() == 3


@pytest.mark.parametrize("rtol", [0.0, 0.5])
def test_large_strided_inputs_keep_original_verbose_coordinates(rtol: float) -> None:
    """Retain logical coordinates when bounded chunks flatten inputs with different strides."""
    a = torch.zeros((3, 200_000), dtype=torch.float32).transpose(0, 1)
    b = a.contiguous()
    coordinates = [(0, 0), (100_000, 1), (199_999, 2)]
    for coordinate in coordinates:
        b[coordinate] = 1

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.0, rtol=rtol, verbose=True)

    assert count == 3
    assert {call.args[0] for call in log_info.call_args_list} == {
        f"diff at {row},{column}: 0.000000 - 1.000000 = -1.000000" for row, column in coordinates
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize("reverse", [False, True])
def test_large_mixed_real_integral_comparisons_preserve_sparse_differences(dtype: torch.dtype, reverse: bool) -> None:
    """Compare common mixed real/integer dumps without losing discrepancies to dtype promotion."""
    a = torch.full((1_000_000,), 7, dtype=dtype)
    b = torch.full((1_000_000,), 7, dtype=torch.int8)
    b[0] = 8
    b[500_000] = 8
    b[-1] = 8
    if reverse:
        a, b = b, a

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=1e-6) == 3


def test_recursive_strided_chunks_preserve_high_dimension_verbose_coordinates() -> None:
    """Keep full coordinates when even a single non-contiguous outer slice exceeds a chunk."""
    a = torch.zeros((2, 200_000, 3), dtype=torch.float16).transpose(1, 2)
    b = a.clone()
    b[0, 2, 150_000] = 1
    b[1, 1, 199_999] = 1

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.0, rtol=0.5, verbose=True)

    assert count == 2
    assert {call.args[0] for call in log_info.call_args_list} == {
        "diff at 0,2,150000: 0.000000 - 1.000000 = -1.000000",
        "diff at 1,1,199999: 0.000000 - 1.000000 = -1.000000",
    }


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("tol", "rtol"), [(0.0, 0.0), (1.5, 0.0), (0.0, 1e-6), (0.0, 0.5)])
def test_mixed_unsigned_and_signed_integers_preserve_exact_comparison(
    dtype: torch.dtype, reverse: bool, tol: float, rtol: float
) -> None:
    """Compare unsigned/signed pairs without relying on unsupported implicit dtype promotion."""
    largest = torch.iinfo(dtype).max
    actual, reference = [0, largest, 1, largest], [-1, largest, 0, largest - 1]
    a = torch.tensor(actual, dtype=dtype)
    b = torch.tensor(reference, dtype=torch.int64)
    if reverse:
        a, b = b, a
        actual, reference = reference, actual
    expected = sum(
        abs(first - second) > Fraction(tol) + Fraction(rtol) * abs(second) for first, second in zip(actual, reference)
    )

    assert compare_tensor.compare_tensors(a, b, tol=tol, rtol=rtol) == expected
