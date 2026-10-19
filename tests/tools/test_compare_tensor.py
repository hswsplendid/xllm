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
    a = torch.tensor([1_000_000.25], dtype=torch.float64)
    b = torch.tensor([1_000_000.0], dtype=torch.float64)

    # A nonzero default rtol could hide this absolute error at a large reference.
    assert compare_tensor.compare_tensors(a, b) == 1
    assert compare_tensor.compare_tensors(a, b, tol=1e-6, rtol=0.0) == 1


def test_legacy_positional_tolerance_and_verbose_are_preserved() -> None:
    a = torch.zeros(2, dtype=torch.float64)
    b = torch.tensor([0.125, 0.25], dtype=torch.float64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, 0.125, True)

    assert type(count) is int
    assert count == 1
    log_info.assert_called_once_with("diff at 1: 0.000000 - 0.250000 = -0.250000")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_combined_tolerance_includes_boundary_and_uses_absolute_reference(dtype: torch.dtype) -> None:
    # Binary fractions keep below/equal/above cases exact in every tested dtype.
    a = torch.tensor([8.75, 9.0, 9.25, -8.75, -9.0, -9.25, 0.25, 0.5, 0.75], dtype=dtype)
    b = torch.tensor([8.0, 8.0, 8.0, -8.0, -8.0, -8.0, 0.0, 0.0, 0.0], dtype=dtype)

    count = compare_tensor.compare_tensors(a, b, tol=0.5, rtol=0.0625)

    assert type(count) is int
    assert count == 3


def test_relative_tolerance_can_be_used_without_absolute_tolerance() -> None:
    a = torch.tensor([9.0, -9.0, 0.125], dtype=torch.float64)
    b = torch.tensor([8.0, -8.0, 0.0], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b, tol=0.0, rtol=0.125) == 1


def test_second_tensor_is_the_relative_tolerance_reference() -> None:
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
    a = torch.tensor([a_value], dtype=torch.float64)
    b = torch.tensor([b_value], dtype=torch.float64)

    count = compare_tensor.compare_tensors(a, b, rtol=rtol)

    assert type(count) is int
    assert count == expected_count


@pytest.mark.parametrize("verbose", [False, True])
def test_default_non_finite_policy_counts_and_logs_only_differences(verbose: bool) -> None:
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
