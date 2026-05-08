# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from nose2.tools import params
from nose_utils import assert_raises
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms

from nvidia.dali.experimental.torchvision import (
    Compose,
    Grayscale,
    RandomApply,
    RandomHorizontalFlip,
)


def verify_non_one_off(t1: torch.Tensor, t2: torch.Tensor):
    if t1.dtype == torch.uint8:
        t1 = t1.to(torch.int16)
        t2 = t2.to(torch.int16)

    diff = (t1 - t2).abs()
    more_than_one_mask = diff > 1

    return more_than_one_mask.sum().item() == 0


dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]


@params("cpu", "gpu")
def test_random_apply_p1(device):
    """p=1.0: transformations always applied — output must match torchvision."""
    td = Compose(
        [RandomApply([Grayscale(num_output_channels=3, device=device)], p=1.0, device=device)]
    )
    t = transforms.Compose(
        [transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=1.0)]
    )

    for fn in test_files:
        img = Image.open(fn)
        out_tv = transforms.functional.pil_to_tensor(t(img))
        out_dali = transforms.functional.pil_to_tensor(td(img))
        assert verify_non_one_off(out_tv, out_dali), f"Images differ: {fn}"


@params("cpu", "gpu")
def test_random_apply_p0(device):
    """p=0.0: transformations never applied — output must equal input."""
    td = Compose(
        [RandomApply([Grayscale(num_output_channels=3, device=device)], p=0.0, device=device)]
    )
    t = transforms.Compose(
        [transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.0)]
    )

    for fn in test_files:
        img = Image.open(fn)
        out_tv = transforms.functional.pil_to_tensor(t(img))
        out_dali = transforms.functional.pil_to_tensor(td(img))
        assert verify_non_one_off(out_tv, out_dali), f"Images differ: {fn}"


@params(-0.1, 2.0, [0.0, 0.8])
def test_invalid_random_apply_probability(p):
    with assert_raises(ValueError, regex="p should be a floating point value in the interval"):
        RandomApply([Grayscale(num_output_channels=3)], p=p)


@params("cpu", "gpu")
def test_random_apply_multi_ops(device):
    """p=1.0 with multiple operators — all applied in sequence."""
    td = Compose(
        [
            RandomApply(
                [
                    RandomHorizontalFlip(p=1.0, device=device),
                    Grayscale(num_output_channels=3, device=device),
                ],
                p=1.0,
                device=device,
            )
        ]
    )
    t = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.Grayscale(num_output_channels=3),
                ],
                p=1.0,
            )
        ]
    )

    for fn in test_files:
        img = Image.open(fn)
        out_tv = transforms.functional.pil_to_tensor(t(img))
        out_dali = transforms.functional.pil_to_tensor(td(img))
        assert verify_non_one_off(out_tv, out_dali), f"Images differ: {fn}"


@params("cpu", "gpu")
def test_random_apply_preserves_shape(device):
    """Output shape must match input shape regardless of p."""
    td_apply = Compose(
        [RandomApply([RandomHorizontalFlip(p=1.0, device=device)], p=1.0, device=device)]
    )
    td_skip = Compose(
        [RandomApply([RandomHorizontalFlip(p=1.0, device=device)], p=0.0, device=device)]
    )

    for fn in test_files:
        img = Image.open(fn)
        out_apply = td_apply(img)
        out_skip = td_skip(img)
        assert out_apply.size == img.size, f"Shape mismatch after apply: {fn}"
        assert out_skip.size == img.size, f"Shape mismatch after skip: {fn}"
