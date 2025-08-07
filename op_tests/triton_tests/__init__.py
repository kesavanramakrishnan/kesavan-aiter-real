# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import sys
TRITON_LIB_PATH = "/workspace/triton"
if TRITON_LIB_PATH not in sys.path:
    sys.path.append(TRITON_LIB_PATH)
from .utils import *  # noqa: F403

