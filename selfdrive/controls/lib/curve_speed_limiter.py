#!/usr/bin/env python3
# The MIT License
#
# Copyright (c) 2019-, Rick Lan, dragonpilot community, and a number of other of contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

from openpilot.common.params import Params
import numpy as np
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.numpy_fast import interp



class CurveSpeedLimiter:
    def __init__(self):
        self._enabled = Params().get_bool("dp_long_curve_speed_limiter")
        self._v = 0.
        pass

    def update(self, v_ego, z, v):
        max_lat_accel = interp(v_ego, [5, 10, 20], [1.5, 2.0, 3.0])
        curvatures = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, z) / np.clip(v, 0.3, 100.0)
        max_v = np.sqrt(max_lat_accel / (np.abs(curvatures) + 1e-3)) - 2.0
        self._v = np.minimum(max_v, v)

    def get_v(self, v):
        return self._v if self._enabled else v