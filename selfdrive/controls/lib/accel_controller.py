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

from openpilot.common.numpy_fast import interp

DP_ACCEL_STOCK = 0
DP_ACCEL_ECO = 1
DP_ACCEL_NORMAL = 2
DP_ACCEL_SPORT = 3

# accel profile by @arne182 modified by cgw
_DP_CRUISE_MIN_V =       [-0.02700, -.0001,  -.001, -.16,   -0.65]
_DP_CRUISE_MIN_V_ECO =   [-0.02700, -.0001,  -.001, -.14,   -0.60]
_DP_CRUISE_MIN_V_SPORT = [-0.02700, -.0001,  -.001, -.28,   -0.70]
_DP_CRUISE_MIN_BP =      [0.,       .003,    1.0,    2.0,    33.]
#_DP_CRUISE_MIN_V =       [-0.0279, -.000002, -.000004, -.0002, -0.07, -0.34, -0.70, -0.75]
#_DP_CRUISE_MIN_V_ECO =   [-0.0278, -.000001, -.000003, -.0001, -0.06, -0.32, -0.65, -0.70]
#_DP_CRUISE_MIN_V_SPORT = [-0.0280, -.000003, -.000005, -.0003, -0.08, -0.36, -0.75, -0.80]

#_DP_CRUISE_MIN_BP =      [0.,       .01,      .1,       1.,     5.5,   8.3,    28.,   42.]
_DP_CRUISE_MAX_V =       [3.5, 3.5, 3.4, 2.3, 1.9, 1.12,  .863, .64,  .42, .33, .12]
_DP_CRUISE_MAX_V_ECO =   [3.5, 3.5, 3.3, 1.9, 1.5, 0.977, .767, .554, .344, .29, .09]
_DP_CRUISE_MAX_V_SPORT = [3.5, 3.5, 3.5, 3.4, 2.8, 1.25,  1.0, .75,  .6,  .38, .2]
_DP_CRUISE_MAX_BP =      [0.,  1.,  3,   6.,  8.,  11.,   15.,  20.,  25., 30., 55.]

class AccelController:

  def __init__(self):
    # self._params = Params()
    self._profile = DP_ACCEL_STOCK

  def set_profile(self, profile):
    try:
      self._profile = int(profile) if int(profile) in [DP_ACCEL_STOCK, DP_ACCEL_ECO, DP_ACCEL_NORMAL, DP_ACCEL_SPORT] else DP_ACCEL_STOCK
    except:
      self._profile = DP_ACCEL_STOCK

  def _dp_calc_cruise_accel_limits(self, v_ego):
    if self._profile == DP_ACCEL_ECO:
      min_v = _DP_CRUISE_MIN_V_ECO
      max_v = _DP_CRUISE_MAX_V_ECO
    elif self._profile == DP_ACCEL_SPORT:
      min_v = _DP_CRUISE_MIN_V_SPORT
      max_v = _DP_CRUISE_MAX_V_SPORT
    else:
      min_v = _DP_CRUISE_MIN_V
      max_v = _DP_CRUISE_MAX_V

    a_cruise_min = interp(v_ego, _DP_CRUISE_MIN_BP, min_v)
    a_cruise_max = interp(v_ego, _DP_CRUISE_MAX_BP, max_v)
    return a_cruise_min, a_cruise_max

  def get_accel_limits(self, v_ego, accel_limits):
    return accel_limits if self._profile == DP_ACCEL_STOCK else self._dp_calc_cruise_accel_limits(v_ego)

  def is_enabled(self):
    return self._profile != DP_ACCEL_STOCK
