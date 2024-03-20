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

import numpy as np

class RoadEdgeDetector:

  def __init__(self):
    self._is_enabled = False

  def get_road_edge_detected(self, road_edge_stds, lane_line_probs, left_blinker, right_blinker):
    if not self._is_enabled:
      return False, False

    # Calculate probabiities for detecting road edges and lane lines
    left_road_edge_prob = np.clip(1.0 - road_edge_stds[0], 0.0, 1.0)
    left_lane_nearside_prob = lane_line_probs[0]

    right_road_edge_prob = np.clip(1.0 - road_edge_stds[1], 0.0, 1.0)
    right_lane_nearside_prob = lane_line_probs[3]

    # Check conditions for detecting road edges
    if right_blinker and right_road_edge_prob > 0.35 and right_lane_nearside_prob < 0.2 and left_lane_nearside_prob >= right_lane_nearside_prob:
      return False, True
    if left_blinker and left_road_edge_prob > 0.35 and left_lane_nearside_prob < 0.2 and right_lane_nearside_prob >= left_lane_nearside_prob:
      return True, False

    return False, False

  def set_enabled(self, enabled):
    self._is_enabled = enabled

  def is_enabled(self):
    return self._is_enabled
