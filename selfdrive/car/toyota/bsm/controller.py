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

# code originally by:
# arne182: https://github.com/arne182
# kumar:   https://github.com/rav4kumar

# rewrite for easier integration hence the license declaration above.

from openpilot.selfdrive.car import make_can_msg
from openpilot.selfdrive.car.toyota.values import TSS2_CAR
from openpilot.dp_ext.selfdrive.car.toyota.bsm.common import ENABLED

LEFT_BLIND_SPOT = b'\x41'
RIGHT_BLIND_SPOT = b'\x42'

class BSMController:
    def __init__(self, CP):
        self._enabled = ENABLED
        self._blind_spot_debug_enabled_left = False
        self._blind_spot_debug_enabled_right = False
        self._blind_spot_frame = 0
        if CP.carFingerprint in TSS2_CAR: # tss2 can do higher hz then tss1 and can be on at all speed/standstill
            self._blind_spot_rate = 2
            self._blind_spot_always_on = True
        else:
            self._blind_spot_rate = 20
            self._blind_spot_always_on = False


    def _create_debug_msg(self, lr, enable):
        if enable:
            m = lr + b'\x02\x10\x60\x00\x00\x00\x00'
        else:
            m = lr + b'\x02\x10\x01\x00\x00\x00\x00'
        return make_can_msg(0x750, m, 0)

    def _create_poll_status_msg(self, lr):
        m = lr + b'\x02\x21\x69\x00\x00\x00\x00'
        return make_can_msg(0x750, m, 0)

    def get_can_sends(self, frame, v_ego, can_sends):
        if not self._enabled or frame <= 200:
            return can_sends

        def handle_blindspot(side, enabled, frame_diff, rate_divisor):
            if not enabled:
                if self._blind_spot_always_on or v_ego > 6: # eagle eye camera will stop working if right bsm is switched on under 6m/s
                    can_sends.append(self._create_debug_msg(side, True))
                    return True
            else:
                if not self._blind_spot_always_on and frame - frame_diff > 50:
                    can_sends.append(self._create_debug_msg(side, False))
                    return False
                if frame % self._blind_spot_rate == rate_divisor:
                    can_sends.append(self._create_poll_status_msg(side))
                    self._blind_spot_frame = frame
            return enabled

        # left bsm
        self._blind_spot_debug_enabled_left = handle_blindspot(
            LEFT_BLIND_SPOT,
            self._blind_spot_debug_enabled_left,
            self._blind_spot_frame,
            0
        )

        # right bsm
        self._blind_spot_debug_enabled_right = handle_blindspot(
            RIGHT_BLIND_SPOT,
            self._blind_spot_debug_enabled_right,
            self._blind_spot_frame,
            self._blind_spot_rate / 2
        )

        return can_sends