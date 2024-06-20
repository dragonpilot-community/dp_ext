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

from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.common.params import Params


GearShifter = car.CarState.GearShifter
UNLOCK_CMD = b'\x40\x05\x30\x11\x00\x40\x00\x00'
LOCK_CMD = b'\x40\x05\x30\x11\x00\x80\x00\x00'
LOCK_AT_SPEED = 10 * CV.KPH_TO_MS


class DoorLockController:

    def __init__(self):
        params = Params()
        self._dp_toyota_auto_lock = params.get_bool("dp_toyota_auto_lock")
        self._dp_toyota_auto_unlock = params.get_bool("dp_toyota_auto_unlock")
        self._dp_toyota_auto_lock_once = False
        self._gear_prev = GearShifter.park


    def process(self, gear, v_ego, door_open):
        # dp - door auto lock / unlock logic
        # thanks to AlexandreSato & cydia2020
        # https://github.com/AlexandreSato/animalpilot/blob/personal/doors.py
        message = []
        if not door_open:
            if gear == GearShifter.park and gear != self._gear_prev:
                if self._dp_toyota_auto_unlock:
                    message = [0x750, UNLOCK_CMD, 0]
                self._dp_toyota_auto_lock_once = False
            elif gear == GearShifter.drive and not self._dp_toyota_auto_lock_once and v_ego >= LOCK_AT_SPEED:
                if self._dp_toyota_auto_lock:
                    message = [0x750, LOCK_CMD, 0]
                self._dp_toyota_auto_lock_once = True
            self._gear_prev = gear
        return message