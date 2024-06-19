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

from openpilot.selfdrive.car.hyundai.values import CarControllerParams

class TacoCarControllerParams(CarControllerParams):

    # rick - re-implement taco mode, thanks to @stupefacient
    def update(self, v_ego_raw: float = 100.):
        if v_ego_raw < 11:
            self.STEER_MAX = 384
            self.STEER_DRIVER_ALLOWANCE = 350
            self.STEER_DRIVER_MULTIPLIER = 2
            self.STEER_THRESHOLD = 350
            self.STEER_DELTA_UP = 10
            self.STEER_DELTA_DOWN = 10
        else:
            # stock values
            self.STEER_MAX = 270
            self.STEER_DRIVER_ALLOWANCE = 250
            self.STEER_DRIVER_MULTIPLIER = 2
            self.STEER_THRESHOLD = 250
            self.STEER_DELTA_UP = 2
            self.STEER_DELTA_DOWN = 3

