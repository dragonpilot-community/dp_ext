#!/usr/bin/env python3

from openpilot.dp_ext.selfdrive.car.gas_interceptor.common import ENABLED
from openpilot.dp_ext.selfdrive.car.gas_interceptor.can import create_gas_interceptor_command
from opendbc.can.packer import CANPacker

class GasInterceptorController:
    def __init__(self):
        self._enabled = ENABLED
        self.packer = CANPacker("gas_interceptor")
        pass

    def is_enabled(self):
        return self._enabled

    def get_can_sends(self, frame, gas):
        if self._enabled:
            return create_gas_interceptor_command(self.packer, gas, frame // 2)
