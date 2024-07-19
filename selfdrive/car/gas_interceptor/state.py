#!/usr/bin/env python3
from opendbc.can.parser import CANParser
from cereal import messaging

class GasInterceptorState:
    def __init__(self, enabled):
        self._enabled = enabled

        if self._enabled:
            self.can_sock = messaging.sub_sock('can')
            messages = [("GAS_SENSOR", 50)]
            self.can_parser = CANParser('gas_interceptor', messages, 0)

    def get_gas_pressed(self, threshold, gas_pressed):
        if not self._enabled:
            return gas_pressed

        can_strings = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
        self.can_parser.update_strings(can_strings)

        gas = (self.can_parser.vl["GAS_SENSOR"]["INTERCEPTOR_GAS"] + self.can_parser.vl["GAS_SENSOR"]["INTERCEPTOR_GAS2"]) // 2
        return gas > threshold