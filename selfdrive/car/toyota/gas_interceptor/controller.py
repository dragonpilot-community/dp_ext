#!/usr/bin/env python3
from openpilot.common.numpy_fast import interp, clip
from openpilot.common.conversions import Conversions as CV

from openpilot.selfdrive.car.toyota.values import CAR
from openpilot.dp_ext.selfdrive.car.gas_interceptor.controller import GasInterceptorController

# directly from values.py, may need to check once in a while
MIN_ACC_SPEED = 19. * CV.MPH_TO_MS
PEDAL_TRANSITION = 10. * CV.MPH_TO_MS


class ToyotaGasInterceptorController(GasInterceptorController):
    def __init__(self, CP):
        super().__init__(CP.enableGasInterceptorDEPRECATED)
        self.interceptor_cmd = 0.
        self.car_fingerprint = CP.carFingerprint

    def update_interceptor_cmd(self, long_active, v_ego, actuators_accel):
        if self._enabled and long_active:
            MAX_INTERCEPTOR_GAS = 0.5
            # RAV4 has very sensitive gas pedal
            if self.car_fingerprint in (CAR.RAV4, CAR.RAV4H, CAR.HIGHLANDER):
                PEDAL_SCALE = interp(v_ego, [0.0, MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_TRANSITION], [0.15, 0.3, 0.0])
            elif self.car_fingerprint in (CAR.COROLLA,):
                PEDAL_SCALE = interp(v_ego, [0.0, MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_TRANSITION], [0.3, 0.4, 0.0])
            else:
                PEDAL_SCALE = interp(v_ego, [0.0, MIN_ACC_SPEED, MIN_ACC_SPEED + PEDAL_TRANSITION], [0.4, 0.5, 0.0])
            # offset for creep and windbrake
            pedal_offset = interp(v_ego, [0.0, 2.3, MIN_ACC_SPEED + PEDAL_TRANSITION], [-.4, 0.0, 0.2])
            pedal_command = PEDAL_SCALE * (actuators_accel + pedal_offset)
            self.interceptor_cmd = clip(pedal_command, 0., MAX_INTERCEPTOR_GAS)
        else:
            self.interceptor_cmd = 0.

