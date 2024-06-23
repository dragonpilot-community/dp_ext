from openpilot.common.params import Params
from cereal import custom
from openpilot.common.conversions import Conversions as CV
import time

LaneChangeAssistMode = custom.LaneChangeAssistMode

class LaneChangeAssistController:
    def __init__(self):
        params = Params()
        self._mode = int(params.get("dp_lat_lane_change_assist_mode"))
        self._speed = int(params.get("dp_lat_lane_change_assist_speed")) * CV.KPH_TO_MS
        self._timer = float(params.get("dp_lat_lane_change_assist_auto_timer"))
        self._timer_start = None
        self._auto_done = False
        self._auto_torque_applied = False

    def update_off(self):
        self._timer_start = None
        self._auto_done = False
        self._auto_torque_applied = False
        pass

    def update_pre_change(self, blindspot_detected):
        # reset timer when there are vehicles in blind spots
        if blindspot_detected:
            self._timer_start = None

        if self._timer_start is None:
            self._timer_start = time.monotonic()
        else:
            if not self._auto_done and time.monotonic() - self._timer_start >= self._timer:
                self._auto_torque_applied = True
        pass

    def update_changing_state(self):
        pass

    def update_finishing(self):
        self._timer_start = None
        self._auto_done = True
        self._auto_torque_applied = False
        pass

    def get_below_lane_change_speed(self, v_ego):
        return v_ego < self._speed if self._mode in [LaneChangeAssistMode.stock, LaneChangeAssistMode.auto] else False

    def get_torque_applied(self, torque_applied):
        # others
        if self._mode in [LaneChangeAssistMode.disable, LaneChangeAssistMode.hold]:
            return False

        # auto
        elif self._mode == LaneChangeAssistMode.auto:
            return self._auto_torque_applied

        # stock
        else:
            return torque_applied