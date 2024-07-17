import numpy as np
import json
import bisect
from collections import deque
from openpilot.common.params import Params

class PersonalizedAccelController:
    def __init__(self, learning_rate=0.01):
        self.enabled = False
        self._params = Params()
        self.learning_rate = learning_rate
        self.smoothing_factor = 0.9
        self.update_counter = 0

        self.samples_per_range = self._initialize_variable_samples()
        self.max_samples_per_range = 50

        self.A_MAX_BP = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 32.5, 35, 37.5, 40]
        self.profile_size = len(self.A_MAX_BP)

        original_bp = [0., 10.0, 25., 40.]
        original_vals = [1.6, 1.2, 0.8, 0.6]

        self.MAX_VALS = np.maximum(0, np.interp(self.A_MAX_BP, original_bp, original_vals)).tolist()
        self.A_CRUISE_MAX_VALS = self.MAX_VALS.copy()

        self.speed_range_samples = {i: deque(maxlen=self.max_samples_per_range) for i in range(self.profile_size)}

        self.load_model()

    def _initialize_variable_samples(self):
        return [5, 5, 8, 8, 10, 10, 12, 12, 15, 15, 20, 20, 25, 30, 35, 35, 40, 40, 40]

    def update(self, car_state, lead_present):
        if (not lead_present and
                not car_state.leftBlinker and
                not car_state.rightBlinker and
                car_state.gasPressed and
                abs(car_state.steeringAngleDeg) < 30):

            speed_index = self._get_speed_index(car_state.vEgo)
            self.speed_range_samples[speed_index].append(car_state.aEgo)

            if speed_index < len(self.samples_per_range):
                if len(self.speed_range_samples[speed_index]) >= self.samples_per_range[speed_index]:
                    self._update_cruise_profile_for_index(speed_index)

            self.update_counter += 1
            if self.update_counter % 1000 == 0:
                self.save_model()

    def _get_speed_index(self, speed):
        index = bisect.bisect_right(self.A_MAX_BP, speed) - 1
        return max(0, min(index, len(self.A_MAX_BP) - 1))

    def _update_cruise_profile_for_index(self, index):
        samples = list(self.speed_range_samples[index])
        if samples:
            new_sample_val = max(0, float(np.percentile(samples, 95)))

            previous_val = self.A_CRUISE_MAX_VALS[index]
            smoothed_val = self.smoothing_factor * previous_val + (1 - self.smoothing_factor) * new_sample_val

            self.A_CRUISE_MAX_VALS[index] = previous_val * (1 - self.learning_rate) + smoothed_val * self.learning_rate

    def get_max_accel(self, v_ego, STOCK_BP, STOCK_VALS):
        if self.enabled:
            return np.interp(v_ego, self.A_MAX_BP, self.A_CRUISE_MAX_VALS)
        else:
            return np.interp(v_ego, STOCK_BP, STOCK_VALS)

    def set_profile_size(self, new_size):
        self.profile_size = new_size
        self.A_CRUISE_MAX_VALS = list(np.linspace(1.6, 0.6, num=new_size))
        self.speed_range_samples = {i: deque(maxlen=self.max_samples_per_range) for i in range(self.profile_size)}

    def set_enabled(self, enabled):
        self.enabled = enabled

    def save_model(self):
        data = {
            'bp': self.A_MAX_BP,
            'vals': self._numpy_to_list(self.A_CRUISE_MAX_VALS)
        }
        self._params.put("dp_long_pac_vals", json.dumps(data))

    def load_model(self):
        params_val = self._params.get("dp_long_pac_vals")
        if params_val is not None:
            data = json.loads(params_val)
            self.A_CRUISE_MAX_VALS = np.maximum(0, np.interp(self.A_MAX_BP, data.get('bp'), data.get('vals'))).tolist()

    def _numpy_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [self._numpy_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._numpy_to_list(value) for key, value in obj.items()}
        else:
            return obj

    def pac_thread(self):
        from cereal.messaging import SubMaster

        sm = SubMaster(['carState', 'radarState'])

        while True:
            sm.update()
            if sm.updated['carState']:
                self.update(sm['carState'], sm['radarState'].leadOne.status and sm['carState'].vEgo > 0. and sm['radarState'].leadOne.dRel / sm['carState'].vEgo < 2.5)
                # print(self.A_CRUISE_MAX_VALS)

def main():
    pac = PersonalizedAccelController()
    pac.pac_thread()

if __name__ == "__main__":
    main()