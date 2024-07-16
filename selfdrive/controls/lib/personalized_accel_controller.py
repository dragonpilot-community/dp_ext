import numpy as np
from collections import deque, defaultdict
import json
from openpilot.common.params import Params

class PersonalizedAccelController:
    def __init__(self, learning_rate=0.01, memory_size=10000, samples_per_range=100):
        self.enabled = False
        self._params = Params()
        # A high smoothing factor with a low learning rate will result in very stable, slow-changing behavior.
        # A low smoothing factor with a high learning rate will be very responsive to new data, potentially unstable.
        self.learning_rate = learning_rate
        self.smoothing_factor = 0.9  # New parameter for smoothing
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.update_counter = 0
        self.samples_per_range = samples_per_range

        # Fixed BP size from 0 to 39 in steps of 3
        self.profile_size = 14  # (39 - 0) / 3 + 1
        self.A_MAX_BP = list(np.arange(0, 40, 3))

        # Default values (ensure non-negative)
        self.MAX_VALS = np.maximum(0, np.interp(self.A_MAX_BP, [0., 10.0, 25., 40.], [1.6, 1.2, 0.8, 0.6])).tolist()

        # Initialize profiles
        self.A_CRUISE_MAX_VALS = self.MAX_VALS.copy()

        # Initialize sample collectors
        self.cruise_samples = defaultdict(list)

        self.speed_range_samples = {i: [] for i in range(len(self.A_MAX_BP))}

        # Load saved model or use defaults
        self.load_model()

    def update(self, car_state, lead_present):
        blinker_off = not car_state.leftBlinker and not car_state.rightBlinker
        straight_road = abs(car_state.steeringAngleDeg) < 30

        if not lead_present and blinker_off and car_state.gasPressed and straight_road:
            speed = car_state.vEgo
            acceleration = car_state.aEgo

            # Get the current speed range
            speed_index = self._get_speed_index(speed)

            # Update only the relevant speed range
            self.speed_range_samples[speed_index].append(acceleration)

            # Update this speed range if we have enough samples
            if len(self.speed_range_samples[speed_index]) >= self.samples_per_range:
                self._update_cruise_profile_for_index(speed_index)

            # Increment update counter and save model periodically
            self.update_counter += 1
            if self.update_counter % 1000 == 0:
                self.save_model()

    def _update_cruise_profile_for_index(self, index):
        samples = self.speed_range_samples[index]
        if samples:
            new_sample_val = max(0, float(np.percentile(samples, 95)))

            previous_val = self.A_CRUISE_MAX_VALS[index]
            smoothed_val = self.smoothing_factor * previous_val + (1 - self.smoothing_factor) * new_sample_val

            self.A_CRUISE_MAX_VALS[index] = previous_val * (1 - self.learning_rate) + smoothed_val * self.learning_rate

            self.speed_range_samples[index] = []  # Clear the samples after updating

    def _check_and_update_all_ranges(self):
        for i in range(len(self.A_MAX_BP)):
            if len(self.cruise_samples[i]) >= self.samples_per_range:
                self._update_cruise_profile_for_index(i)

    def _update_acceleration_profiles(self):
        for state, accel in self.memory:
            speed, lead_present = state
            speed_index = self._get_speed_index(speed)

            if not lead_present:
                self.cruise_samples[speed_index].append((speed, accel))

        # Update profiles if we have enough samples
        self._update_cruise_profile()

    def _get_speed_index(self, speed):
        return min(int(speed / 3), len(self.A_MAX_BP) - 1)

    def _update_cruise_profile(self):
        for i, bp in enumerate(self.A_MAX_BP):
            samples = self.cruise_samples[i]
            if len(samples) >= self.samples_per_range:
                speeds, accels = zip(*samples)
                new_sample_val = max(0, float(np.percentile(accels, 95)))

                # Consider the previous value
                previous_val = self.A_CRUISE_MAX_VALS[i]
                smoothed_val = self.smoothing_factor * previous_val + (1 - self.smoothing_factor) * new_sample_val

                # Apply learning rate
                self.A_CRUISE_MAX_VALS[i] = previous_val * (1 - self.learning_rate) + smoothed_val * self.learning_rate

                self.cruise_samples[i] = []  # Clear the samples after updating

    def get_max_accel(self, v_ego, STOCK_BP, STOCK_VALS):
        if self.enabled:
            return np.interp(v_ego, self.A_MAX_BP, self.A_CRUISE_MAX_VALS)
        else:
            return np.interp(v_ego, STOCK_BP, STOCK_VALS)

    def set_profile_size(self, new_size):
        self.profile_size = new_size
        # Reinitialize all acceleration profiles
        self.A_CRUISE_MAX_VALS = list(np.linspace(1.6, 0.6, num=new_size))
        self._update_acceleration_profiles()

    def set_enabled(self, enabled):
        self.enabled = enabled

    def save_model(self):
        self._params.put("dp_long_pac_vals", json.dumps(self._numpy_to_list(self.A_CRUISE_MAX_VALS)))

    def load_model(self):
        params_val = self._params.get("dp_long_pac_vals")
        if params_val is not None:
            self.A_CRUISE_MAX_VALS = json.loads(params_val)
            print(self.A_CRUISE_MAX_VALS)

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
                self.update(sm['carState'], sm['radarState'].leadOne.status and sm['carState'].vEgo > 0. and sm['radarState'].leadOne.dRel / sm['carState'].vEgo < 1.)
                print(self.A_CRUISE_MAX_VALS)


def main():
    pac = PersonalizedAccelController(learning_rate=0.01, memory_size=10000)
    pac.pac_thread()

if __name__ == "__main__":
    main()
