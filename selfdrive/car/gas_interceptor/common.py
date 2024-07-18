#!/usr/bin/env python3

from openpilot.common.params import Params

ENABLED = Params().get_bool("dp_long_gas_interceptor")