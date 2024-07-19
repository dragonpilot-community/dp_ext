#!/usr/bin/env python3

def crc8_pedal(data):
    crc = 0xFF    # standard init value
    poly = 0xD5   # standard crc8: x8+x7+x6+x4+x2+1
    size = len(data)
    for i in range(size - 1, -1, -1):
        crc ^= data[i]
        for _ in range(8):
            if ((crc & 0x80) != 0):
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc <<= 1
    return crc

def create_gas_interceptor_command(packer, gas_amount, idx):
    # Common gas pedal msg generator
    enable = gas_amount > 0.001

    values = {
        "ENABLE": enable,
        "COUNTER_PEDAL": idx & 0xF,
    }

    if enable:
        values["GAS_COMMAND"] = gas_amount * 255.
        values["GAS_COMMAND2"] = gas_amount * 255.

    dat = packer.make_can_msg("GAS_COMMAND", 0, values)[2]

    checksum = crc8_pedal(dat[:-1])
    values["CHECKSUM_PEDAL"] = checksum

    return packer.make_can_msg("GAS_COMMAND", 0, values)