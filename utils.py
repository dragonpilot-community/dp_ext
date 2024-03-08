import socket


def get_ip_addr():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    # doesn't even have to be reachable
    s.connect(('10.255.255.255', 1))
    return s.getsockname()[0]
  except:
    return None
  finally:
    s.close()

def is_private_ip():
  ip = get_ip_addr()
  if ip is None:
    return False
  # Check if the IP address is in a private network range
  private_networks = [
    "10.",
    "172.16.",
    "192.168.",
  ]
  return any(ip.startswith(net) for net in private_networks)
