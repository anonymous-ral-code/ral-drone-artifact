# test_safe_takeoff.py
import time
import logging

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M'

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    print("Connecting...")
    with SyncCrazyflie(URI) as scf:
        print("Connected.")

        print("Arming...")
        scf.cf.platform.send_arming_request(True)
        time.sleep(1)

        print("Very low takeoff: 15 cm")
        with MotionCommander(scf, default_height=0.15) as mc:
            time.sleep(1.5)

            print("Landing now...")