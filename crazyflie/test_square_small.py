import time
import logging
import warnings

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M'

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    print("Connecting...")
    with SyncCrazyflie(URI) as scf:
        print("Connected.")

        scf.cf.platform.send_arming_request(True)
        time.sleep(1)

        with MotionCommander(scf, default_height=0.25) as mc:
            print("Hover...")
            time.sleep(2)

            print("Square step 1")
            mc.forward(0.1)
            time.sleep(1)

            print("Square step 2")
            mc.right(0.1)
            time.sleep(1)

            print("Square step 3")
            mc.back(0.1)
            time.sleep(1)

            print("Square step 4")
            mc.left(0.1)
            time.sleep(1)

            print("Landing...")