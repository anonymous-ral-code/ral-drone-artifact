# adding also diarm at the end to avoid strange behaviour
import time
import logging
import warnings

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M'

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    print("Connecting...")
    with SyncCrazyflie(URI) as scf:
        print("Connected.")

        try:
            print("Arming...")
            scf.cf.platform.send_arming_request(True)
            time.sleep(1)

            print("Takeoff: 25 cm")
            with MotionCommander(scf, default_height=0.25) as mc:
                print("Hovering for 3 seconds...")
                time.sleep(2)

                print("Landing...")
                mc.land()
                time.sleep(2)

        finally:
            print("Stopping commander and disarming...")
            try:
                scf.cf.commander.send_stop_setpoint()
                time.sleep(0.2)
            except Exception:
                pass

            try:
                scf.cf.platform.send_arming_request(False)
                time.sleep(1)
            except Exception:
                pass

            print("Finished safely.")