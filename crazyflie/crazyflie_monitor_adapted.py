# Crazyflie three-obstacle scripted validation with lightweight runtime-monitor logging
#
# Purpose:
#   Real Crazyflie hardware feasibility validation of the PyBullet obstacle-avoidance idea.
#
#   This is not a full closed-loop planner.
#   It is a conservative scripted maneuver with a lightweight monitor:
#       - estimated x/y path
#       - obstacle clearance
#       - risk score
#       - automatic monitor activation
#       - speed reduction during monitor activation
#
# Run:
#   python crazyflie_two_obstacle_monitor.py

import os
import time
import math
import logging
import warnings
from datetime import datetime

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

import pandas as pd
import matplotlib.pyplot as plt


URI = 'radio://0/80/2M'

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Physical-test configuration
# -----------------------------

DEFAULT_HEIGHT = 0.10

NORMAL_SPEED = 0.10
MONITOR_SPEED = 0.05

# Local coordinate convention:
#   x = forward direction from takeoff
#   y = left direction from takeoff
#
# Tune these obstacle locations to match your rug/demo setup.
OBSTACLES = [
    {"name": "O1", "cx": 0.18, "cy": -0.02, "radius": 0.04},
    {"name": "O2", "cx": 0.50, "cy": -0.07, "radius": 0.06},
    {"name": "O3", "cx": 0.59, "cy": 0.26, "radius": 0.04},
]

SAFE_CLEARANCE = 0.08       # 8 cm
WARNING_CLEARANCE = 0.14    # 14 cm


# -----------------------------
# Simple estimated pose tracker
# -----------------------------

x_est = 0.0
y_est = 0.0
heading_deg = 0.0
logs = []
start_time = None


def now_s():
    return time.time() - start_time


def clearance_to_obstacles(x, y):
    best_clearance = 999.0
    best_name = "none"

    for obs in OBSTACLES:
        dx = x - obs["cx"]
        dy = y - obs["cy"]
        dist = math.sqrt(dx * dx + dy * dy)
        clearance = dist - obs["radius"]

        if clearance < best_clearance:
            best_clearance = clearance
            best_name = obs["name"]

    return best_name, best_clearance


def monitor_decision(default_speed):
    """
    Lightweight real-hardware monitor.

    If the estimated drone position is close to any obstacle,
    activate the monitor and reduce speed from 0.10 m/s to 0.05 m/s.
    """
    global x_est, y_est

    obs_name, clearance = clearance_to_obstacles(x_est, y_est)

    if clearance < WARNING_CLEARANCE:
        intervention = True
        speed = min(default_speed, MONITOR_SPEED)
        reason = "clearance_warning_slowdown"
    else:
        intervention = False
        speed = default_speed
        reason = "nominal"

    return intervention, speed, reason, obs_name, clearance


def log_state(label, speed_mps, intervention, reason):
    global x_est, y_est, heading_deg

    obs_name, clearance = clearance_to_obstacles(x_est, y_est)

    risk_score = max(0.0, SAFE_CLEARANCE - clearance)
    monitor_zone = 1 if clearance < WARNING_CLEARANCE else 0
    normalized_speed = speed_mps / NORMAL_SPEED if NORMAL_SPEED > 0 else 0.0

    logs.append({
        "time_s": now_s(),
        "label": label,
        "x_m": x_est,
        "y_m": y_est,
        "heading_deg": heading_deg,
        "speed_mps": speed_mps,
        "normalized_speed": normalized_speed,
        "nearest_obstacle": obs_name,
        "current_clearance_m": clearance,
        "risk_score": risk_score,
        "monitor_zone": monitor_zone,
        "intervention": int(intervention),
        "reason": reason,
    })


def update_forward(distance_m):
    global x_est, y_est, heading_deg

    h = math.radians(heading_deg)
    x_est += distance_m * math.cos(h)
    y_est += distance_m * math.sin(h)


def update_circle_left(radius_m, angle_degrees):
    """
    Approximate pose update for MotionCommander circle_left.
    Positive heading change.
    """
    global x_est, y_est, heading_deg

    theta0 = math.radians(heading_deg)
    dtheta = math.radians(angle_degrees)

    cx = x_est - radius_m * math.sin(theta0)
    cy = y_est + radius_m * math.cos(theta0)

    vx = x_est - cx
    vy = y_est - cy

    vx_new = vx * math.cos(dtheta) - vy * math.sin(dtheta)
    vy_new = vx * math.sin(dtheta) + vy * math.cos(dtheta)

    x_est = cx + vx_new
    y_est = cy + vy_new
    heading_deg += angle_degrees


def update_circle_right(radius_m, angle_degrees):
    """
    Approximate pose update for MotionCommander circle_right.
    Negative heading change.
    """
    global x_est, y_est, heading_deg

    theta0 = math.radians(heading_deg)
    dtheta = -math.radians(angle_degrees)

    cx = x_est + radius_m * math.sin(theta0)
    cy = y_est - radius_m * math.cos(theta0)

    vx = x_est - cx
    vy = y_est - cy

    vx_new = vx * math.cos(dtheta) - vy * math.sin(dtheta)
    vy_new = vx * math.sin(dtheta) + vy * math.cos(dtheta)

    x_est = cx + vx_new
    y_est = cy + vy_new
    heading_deg -= angle_degrees


# -----------------------------
# Wrapped motion commands
# -----------------------------

def monitored_forward(mc, distance_m, default_velocity, label):
    print(label)

    intervention, velocity, reason, _, _ = monitor_decision(default_velocity)
    log_state(label + " - before", velocity, intervention, reason)

    mc.forward(distance_m, velocity=velocity)
    update_forward(distance_m)

    intervention_after, velocity_after, reason_after, _, _ = monitor_decision(default_velocity)
    log_state(label + " - after", velocity_after, intervention_after, reason_after)

    time.sleep(1.0)


def monitored_circle_left(mc, radius_m, default_velocity, angle_degrees, label):
    print(label)

    intervention, velocity, reason, _, _ = monitor_decision(default_velocity)
    log_state(label + " - before", velocity, intervention, reason)

    mc.circle_left(radius_m=radius_m, velocity=velocity, angle_degrees=angle_degrees)
    update_circle_left(radius_m, angle_degrees)

    intervention_after, velocity_after, reason_after, _, _ = monitor_decision(default_velocity)
    log_state(label + " - after", velocity_after, intervention_after, reason_after)

    time.sleep(1.0)


def monitored_circle_right(mc, radius_m, default_velocity, angle_degrees, label):
    print(label)

    intervention, velocity, reason, _, _ = monitor_decision(default_velocity)
    log_state(label + " - before", velocity, intervention, reason)

    mc.circle_right(radius_m=radius_m, velocity=velocity, angle_degrees=angle_degrees)
    update_circle_right(radius_m, angle_degrees)

    intervention_after, velocity_after, reason_after, _, _ = monitor_decision(default_velocity)
    log_state(label + " - after", velocity_after, intervention_after, reason_after)

    time.sleep(1.0)


# -----------------------------
# Plotting
# -----------------------------

def save_results():
    out_dir = os.path.join(
        "results",
        "crazyflie_real_monitor",
        datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(logs)
    csv_path = os.path.join(out_dir, "crazyflie_real_monitor_log.csv")
    df.to_csv(csv_path, index=False)

    print("Saved CSV:", csv_path)

    # -----------------------------
    # Trajectory plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(
        df["x_m"],
        df["y_m"],
        marker="o",
        label="Crazyflie hardware path"
    )

    ax = plt.gca()

    for obs in OBSTACLES:
        circ = plt.Circle(
            (obs["cx"], obs["cy"]),
            obs["radius"],
            fill=False,
            linestyle="--",
            linewidth=2
        )
        ax.add_patch(circ)
        ax.text(obs["cx"], obs["cy"], obs["name"], ha="center", va="center")

    avoid_df = df[df["intervention"] == 1]
    if len(avoid_df) > 0:
        plt.scatter(
            avoid_df["x_m"],
            avoid_df["y_m"],
            marker="x",
            s=60,
            label="Monitor active"
        )

    plt.xlabel("estimated x [m]")
    plt.ylabel("estimated y [m]")
    plt.title("Real Crazyflie Hardware Monitor Demonstration")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_crazyflie_trajectory.png"), dpi=300)
    plt.show()

    # -----------------------------
    # Clearance plot
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(
        df["time_s"],
        df["current_clearance_m"],
        marker="o",
        label="Estimated clearance"
    )
    plt.axhline(SAFE_CLEARANCE, linestyle="--", label="safe clearance")
    plt.axhline(WARNING_CLEARANCE, linestyle="--", label="warning clearance")
    plt.axhline(0.0, linestyle="--", label="collision boundary")
    plt.xlabel("time [s]")
    plt.ylabel("clearance [m]")
    plt.title("Real Crazyflie Estimated Obstacle Clearance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_crazyflie_clearance.png"), dpi=300)
    plt.show()

    # -----------------------------
    # Normalized speed and monitor activity
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(
        df["time_s"],
        df["normalized_speed"],
        marker="o",
        label="Normalized commanded speed"
    )
    plt.step(
        df["time_s"],
        df["intervention"],
        where="post",
        label="Monitor active"
    )
    plt.xlabel("time [s]")
    plt.ylabel("normalized value")
    plt.title("Speed Reduction Under Real Crazyflie Monitor")
    plt.ylim(-0.05, 1.10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_crazyflie_normalized_speed_activity.png"), dpi=300)
    plt.show()

    # -----------------------------
    # Risk score
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(
        df["time_s"],
        df["risk_score"],
        marker="o",
        label="Risk score"
    )
    plt.xlabel("time [s]")
    plt.ylabel("risk score")
    plt.title("Real Crazyflie Monitor Risk Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_crazyflie_risk_score.png"), dpi=300)
    plt.show()

    print("Saved plots in:", out_dir)

    print("\nSummary:")
    print("Minimum clearance [m]:", df["current_clearance_m"].min())
    print("Maximum risk score:", df["risk_score"].max())
    print("Monitor active samples:", int(df["intervention"].sum()))
    print("Total samples:", len(df))


# -----------------------------
# Main flight
# -----------------------------

if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    print("Connecting...")
    with SyncCrazyflie(URI) as scf:
        print("Connected.")

        try:
            print("Arming...")
            scf.cf.platform.send_arming_request(True)
            time.sleep(1.0)

            print(f"Takeoff: {DEFAULT_HEIGHT * 100:.0f} cm")

            start_time = time.time()

            with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
                print("Hover 2 sec")
                log_state("takeoff_hover", 0.0, False, "takeoff_hover")
                time.sleep(2.0)
                # --------- 01 --------------------
                monitored_forward(
                    mc,
                    distance_m=0.10,
                    default_velocity=NORMAL_SPEED,
                    label="Move FORWARD 10 cm"
                )
                # --------- 02 --------------------
                monitored_circle_left(
                    mc,
                    radius_m=0.03,
                    default_velocity=NORMAL_SPEED,
                    angle_degrees=90,
                    label="Circle LEFT 90 degrees near O1"
                )
                # --------- 03 --------------------
                monitored_forward(
                    mc,
                    distance_m=0.10,
                    default_velocity=NORMAL_SPEED,
                    label="Move FORWARD 10 cm after O1"
                )
                # --------- 04 --------------------
                monitored_circle_right(
                    mc,
                    radius_m=0.03,
                    default_velocity=NORMAL_SPEED,
                    angle_degrees=90,
                    label="Circle RIGHT 90 degrees to align"
                )
                # --------- 05 --------------------
                monitored_forward(
                    mc,
                    distance_m=0.2,
                    default_velocity=NORMAL_SPEED,
                    label="Move FORWARD 10 cm between obstacles"
                )
                # --------- 06 --------------------
                monitored_circle_right(
                    mc,
                    radius_m=0.03,
                    default_velocity=NORMAL_SPEED,
                    angle_degrees=90,
                    label="Second Circle RIGHT 90 degrees near O2/O3"
                )
                # --------- 07 --------------------
                monitored_forward(
                    mc,
                    distance_m=0.10,
                    default_velocity=NORMAL_SPEED,
                    label="Move FORWARD 20 cm during return corridor"
                )
                # --------- 08 --------------------
                monitored_circle_left(
                    mc,
                    radius_m=0.03,
                    default_velocity=NORMAL_SPEED,
                    angle_degrees=90,
                    label="Circle LEFT 90 degrees to return direction"
                )
                # --------- 09 --------------------
                monitored_forward(
                    mc,
                    distance_m=0.12,
                    default_velocity=NORMAL_SPEED,
                    label="Move FORWARD 20 cm after obstacles"
                )

                print("Landing...")
                log_state("landing", 0.0, False, "landing")
                mc.land()
                time.sleep(2.0)

        finally:
            print("Stopping commander and disarming...")
            try:
                scf.cf.commander.send_stop_setpoint()
                time.sleep(0.2)
            except Exception:
                pass

            try:
                scf.cf.platform.send_arming_request(False)
                time.sleep(1.0)
            except Exception:
                pass

            print("Finished safely.")

    save_results()