"""Obstacle-aware runtime safety supervisor in PyBullet.

Version 1:
    Direct baseline path
        -> early risk prediction
        -> guided smooth avoidance corridor
        -> soft repulsion only as correction
        -> no freezing near obstacles

Run:
    python runtime_three_obstacle.py --gui False --plot True --duration_sec 35
"""

import os
import time
import argparse
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.utils import sync, str2bool


DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 18
DEFAULT_OUTPUT_FOLDER = "results/obstacle_avoidance_supervisor"


@dataclass
class Obstacle:
    cx: float
    cy: float
    radius: float
    name: str


@dataclass
class SupervisorConfig:
    xmin: float = -1.4
    xmax: float = 1.7
    ymin: float = -1.4
    ymax: float = 1.4
    zmin: float = 0.15
    zmax: float = 1.0

    nominal_speed: float = 0.34
    slow_speed: float = 0.26
    min_speed: float = 0.20

    horizon_steps: int = 80
    dt: float = 1.0 / 48.0

    safe_clearance: float = 0.07
    warning_clearance: float = 0.20
    stop_clearance: float = 0.005
    boundary_margin: float = 0.05


def obstacle_map():
    return [
        Obstacle(cx=-0.05, cy=0.12, radius=0.16, name="O1"),
        Obstacle(cx=0.62, cy=-0.20, radius=0.16, name="O2"),
        Obstacle(cx=1.28, cy=0.10, radius=0.16, name="O3"),
    ]


class DirectWaypointController:
    def __init__(self):
        self.waypoints = [
            np.array([-1.10, 0.12]),
            np.array([1.55, -0.08]),
        ]
        self.index = 0
        self.speed = 0.34
        self.switch_dist = 0.18

    def current_target(self):
        return self.waypoints[self.index]

    def update_and_get_target(self, pos_xy):
        target = self.current_target()

        if np.linalg.norm(target - pos_xy) < self.switch_dist and self.index < len(self.waypoints) - 1:
            self.index += 1
            target = self.current_target()

        return target

    def command_to_target(self, pos_xy, target_xy, speed=None):
        if speed is None:
            speed = self.speed

        diff = target_xy - pos_xy
        dist = np.linalg.norm(diff)

        if dist < 1e-9:
            return np.array([0.0, 0.0, 0.0, 0.0])

        direction = diff / dist
        return np.array([direction[0], direction[1], 0.0, speed])

    def command(self, pos):
        pos_xy = np.array(pos[:2], dtype=float)
        target = self.update_and_get_target(pos_xy)
        return self.command_to_target(pos_xy, target, self.speed), target


class ObstacleAvoidanceSupervisor:
    def __init__(self, cfg: SupervisorConfig, obstacles):
        self.cfg = cfg
        self.obstacles = obstacles
        self.prev_dir = None
        self.avoidance_mode = False
        self.guide_index = 0

        self.guide_points = [
            np.array([-0.35, -0.06]),
            np.array([0.00, -0.12]),
            np.array([0.35, -0.08]),
            np.array([0.65, -0.02]),
            np.array([0.95, -0.10]),
            np.array([1.22, -0.20]),
            np.array([1.42, -0.16]),
            np.array([1.55, -0.08]),
        ]

    def command_to_velocity(self, cmd):
        direction = np.array(cmd[:3], dtype=float)
        speed = float(cmd[3])
        n = np.linalg.norm(direction)

        if n < 1e-9:
            return np.zeros(3)

        return direction / n * speed

    def velocity_to_command(self, vel):
        speed = float(np.linalg.norm(vel))

        if speed < 1e-9:
            return np.array([0.0, 0.0, 0.0, 0.0])

        direction = vel / speed
        return np.array([direction[0], direction[1], direction[2], speed])

    def predict(self, pos, vel):
        p = np.array(pos, dtype=float).copy()
        traj = []

        for _ in range(self.cfg.horizon_steps):
            p = p + vel * self.cfg.dt
            traj.append(p.copy())

        return np.array(traj)

    def clearance_to_obstacle(self, xy, obs):
        return float(np.sqrt((xy[0] - obs.cx) ** 2 + (xy[1] - obs.cy) ** 2) - obs.radius)

    def nearest_obstacle(self, xy):
        best_obs = None
        best_clearance = np.inf

        for obs in self.obstacles:
            c = self.clearance_to_obstacle(xy, obs)

            if c < best_clearance:
                best_clearance = c
                best_obs = obs

        return best_obs, float(best_clearance)

    def min_predicted_clearance(self, traj):
        min_c = np.inf
        min_obs = None

        for p in traj:
            obs, c = self.nearest_obstacle(p[:2])

            if c < min_c:
                min_c = c
                min_obs = obs

        return min_obs, float(min_c)

    def boundary_safe(self, traj):
        m = self.cfg.boundary_margin

        for p in traj:
            x, y, z = p

            if not (self.cfg.xmin + m <= x <= self.cfg.xmax - m):
                return False
            if not (self.cfg.ymin + m <= y <= self.cfg.ymax - m):
                return False
            if not (self.cfg.zmin <= z <= self.cfg.zmax):
                return False

        return True

    def supervise(self, pos, original_target):
        pos_xy = np.array(pos[:2], dtype=float)

        direct_cmd = DirectWaypointController().command_to_target(
            pos_xy,
            original_target,
            self.cfg.nominal_speed,
        )

        direct_vel = self.command_to_velocity(direct_cmd)
        direct_traj = self.predict(pos, direct_vel)
        risky_obs, direct_predicted_clearance = self.min_predicted_clearance(direct_traj)

        nearest_obs, current_clearance = self.nearest_obstacle(pos_xy)
        # Disable avoidance once obstacles are behind the drone
        if pos_xy[0] > 1.45:
            self.avoidance_mode = False

        if (
            direct_predicted_clearance < self.cfg.warning_clearance
            and pos_xy[0] <  1.45
        ):
            self.avoidance_mode = True

        if self.avoidance_mode:
            target = self.guide_points[self.guide_index]

            if np.linalg.norm(target - pos_xy) < 0.12 and self.guide_index < len(self.guide_points) - 1:
                self.guide_index += 1
                target = self.guide_points[self.guide_index]

            if self.guide_index >= 6 and pos_xy[0] > 1.42:
                self.avoidance_mode = False
                target = original_target
        else:
            target = original_target

        to_target = target - pos_xy
        n = np.linalg.norm(to_target)

        if n < 1e-9:
            desired_dir = np.array([0.0, 0.0])
        else:
            desired_dir = to_target / n

        repulse = np.zeros(2)

        for obs in self.obstacles:
            obs_xy = np.array([obs.cx, obs.cy])
            away = pos_xy - obs_xy
            d_center = np.linalg.norm(away)

            if d_center < 1e-9:
                continue

            clearance = d_center - obs.radius

            if clearance < self.cfg.warning_clearance:
                away_dir = away / d_center
                strength = (self.cfg.warning_clearance - clearance) / self.cfg.warning_clearance
                strength = float(np.clip(strength, 0.0, 1.0))
                repulse += 0.35 * strength**1.5 * away_dir

        mixed_dir = desired_dir + repulse

        alpha = 0.88

        if self.prev_dir is None:
            self.prev_dir = mixed_dir.copy()

        mixed_dir = alpha * self.prev_dir + (1.0 - alpha) * mixed_dir

        if np.linalg.norm(mixed_dir) < 1e-9:
            mixed_dir = desired_dir
        else:
            mixed_dir = mixed_dir / np.linalg.norm(mixed_dir)

        self.prev_dir = mixed_dir.copy()

        speed = self.cfg.nominal_speed

        if current_clearance < self.cfg.warning_clearance:
            ratio = (current_clearance - self.cfg.stop_clearance) / (
                self.cfg.warning_clearance - self.cfg.stop_clearance
            )
            ratio = float(np.clip(ratio, 0.0, 1.0))
            ratio = ratio**0.7
            speed = self.cfg.min_speed + ratio * (self.cfg.nominal_speed - self.cfg.min_speed)

        speed = max(speed, 0.20)

        safe_vel = np.array([mixed_dir[0], mixed_dir[1], 0.0]) * speed
        safe_traj = self.predict(pos, safe_vel)

        reason = "guided_smooth_avoidance"

        _, safe_predicted_clearance = self.min_predicted_clearance(self.predict(pos, safe_vel))

        #intervention = self.avoidance_mode or direct_predicted_clearance < self.cfg.warning_clearance
        intervention = self.avoidance_mode
        risk_score = max(0.0, self.cfg.safe_clearance - safe_predicted_clearance)

        return {
            "cmd": self.velocity_to_command(safe_vel),
            "selected_target": target,
            "target_type": "avoidance" if intervention else "original",
            "intervention": intervention,
            "reason": reason,
            "current_clearance": current_clearance,
            "direct_predicted_clearance": direct_predicted_clearance,
            "safe_predicted_clearance": safe_predicted_clearance,
            "adaptive_speed": speed,
            "risk_score": risk_score,
            "avoid_active": int(intervention),
            "avoid_obstacle": nearest_obs.name if nearest_obs is not None else "none",
        }


def extract_position(obs_one_drone):
    return np.array(obs_one_drone[0:3], dtype=float)


def command_speed(cmd):
    direction = np.array(cmd[:3], dtype=float)

    if np.linalg.norm(direction) < 1e-9:
        return 0.0

    return float(cmd[3])


def make_output_dir(base_folder):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_folder, f"run_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_case(use_supervisor=True, **kwargs):
    drone = kwargs.get("drone", DEFAULT_DRONE)
    gui = kwargs.get("gui", DEFAULT_GUI)
    record_video = kwargs.get("record_video", DEFAULT_RECORD_VIDEO)
    user_debug_gui = kwargs.get("user_debug_gui", DEFAULT_USER_DEBUG_GUI)
    obstacles_flag = kwargs.get("obstacles", DEFAULT_OBSTACLES)
    simulation_freq_hz = kwargs.get("simulation_freq_hz", DEFAULT_SIMULATION_FREQ_HZ)
    control_freq_hz = kwargs.get("control_freq_hz", DEFAULT_CONTROL_FREQ_HZ)
    duration_sec = kwargs.get("duration_sec", DEFAULT_DURATION_SEC)

    init_xyzs = np.array([[-1.10, 0.12, 0.30]])
    init_rpys = np.array([[0.0, 0.0, 0.0]])

    env = VelocityAviary(
        drone_model=drone,
        num_drones=1,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles_flag,
        user_debug_gui=user_debug_gui,
    )

    cfg = SupervisorConfig(dt=1.0 / control_freq_hz)
    obstacles = obstacle_map()
    controller = DirectWaypointController()
    supervisor = ObstacleAvoidanceSupervisor(cfg, obstacles)

    action = np.zeros((1, 4))
    steps = int(duration_sec * env.CTRL_FREQ)
    logs = []
    start = time.time()

    for i in range(steps):
        t = i / env.CTRL_FREQ

        obs, reward, terminated, truncated, info = env.step(action)

        pos = extract_position(obs[0])
        pos_xy = np.array(pos[:2], dtype=float)

        direct_cmd, original_target = controller.command(pos)

        if use_supervisor:
            decision = supervisor.supervise(pos, original_target)
            executed_cmd = decision["cmd"]
        else:
            executed_cmd = direct_cmd

            direct_vel = supervisor.command_to_velocity(direct_cmd)
            traj = supervisor.predict(pos, direct_vel)

            _, pred_clearance = supervisor.min_predicted_clearance(traj)
            _, current_clearance = supervisor.nearest_obstacle(pos_xy)

            decision = {
                "selected_target": original_target,
                "target_type": "original",
                "intervention": False,
                "reason": "no_supervisor",
                "current_clearance": current_clearance,
                "direct_predicted_clearance": pred_clearance,
                "safe_predicted_clearance": pred_clearance,
                "adaptive_speed": command_speed(direct_cmd),
                "risk_score": max(0.0, cfg.safe_clearance - pred_clearance),
                "avoid_active": 0,
                "avoid_obstacle": "none",
            }

        action[0, :] = executed_cmd

        logs.append({
            "time_s": t,
            "x_m": pos[0],
            "y_m": pos[1],
            "z_m": pos[2],
            "original_target_x_m": original_target[0],
            "original_target_y_m": original_target[1],
            "selected_target_x_m": decision["selected_target"][0],
            "selected_target_y_m": decision["selected_target"][1],
            "target_type": decision["target_type"],
            "proposed_speed_mps": command_speed(direct_cmd),
            "executed_speed_mps": command_speed(executed_cmd),
            "intervention": int(decision["intervention"]),
            "reason": decision["reason"],
            "current_clearance_m": decision["current_clearance"],
            "direct_predicted_clearance_m": decision["direct_predicted_clearance"],
            "safe_predicted_clearance_m": decision["safe_predicted_clearance"],
            "adaptive_speed_mps": decision["adaptive_speed"],
            "risk_score": decision["risk_score"],
            "avoid_active": decision["avoid_active"],
            "avoid_obstacle": decision["avoid_obstacle"],
        })
        # after obstacles
        final_goal = controller.waypoints[-1]
        if np.linalg.norm(pos_xy - final_goal) < 0.12:
            print(f"Reached final goal at t={t:.2f} s")
            break

        env.render()

        if gui:
            sync(i, start, env.CTRL_TIMESTEP)

    env.close()

    return pd.DataFrame(logs), cfg, obstacles


def summarize(df, case):
    return {
        "case": case,
        "steps": len(df),
        "interventions": int(df["intervention"].sum()),
        "avoidance_steps": int((df["target_type"] == "avoidance").sum()),
        "collision_steps_virtual": int((df["current_clearance_m"] < 0).sum()),
        "close_steps_less_8cm": int((df["current_clearance_m"] < 0.08).sum()),
        "min_clearance_m": float(df["current_clearance_m"].min()),
        "min_predicted_clearance_m": float(df["direct_predicted_clearance_m"].min()),
        "max_risk_score": float(df["risk_score"].max()),
        "mean_executed_speed_mps": float(df["executed_speed_mps"].mean()),
    }


def plot_results(df_base, df_sup, cfg, obstacles, out_dir, show=True):
    plt.figure(figsize=(7, 7))

    plt.plot(df_base["x_m"], df_base["y_m"], label="Baseline direct path", linewidth=2)
    plt.plot(df_sup["x_m"], df_sup["y_m"], label="Runtime supervisor", linewidth=2)

    plt.scatter(
        df_sup["original_target_x_m"],
        df_sup["original_target_y_m"],
        s=35,
        marker="x",
        label="Original waypoints",
    )

    avoid_df = df_sup[df_sup["target_type"] == "avoidance"]

    if len(avoid_df) > 0:
        sampled = avoid_df.iloc[::35]
        plt.scatter(
            sampled["selected_target_x_m"],
            sampled["selected_target_y_m"],
            s=20,
            marker="^",
            label="Supervisor guide targets",
        )

    xs = [cfg.xmin, cfg.xmax, cfg.xmax, cfg.xmin, cfg.xmin]
    ys = [cfg.ymin, cfg.ymin, cfg.ymax, cfg.ymax, cfg.ymin]
    plt.plot(xs, ys, linestyle="--", label="Flight boundary")

    ax = plt.gca()

    for obs in obstacles:
        circ = plt.Circle(
            (obs.cx, obs.cy),
            obs.radius,
            fill=False,
            linestyle="--",
            linewidth=2,
        )
        ax.add_patch(circ)
        ax.text(obs.cx, obs.cy, obs.name, ha="center", va="center", fontsize=12)

    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("Obstacle-Aware Runtime Safety Supervisor")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_supervisor_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "figure_supervisor_trajectory.pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(df_base["time_s"], df_base["current_clearance_m"], label="Baseline clearance")
    plt.plot(df_sup["time_s"], df_sup["current_clearance_m"], label="Supervisor clearance")
    plt.axhline(cfg.safe_clearance, linestyle="--", label="safe clearance")
    plt.axhline(cfg.stop_clearance, linestyle="--", label="stop clearance")
    plt.axhline(0.0, linestyle="--", label="collision boundary")
    plt.xlabel("Time [s]")
    plt.ylabel("Clearance [m]")
    plt.title("Obstacle Clearance Improvement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_supervisor_clearance.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "figure_supervisor_clearance.pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(df_sup["time_s"], df_sup["proposed_speed_mps"], label="Proposed speed")
    plt.plot(df_sup["time_s"], df_sup["executed_speed_mps"], label="Executed speed")
    plt.plot(df_sup["time_s"], df_sup["adaptive_speed_mps"], linestyle="--", label="Adaptive speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed Regulation Under Obstacle-Aware Supervision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_supervisor_speed.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "figure_supervisor_speed.pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(df_base["time_s"], df_base["risk_score"], label="Baseline risk")
    plt.plot(df_sup["time_s"], df_sup["risk_score"], label="Supervisor risk")
    plt.xlabel("Time [s]")
    plt.ylabel("Risk score")
    plt.title("Predicted Risk Reduction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_supervisor_risk.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "figure_supervisor_risk.pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9, 3))
    plt.step(df_sup["time_s"], df_sup["intervention"], where="post", label="Intervention")
    plt.step(df_sup["time_s"], df_sup["avoid_active"], where="post", label="Avoidance active")
    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title("Runtime Supervisor Activity")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_supervisor_activity.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "figure_supervisor_activity.pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def run(**kwargs):
    out_dir = make_output_dir(kwargs.get("output_folder", DEFAULT_OUTPUT_FOLDER))
    print(f"Saving results to: {out_dir}")

    print("Running baseline direct path...")
    df_base, cfg, obstacles = run_case(use_supervisor=False, **kwargs)

    print("Running obstacle-aware runtime supervisor...")
    df_sup, cfg, obstacles = run_case(use_supervisor=True, **kwargs)

    df_base.to_csv(os.path.join(out_dir, "baseline_log.csv"), index=False)
    df_sup.to_csv(os.path.join(out_dir, "runtime_supervisor_log.csv"), index=False)

    summary = pd.DataFrame([
        summarize(df_base, "baseline_direct"),
        summarize(df_sup, "runtime_supervisor"),
    ])

    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_results(df_base, df_sup, cfg, obstacles, out_dir, show=kwargs.get("plot", True))

    print("\nSaved outputs in:")
    print(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obstacle-aware runtime safety supervisor")

    parser.add_argument("--drone", default=DEFAULT_DRONE, type=DroneModel, choices=DroneModel)
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool)
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VIDEO, type=str2bool)
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool)
    parser.add_argument("--user_debug_gui", default=DEFAULT_USER_DEBUG_GUI, type=str2bool)
    parser.add_argument("--obstacles", default=DEFAULT_OBSTACLES, type=str2bool)
    parser.add_argument("--simulation_freq_hz", default=DEFAULT_SIMULATION_FREQ_HZ, type=int)
    parser.add_argument("--control_freq_hz", default=DEFAULT_CONTROL_FREQ_HZ, type=int)
    parser.add_argument("--duration_sec", default=DEFAULT_DURATION_SEC, type=int)
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str)

    args = parser.parse_args()
    run(**vars(args))