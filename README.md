# Anonymous Drone Experiment Repository

This repository contains the source code and experimental scripts used to reproduce the drone simulation and Crazyflie monitoring experiments.

The repository is organized into two main parts:

- `gym_pybullet_drones/`: PyBullet-based drone simulation experiments.
- `crazyflie/`: Crazyflie monitoring and real-drone execution scripts.

The code is released anonymously for review purposes.

---

## Repository Structure

```text
repository-root/
│
├── gym_pybullet_drones/
│   ├── runtime_three_obstacle.py
│   ├── runtime_four_obstacle.py
│   ├── Plot/
│   │   └── plot_obstacle_diagnostics_v3.py
│   └── results/
│
└── crazyflie/
    ├── crazyflie_monitor_adapted.py
    ├── crazyflie_monitor.py
    ├── test_hover_25cm.py
    ├── test_hover_100cm.py
    ├── test_safe_takeoff.py
    ├── test_square_small.py
    └── results/
```    

## 1. PyBullet Drone Simulation Experiments
The PyBullet simulation experiments are located in the following folder:
```text
gym_pybullet_drones/
```
This folder contains the scripts for running the obstacle-avoidance experiments.

Three-Obstacle Experiment

To run the three-obstacle simulation:

```bash
python runtime_three_obstacle.py --gui False --plot True --duration_sec 45
Four-Obstacle Experiment
```

To run the four-obstacle simulation:
```bash
python runtime_four_obstacle.py --gui False --plot True --duration_sec 60
Command Arguments
```

---
## 2. Generating Diagnostic Plots

After running the PyBullet experiments, log/result files are generated in the results folder.

To generate the final diagnostic plots:

Copy the generated log/result file into the appropriate plotting/results directory.
Go to the plot folder:
```text
cd gym_pybullet_drones/Plot
```
Run the diagnostic plotting script:
```bash
python plot_obstacle_diagnostics_v3.py
```

The script processes the saved log file and generates obstacle-diagnostic plots for analyzing the drone trajectory, obstacle distance, control behavior, and experiment performance.

---
## 3. Crazyflie Real-Drone Monitoring

The Crazyflie scripts are located in:
```text
crazyflie/
```

To run the adapted Crazyflie monitoring script, go to the folder:
```bash
cd crazyflie


python crazyflie_monitor_adapted.py
```

This script is used for monitoring and running the Crazyflie-based experiment.

Other available Crazyflie test scripts include:
```text
python test_hover_25cm.py
python test_hover_100cm.py
python test_safe_takeoff.py
python test_square_small.py
```
These scripts provide basic flight and monitoring tests.

## 4. Notes
- The PyBullet experiments are intended for simulation-based evaluation.
- The Crazyflie scripts are intended for real-drone monitoring and testing.
- For plotting, make sure the required log/result file is available before running plot_obstacle_diagnostics_v3.py.
- The repository is provided for anonymous review and experiment replication.
