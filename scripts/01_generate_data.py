"""Collect BeamNG telemetry for SINDy training.

This file mirrors the BeamNG connection style used in Hybrid-VLA while keeping
the learning-sensitive pieces as TODOs:

* control excitation math,
* quaternion-to-yaw conversion,
* packing raw samples into final SINDy arrays.
"""

from pathlib import Path

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import State


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "beamng_telemetry.npz"

HOST = "localhost"
PORT = 64256
LEVEL_ID = "gridmap_v2"
SCENARIO_NAME = "sindympc_beamng_telemetry"
VEHICLE_ID = "ego_vehicle"
VEHICLE_MODEL = "etk800"

SIM_HZ = 100
NUM_STEPS = 300
DT = 1.0 / SIM_HZ


def build_scenario():
    """Create the GridMap v2 scenario and spawn an ETK800 vehicle.

    BeamNG uses internal level ids; the id for "GridMap v2" is
    ``gridmap_v2``. The vehicle spawn pose is intentionally simple so you can
    change it later after checking the map.
    """
    scenario = Scenario(
        LEVEL_ID,
        SCENARIO_NAME,
        description="SINDy-MPC BeamNG telemetry collection",
    )
    vehicle = Vehicle(
        VEHICLE_ID,
        model=VEHICLE_MODEL,
        license="SINDY",
        color="Red",
    )
    scenario.add_vehicle(
        vehicle,
        pos=(0.0, 0.0, 0.0),
        rot_quat=(0.0, 0.0, 0.0, 1.0),
    )
    return scenario, vehicle


def attach_sensors(vehicle):
    """Attach the minimal sensor set needed for state telemetry.

    The ``State`` sensor updates ``vehicle.state`` after
    ``vehicle.sensors.poll("state")``. The fields you will use for the 2D
    training state are typically:

    * ``vehicle.state["pos"]``: world position ``(x, y, z)``,
    * ``vehicle.state["vel"]``: world velocity ``(vx, vy, vz)``,
    * ``vehicle.state["rotation"]``: quaternion ``(qx, qy, qz, qw)``.
    """
    if "state" not in vehicle.sensors.data:
        vehicle.sensors.attach("state", State())


def generate_control_command(step_idx, t):
    """Return a BeamNG control command for this timestep.

    The returned dictionary is passed directly into ``vehicle.control(...)``.
    BeamNG expects values such as ``throttle``, ``brake``, ``steering``,
    ``parkingbrake``, and optionally ``gear``.
    """
    # TODO: Write your random or oscillating excitation policy.
    # Suggested output shape:
    # return {
    #     "throttle": ...,
    #     "brake": ...,
    #     "steering": ...,
    #     "parkingbrake": 0.0,
    #     "gear": 2,
    # }
    raise NotImplementedError("Implement BeamNG excitation/control generation.")


def state_to_training_sample(vehicle_state, control_command, t):
    """Convert raw BeamNG state into one SINDy training sample.

    Args:
        vehicle_state: The latest ``vehicle.state`` dictionary.
        control_command: The command just applied with ``vehicle.control``.
        t: Simulation time in seconds.

    Returns:
        Your chosen raw sample structure.

    Notes:
        To form ``[x, y, theta, v]``:

        * take ``x, y`` from BeamNG world position ``pos[:2]``;
        * compute ``theta`` from the quaternion in ``rotation``;
        * compute planar speed ``v`` from the XY velocity components.

        To form ``[throttle/brake, steering]`` or ``[a, delta]``, decide how
        you want to encode throttle and brake for SINDy.
    """
    # TODO: Implement quaternion-to-yaw conversion yourself.
    # TODO: Decide whether the first control channel is throttle-minus-brake,
    #       measured/estimated acceleration, or separate throttle/brake fields.
    raise NotImplementedError("Implement BeamNG state/control sample extraction.")


def pack_samples_for_sindy(samples):
    """Pack raw per-step samples into NumPy arrays for saving.

    Returns:
        ``(states, controls, timestamps)`` where states should have shape
        ``(N, 4)`` and controls should have shape ``(N, 2)``.
    """
    # TODO: Build these arrays from your sample structure.
    raise NotImplementedError("Pack samples into states, controls, timestamps.")


def save_telemetry(states, controls, timestamps, output_path=OUTPUT_PATH):
    """Save SINDy training arrays to ``data/beamng_telemetry.npz``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=np.asarray(states, dtype=np.float32),
        controls=np.asarray(controls, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        dt=np.asarray(DT, dtype=np.float32),
    )


def collect_telemetry(beamng_home=None, headless=False):
    """Connect to BeamNG, load GridMap v2, and collect 300 state samples.

    This follows the same high-level pattern as Hybrid-VLA:

    1. create ``BeamNGpy`` with host/port/home/headless,
    2. call ``bng.open(launch=True)``,
    3. create and load a scenario,
    4. run deterministic physics steps with ``bng.control.step(...)``.
    """
    set_up_simple_logging()

    bng = BeamNGpy(HOST, PORT, home=beamng_home, headless=headless)
    scenario, vehicle = build_scenario()
    attach_sensors(vehicle)

    samples = []

    print(f"[Init] Connecting to BeamNG at {HOST}:{PORT}")
    bng.open(launch=True)
    try:
        scenario.make(bng)
        bng.settings.set_deterministic(SIM_HZ)
        bng.scenario.load(scenario)
        bng.ui.hide_hud()
        bng.scenario.start()
        bng.control.pause()

        vehicle.queue_lua_command('controller.mainController.setGearboxMode("arcade")')

        # Warm up the state sensor before logging.
        for _ in range(10):
            bng.control.step(1)
            vehicle.sensors.poll("state")

        print(f"[Sim] Collecting {NUM_STEPS} samples at {SIM_HZ} Hz")
        for step_idx in range(NUM_STEPS):
            t = step_idx * DT
            control_command = generate_control_command(step_idx, t)

            vehicle.control(**control_command)
            bng.control.step(1)

            # Polling the named State sensor refreshes ``vehicle.state``.
            vehicle.sensors.poll("state")
            sample = state_to_training_sample(vehicle.state, control_command, t)
            samples.append(sample)

    finally:
        try:
            bng.ui.show_hud()
        except Exception:
            pass
        bng.close()

    return samples


def main():
    """Collect, pack, and save BeamNG telemetry for SINDy."""
    samples = collect_telemetry()
    states, controls, timestamps = pack_samples_for_sindy(samples)
    save_telemetry(states, controls, timestamps)
    print(f"Saved BeamNG telemetry to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
