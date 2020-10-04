import argparse
import os
import yaml
import simtk.unit as u
from .MD_utils.openmm_reporter import CopySender
from .MD_utils.openmm_simulation import (
    openmm_simulate_amber_implicit,
    openmm_simulate_amber_explicit,
)


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="YAML config file", required=True)
    path = parser.parse_args().config
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


def build_simulation_params(cfg: dict) -> dict:
    top_file = cfg.get("top_file", None)
    chk_file = cfg.get("checkpoint_file", None)

    senders = []
    if cfg.get("h5_scp_path"):
        # Send HDF5 files to any remote machine (medulla)
        # Requires an IdentityFile entry in ~/.ssh/config
        senders.append(CopySender(cfg["h5_scp_path"], method="scp"))

    if cfg["result_dir"]:
        # Send HDF5 files to any locally-mounted directory (lustre)
        senders.append(CopySender(cfg["result_dir"], method="cp"))

    return dict(
        pdb_file=cfg["pdb_file"],
        reference_pdb_file=cfg["reference_pdb_file"],
        top_file=top_file,
        check_point=chk_file,
        run_base_id=cfg["run_base_id"],
        local_run_dir=cfg["local_run_dir"],
        GPU_index=0,
        output_traj="output.dcd",
        output_log="output.log",
        output_cm="output_cm",
        report_time=float(cfg["report_interval_ps"]) * u.picoseconds,
        sim_time=float(cfg["simulation_length_ns"]) * u.nanoseconds,
        reeval_time=float(cfg["reeval_time_ns"]) * u.nanoseconds,
        dt_ps=float(cfg["dt_ps"]) * u.picoseconds,
        senders=senders,
    )


if __name__ == "__main__":
    cfg = get_config()
    simulation_kwargs = build_simulation_params(cfg)

    if cfg["sim_type"] == "explicit":
        run_simulation = openmm_simulate_amber_explicit
    else:
        assert cfg["sim_type"] == "implicit"
        run_simulation = openmm_simulate_amber_implicit

    run_simulation(**simulation_kwargs)