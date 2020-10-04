import argparse
import os
import yaml
import simtk.unit as u
from deepdrivemd.util import CopySender
from deepdrivemd.sim.openmm_sim import run_simulation


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="YAML config file", required=True)
    path = parser.parse_args().config
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


def build_simulation_params(cfg: dict) -> dict:
    top_file = cfg.get("top_file", None)
    checkpoint_file = cfg.get("checkpoint_file", None)

    return dict(
        pdb_file=cfg["pdb_file"],
        reference_pdb_file=cfg["reference_pdb_file"],
        top_file=top_file,
        checkpoint_file=checkpoint_file,
        omm_dir_prefix=cfg["omm_dir_prefix"],
        local_run_dir=cfg["local_run_dir"],
        gpu_index=0,
        sim_type=cfg["sim_type"],
        output_traj="output.dcd",
        output_log="output.log",
        report_interval_ps=float(cfg["report_interval_ps"]) * u.picoseconds,
        sim_time=float(cfg["simulation_length_ns"]) * u.nanoseconds,
        reeval_time=float(cfg["reeval_time_ns"]) * u.nanoseconds,
        dt_ps=float(cfg["dt_ps"]) * u.picoseconds,
        h5_scp_path=cfg["h5_scp_path"],
        result_dir=cfg["result_dir"],
        input_dir=cfg["input_dir"],
    )


if __name__ == "__main__":
    cfg = get_config()
    simulation_kwargs = build_simulation_params(cfg)
    run_simulation(**simulation_kwargs)
