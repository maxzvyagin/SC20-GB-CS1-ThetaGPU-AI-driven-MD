# SC20-GB-CS1-ThetaGPU-AI-driven-MD

Stream-AI-MD: Streaming AI-driven Adaptive Molecular Simulations

## How to run

### Setup

Install `deepdrivemd` into a virtualenv with:

```
pip install -e .
```

### Generating a YAML input spec:

First, run this command to get a _sample_ YAML config file:

```
 python -m deepdrivemd.driver.config
```

This will write a file named `deepdrivemd_template.yaml` which should be adapted for the experiment at hand. You should configure the `md_runner` and `outlier_detection` sections to use the appropriate run commands and environment setups.

### Running an experiment

Then, launch an experiment with:

```
python -m deepdrivemd.driver.experiment_main <experiment_config.yaml>
```

This experiment should be launched
