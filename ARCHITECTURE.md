# High-level overview

`deepdrivemd` is organized into the following subpackages and modules:

1. `deepdrivemd.agents` contains the Agent main loop and various Agent implementations.
2. `deepdrivemd.models` contains the `Model` base class, from which all model wrappers are subclassed
3. `train.py` contains the training service main loop
4. `deepdrivemd.data` contains the `DataController` API and its implementations

# Agents

`deepdrivemd.agents.agent` is the main agent entrypoint (execute `python -m deepdrivemd.agents.agent`). 
It defines the `Agent` class which all agents must subclass.  It also contains the main loop which loads a 
given configuration file and runs the agent service.

All Agent implementations must be model-agnostic and use only the `Model` API (defined below). This is to ensure
that Agents and Models are completely orthogonal.

## The `Agent` base class

```py3

class Agent:
    def __init__(self, model: Model) -> None:
        """A Model subclass is injected into the Agent"""
        ...
```


# Models

## The `Model` base class

This is a common interface to all ML models used by `train.py` or `deepdrivemd.agents.agent`. The API
must cover the **union** of all features needed to pre-exec, preprocess, forward-pass, and train models
on various systems.  The API must also be platform-agnostic and model framework agnostic.

Concrete `Model` implementations can import other platform or framework-specific code (e.g. Torch models) 

```py3

class Model:

    def __init__(self, model_config: ModelConfig) -> None:
        ...

    def pre_train(self) -> None:
        ...

    def pre_predict(self) -> None:
        ...

    def train(self, num_steps: int = 1000) -> TrainingResult:
        ...

    def predict(self, h5_files: List[Path]) -> List[...]:
        ...
```
