# SC20-GB-CS1-ThetaGPU-AI-driven-MD
Stream-AI-MD: Streaming AI-driven Adaptive Molecular Simulations 

## To generate a template YAML file

bash```
 python -m deepdrivemd.driver.config
 ```

 This will write a file named `deepdrivemd_template.yaml` which should be adapted for the experiment.

 Then, launch an experiment with:

 bash```
 python -m deepdrivemd.driver.experiment_main -c <template_file>
 ```
