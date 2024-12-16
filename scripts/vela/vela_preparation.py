"""Vela preparation script."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import sys
from pathlib import Path

import yaml

configuration_file_path = Path(sys.argv[1])
vela_template_configuration_file_path = Path(sys.argv[2])
helm_template_config_file_path = Path(sys.argv[3])

with configuration_file_path.open("r") as f:
    configuration = yaml.safe_load(f)

with vela_template_configuration_file_path.open("r") as f:
    vela_configuration = yaml.safe_load(f)

for element in configuration["vela"]:
    if element in {"environmentVariables", "setupCommands"}:
        if f"reset{element}" in configuration["vela"] and configuration["vela"][f"reset{element}"]:
            vela_configuration[element] = configuration["vela"][element]
        else:
            vela_configuration[element] += configuration["vela"][element]
    else:
        vela_configuration[element] = configuration["vela"][element]

vela_configuration["trainingConfigFileName"] = f"{vela_configuration['jobName']}.yaml"

helm_template_config_file_path.mkdir(exist_ok=True)
job_configuration_file_path = (
    helm_template_config_file_path / f"{vela_configuration['jobName']}.yaml"
)

with Path(vela_configuration["trainingConfigFileName"]).open("w") as file:
    yaml.dump(
        vela_configuration,
        file,
    )

del configuration["vela"]
with job_configuration_file_path.open("w") as file:
    yaml.dump(configuration, file, default_flow_style=True, width=float("inf"))

print(f"{vela_configuration['jobName']}.yaml")
