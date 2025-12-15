[project]
name = "betta-pipeline"
version = "0.1.0"
description = "Betta fish pose estimation pipeline"
requires-python = ">=3.9"

dependencies = [
    "torch",
    "omegaconf",
    "pandas",
    "lightning-pose",
]

[project.scripts]
betta-train = "betta_pipeline.cli:main"