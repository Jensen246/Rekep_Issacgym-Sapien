# ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation

A reproduction of the [ReKep](https://rekep-robot.github.io/) framework for robotic manipulation task planning, featuring two independent simulation backends: **Isaac Gym** and **Sapien**.

## Overview

ReKep is a vision-language model (VLM) based task planning framework that enables robots to perform complex manipulation tasks through:

1. **Keypoint Proposal**: Automatically detect task-relevant keypoints from RGB-D observations using DINOv2 visual features
2. **Constraint Generation**: Generate manipulation constraints using GPT-4 based on keypoint positions and task instructions
3. **Subgoal Optimization**: Solve for intermediate end-effector poses that satisfy task constraints
4. **Path Planning**: Generate collision-free paths between subgoals
5. **Execution**: Execute planned trajectories in physics simulation

## Project Structure

```
Rekep_Issacgym/
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── configs/                       # Shared configuration templates
│   └── config.yaml
├── data/                          # Shared robot models and sensor data
│   ├── franka_description/        # Franka Panda URDF for Isaac Gym
│   └── sensor/                    # Sensor calibration data
├── vlm_query/                     # VLM query cache and prompt templates
│   ├── prompt_template.txt        # GPT-4 prompt template
│   ├── cube/                      # Cached results for cube task
│   ├── stick/                     # Cached results for stick task
│   └── bigger_cube/               # Cached results for bigger cube task
│
├── rekep_isaacgym/               # Isaac Gym backend implementation
│   ├── main.py                    # Main entry point
│   ├── environment.py             # Environment abstraction
│   ├── ik_solver.py               # IK solver (PyBullet-based)
│   ├── isaac_gen_data_path.py     # Isaac Gym data generation
│   ├── subgoal_solver.py          # Subgoal optimization
│   ├── path_solver.py             # Path planning
│   ├── keypoint_proposal.py       # Keypoint detection (DINOv2)
│   ├── constraint_generation.py   # VLM constraint generation
│   ├── transform_utils.py         # Coordinate transformations
│   ├── utils.py                   # Utility functions
│   ├── visualizer.py              # 3D visualization
│   ├── configs/                   # Isaac Gym specific config
│   └── test/                      # Test scripts
│
├── rekep_sapien/                  # Sapien backend implementation
│   ├── main.py                    # Main entry point
│   ├── main_with_exec.py          # Main with execution
│   ├── environment.py             # Environment abstraction
│   ├── ik_solver.py               # IK solver (mplib-based)
│   ├── sapien_gen_data_path.py    # Sapien data generation
│   ├── subgoal_solver.py          # Subgoal optimization
│   ├── path_solver.py             # Path planning
│   ├── keypoint_proposal.py       # Keypoint detection (DINOv2)
│   ├── constraint_generation.py   # VLM constraint generation
│   ├── path_visualizer.py         # Path visualization
│   ├── configs/                   # Sapien specific config
│   ├── data/panda/                # Panda URDF for Sapien
│   └── test/                      # Test scripts
│
└── check/                         # Debugging outputs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x (for GPU acceleration)
- Conda (recommended)

### Isaac Gym Version

```bash
# Create conda environment
conda create -n rekep_isaacgym python=3.8
conda activate rekep_isaacgym

# Install PyTorch
pip install torch torchvision

# Install Isaac Gym (follow NVIDIA's official installation guide)
# Download from: https://developer.nvidia.com/isaac-gym
cd isaacgym/python
pip install -e .

# Install other dependencies
pip install pybullet scipy numpy pillow imageio opencv-python
pip install openai  # For VLM queries
pip install transformers  # For DINOv2
```

### Sapien Version

```bash
# Create conda environment
conda create -n rekep_sapien python=3.8
conda activate rekep_sapien

# Install PyTorch
pip install torch torchvision

# Install Sapien
pip install sapien

# Install mplib for motion planning
pip install mplib

# Install other dependencies
pip install scipy numpy pillow imageio opencv-python
pip install openai  # For VLM queries
pip install transformers  # For DINOv2
```

## Usage

### Running Isaac Gym Version

```bash
cd rekep_isaacgym

# Run with cached VLM query (recommended for testing)
python main.py --task stick --use_cached_query --visualize

# Run with live VLM query (requires OpenAI API key)
python main.py --task cube --visualize
```

### Running Sapien Version

```bash
cd rekep_sapien

# Run with cached VLM query
python main.py --task cube --use_cached_query --visualize
```

### Available Tasks

| Task | Description |
|------|-------------|
| `cube` | Pick up a red cube and place it on a green cube |
| `stick` | Grasp the center of a stick and place it on a cube |

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Task name to perform | `stick` (Isaac) / `cube` (Sapien) |
| `--use_cached_query` | Use cached VLM responses instead of querying | `True` |
| `--visualize` | Enable visualization (press ESC to continue) | `True` |

## Configuration

Configuration files are located in `rekep_*/configs/config.yaml`. Key parameters include:

### Main Settings
```yaml
main:
  seed: 0                           # Random seed
  interpolate_pos_step_size: 0.05   # Position interpolation step (m)
  interpolate_rot_step_size: 0.34   # Rotation interpolation step (rad)
  grasp_depth: 0.20                 # Grasp approach depth (m)
  bounds_min: [-0.85, -0.85, 0.0]   # Workspace bounds minimum
  bounds_max: [0.85, 0.85, 1.2]     # Workspace bounds maximum
```

### Solver Settings
```yaml
path_solver:
  opt_pos_step_size: 0.10           # Control point position step
  opt_rot_step_size: 0.39           # Control point rotation step
  sampling_maxfun: 5000             # Maximum optimization iterations

subgoal_solver:
  sampling_maxiter: 100             # Subgoal optimization iterations
```

### VLM Settings
```yaml
constraint_generator:
  model: gpt-4o                     # OpenAI model to use
  temperature: 0.0                  # Sampling temperature
  max_tokens: 2048                  # Maximum response tokens
```

## Architecture

### Core Components

1. **KeypointProposer** (`keypoint_proposal.py`)
   - Extracts manipulation-relevant keypoints from RGB-D images
   - Uses DINOv2 (vits14) for visual feature extraction
   - Applies K-means and MeanShift clustering for keypoint selection

2. **ConstraintGenerator** (`constraint_generation.py`)
   - Interfaces with GPT-4 to generate Python constraint functions
   - Parses VLM output into executable constraint code
   - Manages multi-stage task decomposition

3. **SubgoalSolver** (`subgoal_solver.py`)
   - Optimizes end-effector subgoal poses
   - Uses dual annealing and local minimization
   - Enforces task constraints and collision avoidance

4. **PathSolver** (`path_solver.py`)
   - Plans paths between current pose and subgoal
   - Uses control points for path parameterization
   - Supports path constraint checking

5. **IKSolver** (`ik_solver.py`)
   - Computes inverse kinematics for Franka Panda
   - Isaac Gym version: PyBullet Jacobian pseudoinverse
   - Sapien version: mplib motion planning library

### Backend Differences

| Component | Isaac Gym | Sapien |
|-----------|-----------|--------|
| Physics Engine | PhysX (NVIDIA) | PhysX |
| IK Solver | PyBullet Jacobian | mplib |
| Renderer | Isaac Gym Viewer | Sapien Viewer |
| URDF Format | Standard | Sapien-compatible |

## VLM Query Cache

The `vlm_query/` directory contains cached VLM responses to avoid repeated API calls:

```
vlm_query/
├── prompt_template.txt           # Template for GPT-4 prompts
├── cube/
│   ├── prompt.txt                # Full prompt sent to VLM
│   ├── output_raw.txt            # Raw VLM response
│   ├── stage1_subgoal_constraints.txt
│   ├── stage2_subgoal_constraints.txt
│   ├── stage2_path_constraints.txt
│   └── metadata.json             # Task metadata
└── ...
```

## Environment Variables

Set these environment variables before running:

```bash
# OpenAI API key (required for live VLM queries)
export OPENAI_API_KEY="your-api-key"

# CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Common Issues

1. **URDF loading error**: Ensure the robot URDF files exist in `data/` directory
2. **VLM query fails**: Check your OpenAI API key and network connection
3. **Visualization not showing**: Press ESC to continue after each visualization step
4. **IK solver fails**: The target pose might be outside the robot's workspace

### Debug Mode

Enable verbose output by modifying the config:
```yaml
env:
  verbose: true
```

## Citation

If you use this code in your research, please cite the original ReKep paper:

```bibtex
@article{huang2024rekep,
  title={ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation},
  author={Huang, Wenlong and Wang, Chen and Li, Yunzhu and Zhang, Ruohan and Fei-Fei, Li},
  journal={arXiv preprint arXiv:2409.01652},
  year={2024}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

- [Original ReKep Paper](https://rekep-robot.github.io/) - Stanford University
- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) - Physics simulation
- [Sapien](https://sapien.ucsd.edu/) - Robotics simulation platform
- [DINOv2](https://github.com/facebookresearch/dinov2) - Visual feature extraction
- [OpenAI GPT-4](https://openai.com/) - Vision-language model
