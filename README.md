# Deep Planning Network

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson

![PlaNet policies and predictions](https://imgur.com/UeeQIfo.gif)

This project provides the open source implementation of the PlaNet agent
introduced in [Learning Latent Dynamics for Planning from Pixels][paper].
PlaNet is a purely model-based reinforcement learning algorithm that solves
control tasks from images by efficient planning in a learned latent space.
PlaNet competes with top model-free methods in terms of final performance and
training time while using substantially less interaction with the environment.

If you find this open source release useful, please reference in your paper:

```
@article{hafner2018planet,
  title={Learning Latent Dynamics for Planning from Pixels},
  author={Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  journal={arXiv preprint arXiv:1811.04551},
  year={2018}
}
```

## Method

![PlaNet model diagram](https://i.imgur.com/fpvrAqw.png)

PlaNet models the world as a compact sequence of hidden states. For planning,
we first encode the history of past images into the current state. From there,
we efficiently predict future rewards for multiple action sequences in latent
space. We execute the first action of the best sequence found and replan after
observing the next image.

Find more information:

- [Google AI Blog post][blog]
- [Paper as PDF][paper]
- [Paper as website][website]

[blog]: https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html
[paper]: https://danijar.com/publications/2019-planet.pdf
[website]: https://planetrl.github.io/

## Instructions

To train an agent, install the dependencies and then run:

```sh
python3 -m planet.scripts.train  \
    --logdir /path/to/logdir \
    --config default \
    --params '{tasks: [cheetah_run]}'
```

The available tasks are listed in `scripts/tasks.py`. The default parameters
can be found in `scripts/configs.py`. To replicate the experiments in our
paper, pass the following parameters to `--params {...}` in addition to the
list of tasks:

| Experiment | Parameters |
| :--------- | :--------- |
| PlaNet | No additional parameters. |
| No overshooting | `overshooting: 0` |
| Random dataset | `collect_every: 999999999, num_seed_episodes: 1000` |
| Purely deterministic | `overshooting: 0, mean_only: True, divergence_scale: 0.0, global_divergence_scale: 0.0` |
| Purely stochastic | `model: ssm` |
| One agent all tasks | `collect_every: 30000` |

## Modifications

During development, you can set `--config debug` to reduce the episode length,
batch size, and collect data more freqnently. This helps to quickly reach all
parts of the code. You can use `--num_runs 1000 --resume_runs False` to start a
run in a new sub directory every time to execute the script. These are good
places to start when modifying the code:

| Directory | Description |
| :-------- | :---------- |
| `scripts/configs.py` | Add new parameters or change defaults. |
| `scripts/tasks.py` | Add or modify environments. |
| `models` | Add or modify latent transition models. |
| `networks` | Add or modify encoder and  decoder networks. |

## Dependencies

- dm_control
- gym
- mujoco_py
- ruamel.yaml
- scikit-image
- scipy
- tensorflow-gpu
- tensorflow_probability

Disclaimer: This is not an official Google product.

