#### Improved learning of robot manipulation tasks via tactile intrinsic motivation

##### Project page: 

https://ait.ethz.ch/projects/2021/tactile_rl/

##### Citation

Please cite following publication when using this repository:

```
 @article{vulin2021improved,
  title={Improved learning of robot manipulation tasks via tactile intrinsic motivation},
  author={Vulin, Nikola and Christen, Sammy and Stev{\v{s}}i{\'c}, Stefan and Hilliges, Otmar},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={2194--2201},
  year={2021},
  publisher={IEEE}
}
```

##### Code

Our code extends the method HER and the environment Robotics of Open AI.

Please find our method CPER in the [HER](Algorithm/baselines/her) directory and
the environment extended with the intrinsic reward in the
[Robotics](Gym/gym/envs/robotics) directory. For a successful implementation,
please change one source file in the [MuJoCo](MuJoCo) source code.
