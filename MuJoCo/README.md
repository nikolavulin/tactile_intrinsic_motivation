Replace the original `mjsim.pyx` file with the one here in the directory.

Since MuJoCo makes multiple simulation steps for one action of the RL agent,
one can miss a force measurement. Therefore, with this file we collect also the 
sensor readings of all simulation steps and return it to the agent.