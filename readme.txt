This repository contains code used to generate examples for the paper "Joint Chance Constrained Optimal Control via Linear Programming" by Niklas Schmid, Marta Fochesato, Tobias Sutter and John Lygeros.

The main.py file features a quadcopter simulation that solves a joint chance constrained optimal control problem, where the joint chance constraint represents an invariance, reachability or reach-avoid specification. 

The setting can be adapted as follows:
- alpha denotes the probability of achieving the joint chance constraint,
- OBJECTIVE_TYPE determines whether the specification is an invariance, reachability or reach-avoid problem,
- N denotes the time-horizon,
- the files "invariance_low.png", "reachability_low.png", "reachavoid_low.png" depict the target, safe and unsafe sets. The green dot represents the initial state of the quadcopter.