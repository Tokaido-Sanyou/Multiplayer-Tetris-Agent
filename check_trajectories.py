#!/usr/bin/env python3
import pickle

# Check trajectory format
data = pickle.load(open('expert_trajectories_new/trajectory_ep000000.pkl', 'rb'))
print("Trajectory Keys:", list(data.keys()))
print("Steps data type:", type(data['steps']))
print("Number of steps:", len(data['steps']) if hasattr(data['steps'], '__len__') else 'No length')
if data['steps']:
    print("First step keys:", list(data['steps'][0].keys()) if isinstance(data['steps'][0], dict) else 'Not dict') 