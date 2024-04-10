import numpy as np




def load_simulation_trajectory(n_traj: int):
    data_dir = '../../dataset/sim_data/'
    traj_name = f'state_{n_traj}.npy'
    filepath = data_dir + traj_name
    
    data = np.load(filepath)
    return data