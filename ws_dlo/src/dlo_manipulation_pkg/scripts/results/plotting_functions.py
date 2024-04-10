import matplotlib.pyplot as plt
import numpy as np
path = "../../../../../results/sim/control/"
def plot_time_series(experiment_name, idx_range, save_fig_flag=False):
    plt.figure(figsize=(2.5,2.5))
    plt.xlabel('$t$ [s]')
    for i in range(idx_range[0], idx_range[1]):
        state = np.load(path+experiment_name+"/3D/state_"+str(i)+".npy")
        t = [i*0.1 for i in range(state.shape[0])]

        plt.ylabel('$|| \Delta x(t) ||_2$ [m]')
        metric = []
        for k in range(state.shape[0]):
            position_error = state[k,1+3:31-3] - state[k,-31+3:-3-1]
            # if k == 0: err_at_start = position_error
            num_fps = 8
            fps_error_sum = 0
            for j in range(num_fps):
                fps_error = np.linalg.norm(position_error[j*3:j*3+3])
                fps_error_sum += fps_error
            # metric.append(fps_error_sum / num_fps)
            metric.append(fps_error_sum)
                    
        plt.plot(t,metric)
    plt.grid()
        
    if save_fig_flag:
        plt.tight_layout()
        plt.savefig("figs/tracking_error_"+experiment_name+".pdf")


def plot_tracking_error_box_and_whiskers(experiment_names, experiment_names_pretty, idx_range, log_scale=False,save_fig_flag=False):
    dists = []
    for experiment_name in experiment_names:
        dist = []
        for i in range(idx_range[0], idx_range[1]):
            state = np.load(path+experiment_name+"/3D/state_"+str(i)+".npy")
            N = state.shape[0]
            num_fps = 8
            output = [np.linalg.norm(x) for x in state[:,1+3:31-3] - state[:,-31+3:-3-1]]
            sum = np.array(output).sum()/N
            dist += [sum]
        dists.append(dist)
        
    # make box and whisker plot of different experiments
    plt.figure(figsize=(5,2.5))
    plt.ylabel('$\overline{\Delta x}$ [m]')
    plt.boxplot(dists, labels=experiment_names_pretty)
    plt.xlabel('Horizon length $N$')
    plt.grid()
        
    if log_scale:
        plt.yscale('log')
    
    if save_fig_flag:
        plt.tight_layout()
        plt.savefig("figs/tracking_error.pdf")
    
def plot_calc_time_box_and_whiskers(experiment_names, experiment_names_pretty, idx_range, log_scale=False, save_fig_flag=False):
    experiment_calc_times = []
    for experiment_name in experiment_names:
        all_calc_times = []
        for i in range(idx_range[0], idx_range[1]):
            state = np.load(path+experiment_name+"/3D/state_"+str(i)+".npy")
            calc_time = state[:,-1]
            all_calc_times += list(calc_time)
        
        experiment_calc_times.append(all_calc_times)
        
    plt.figure(figsize=(6,2))
    plt.xlabel('calculation time [s]')
    plt.boxplot(experiment_calc_times, labels=experiment_names_pretty, vert=False)
    plt.axvline(x=0.1, color='r', linestyle='--')
    plt.grid()

    if log_scale:
        plt.xscale('log')

    if save_fig_flag:
        plt.tight_layout()
        plt.savefig("figs/computation_time.pdf")