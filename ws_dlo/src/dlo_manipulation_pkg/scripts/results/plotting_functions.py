import matplotlib.pyplot as plt
import numpy as np

path = "/home/ruan/sandbox/cca_workspace/"
def plot_time_series(experiment_name, idx_range, metric_name, save_fig_flag=False):
    plt.figure(figsize=(6,3))
    plt.xlabel('time [s]')
    for i in range(idx_range[0], idx_range[1]):
        state = np.load(path+"/results/"+experiment_name+"/state_"+str(i)+".npy")
        t = [i*0.1 for i in range(state.shape[0])]
        if metric_name == "tracking_error":
            plt.ylabel('total tracking error [m]')
            metric = [np.linalg.norm(x) for x in state[:,1:31] - state[:,-30:]]
        if metric_name == "fps_error":
            plt.ylabel('average tracking error [m]')
            metric = []
            for k in range(state.shape[0]):
                if experiment_name in ["gain1", "baseline_elim", "nmpc", "original_10points", "baseline_elim_10points"]:
                    position_error = state[k,1+3:31-3] - state[k,-30+3:-3]
                else:
                    position_error = state[k,1+3:31-3] - state[k,-31+3:-3-1]
                # if k == 0: err_at_start = position_error
                num_fps = 8
                fps_error_sum = 0
                for j in range(num_fps):
                    fps_error = np.linalg.norm(position_error[j*3:j*3+3])
                    fps_error_sum += fps_error
                metric.append(fps_error_sum / num_fps)
                    
        plt.plot(t,metric)
    # draw a dashed red line at y=0.05
    # plt.axhline(y=0.01, color='r', linestyle='--')
    plt.grid()
        
    if save_fig_flag:
        plt.tight_layout()
        plt.savefig(path+"/results/figs/tracking_error_"+experiment_name+".pdf")


def plot_tracking_error_box_and_whiskers(experiment_names, experiment_names_pretty, idx_range, log_scale=False,save_fig_flag=False):
    dists = []
    for experiment_name in experiment_names:
        dist = []
        for i in range(idx_range[0], idx_range[1]):
            state = np.load(path+"/results/"+experiment_name+"/state_"+str(i)+".npy")
            N = state.shape[0]
            num_fps = 8
            output = [np.linalg.norm(x) for x in state[:,1+3:31-3] - state[:,-31+3:-3-1]]
            sum = np.array(output).sum()/N/num_fps
            dist += [sum]
        dists.append(dist)
        
    # make box and whisker plot of different experiments
    plt.figure(figsize=(6,3))
    plt.ylabel('tracking error [m]')
    plt.boxplot(dists, labels=experiment_names_pretty)
    # plt.boxplot(dists, labels=np.array([1,3,5,7,10,12,15,10])*0.1)
    plt.xlabel('Horizon length')
    plt.grid()
        
    if log_scale:
        plt.yscale('log')
    
    if save_fig_flag:
        plt.tight_layout()
        plt.savefig(path+"/results/figs/tracking_error.pdf")
    
def plot_calc_time_box_and_whiskers(experiment_names, experiment_names_pretty, idx_range, log_scale=False, save_fig_flag=False):
    experiment_calc_times = []
    for experiment_name in experiment_names:
        all_calc_times = []
        for i in range(idx_range[0], idx_range[1]):
            state = np.load(path+"/results/"+experiment_name+"/state_"+str(i)+".npy")
            calc_time = state[:,-1]
            all_calc_times += list(calc_time)
        
        experiment_calc_times.append(all_calc_times)
        
    plt.figure(figsize=(6,3))
    plt.ylabel('calculation time [s]')
    plt.boxplot(experiment_calc_times, labels=experiment_names_pretty)
    plt.axhline(y=0.1, color='r', linestyle='--')
    plt.grid()

    if log_scale:
        plt.yscale('log')

    if save_fig_flag:
        plt.tight_layout()
        plt.savefig(path+"/results/figs/computation_time.pdf")
        
        
# plot of percentage time within ball of eps radius during last 15sdef plot_time_series(experiment_name, idx_range, output_name):
def plot_box_whisker(experiment_names, experiment_names_pretty, idx_range, metric_name):
    dists = []
    for experiment_name in experiment_names:
        dist = []
        for i in range(idx_range[0], idx_range[1]):
            state = np.load(path+"/results/"+experiment_name+"/state_"+str(i)+".npy")
            if metric_name == "fps_error":
                sum = 0
                for k in range(150,state.shape[0]):
                    position_error = state[k,1:31] - state[k,-30:]
                    # if k == 0: err_at_start = position_error
                    num_fps = 10
                    for j in range(num_fps):
                        fps_error = np.linalg.norm(position_error[j*3:j*3+3])
                        # fps_error_norm = fps_error / np.linalg.norm(err_at_start[j*3:j*3+3])
                        
                        sum += fps_error
                        # sum += fps_error_norm
                # output = [np.linalg.norm(x) for x in state[:,1:31] - state[:,-30:]]
                # output = [np.linalg.norm(x) for x in state[:-idx_from_end,1:31] - state[:-idx_from_end,-30:]]
                sum = sum / state.shape[0] / num_fps
            dist += [sum]
            # dist += output
            
        dists.append(dist)
        
    # make box and whisker plot of different experiments
    plt.figure(figsize=(10,5))
    plt.ylabel('tracking error [m]')
    plt.boxplot(dists, labels=experiment_names_pretty)


# def get_number_of_successful_trials(experiment_name, idx_range, threshhold, start_time):
#     num_successes = 0
#     for i in range(idx_range[0], idx_range[1]):
#         state = np.load(path+"/results/"+experiment_name+"/state_"+str(i)+".npy")
#         trial_successful = True
#         for j in range(int(start_time/0.1),state.shape[0]):
#             position_error = state[j,1:31] - state[j,-30:]
#             num_fps = 10
#             for k in range(num_fps):
#                 fps_error = np.linalg.norm(position_error[k*3:k*3+3])
#                 if fps_error > threshhold: trial_successful = False
#         if trial_successful: num_successes += 1
#     print(num_successes)

def get_number_of_successful_trials(experiment_names, experiment_names_pretty, idx_range, threshhold):
    time_vecs = []
    for experiment_name in experiment_names:
        time_vec = []
        for i in range(idx_range[0], idx_range[1]):
            state = np.load(path+"/results/"+experiment_name+"/state_"+str(i)+".npy")
            inside_threshhold_vec = []
            for j in range(state.shape[0]):
                if experiment_name == "gain1":
                    position_error = state[j,1+3:31-3] - state[j,-30+3:-3]
                else:
                    position_error = state[j,1+3:31-3] - state[j,-30+3-1:-1-3]
                num_fps = 10
                inside_threshhold = True
                for k in range(num_fps):
                    fps_error = np.linalg.norm(position_error[k*3:k*3+3])
                    if fps_error > threshhold: inside_threshhold = False
                inside_threshhold_vec.append(inside_threshhold)
            # Find idx where all values remaining are true
            idx = -1
            for j in range(1,len(inside_threshhold_vec)):
                if inside_threshhold_vec[-j] == False:
                    idx = j
                    break
            dt = 0.1
            time_val = (state.shape[0]-idx)*dt
            time_vec.append(time_val)
        time_vecs.append(time_vec)
    # boxplot of time_vec
    plt.figure(figsize=(10,5))
    plt.ylabel('time [s]')
    plt.boxplot(time_vecs, labels=experiment_names_pretty)
    
    # compute how many of them do it within 15s, and plot histogram
    num_successes_vec = []
    for time_vec in time_vecs:
        num_successes = len(np.where(np.array(time_vec) < 20)[0])
        num_successes_vec.append(num_successes)
    print(num_successes_vec)
    
    