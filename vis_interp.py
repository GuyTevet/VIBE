import matplotlib.pyplot as plt
import json
import os

# collect data
dir_path = 'results/seqlen_64'
dataset = '3DPW'
model = 'VIBE'
results_path = os.path.join(dir_path, 'eval.json')
interp_types = ['nearest', 'linear', 'cubic']
with open(results_path, 'r') as f:
    data_dict = json.load(f)
metrics = list(data_dict['no_interp'].keys())


# visualize data
vis_dir = os.path.join(dir_path, '{}_interp_vis'.format(model))
if not os.path.isdir(vis_dir):
    os.makedirs(vis_dir)
for met in metrics:
    fig = plt.figure()
    fig.suptitle('{} - {} | {} | {}'.format(model, os.path.basename(dir_path), dataset, met))
    ax = fig.add_subplot()
    for type_name in interp_types:
        data_preserved = [1/d[0] for d in data_dict[type_name]]
        metric_value = [d[1][met] for d in data_dict[type_name]]
        ax.plot(data_preserved, metric_value, linestyle='--', marker='o', label=type_name)
        ax.set_xlabel('Data preserved after sampling')
        ax.set_ylabel('Metric value')
    ax.legend()
    out_path = os.path.join(vis_dir, met + '.png')
    fig.savefig(out_path)