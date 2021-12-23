import matplotlib.pyplot as plt
import json
import os
from copy import copy

# collect data
dataset = '3DPW'
model = 'VIBE'
dir_path = 'eval'
results_list = ['eval/eval_input_dilation.json', 'eval/eval_output_dilation.json', 'eval/eval_diff_interp.json', 'eval/eval_smooth.json', 'eval/eval_no_exp.json', 'eval/eval_interp.json', ]  # 'results/seqlen_64/eval.json',]
data_dict = {}
for res in results_list:
    with open(res, 'r') as f:
        data_dict.update(json.load(f))
metrics = list(data_dict['no_interp'].keys())
interp_types = [d for d in data_dict.keys() if d != 'no_interp']
for type_name in interp_types:
    data_dict[type_name].insert(0, [1, data_dict['no_interp']])

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