import subprocess
import numpy as np
from simtools import *
from metrics import *
from simulation import *
import argparse
import os

def execute_array_task():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
    job_name = os.environ['SLURM_JOB_NAME']

    cluster_params_filename = f'cluster_params_{slurm_id}.json'

    cluster_params_dict = load_dict(cluster_params_filename)

    value, repeat = exp_values_from_index(cluster_params_dict, task_id)

    sim_params = SimulationParameters(cluster_params_dict['sim_params_dict'])

    # Adjust some things for cluster
    sim_params.output_folder = f'{job_name}_{slurm_id}'
    sim_params.output_base_filename = f'{job_name}_{task_id}'

    net = Network(sim_params.net_params_dict)
    
    # Set network variable
    setattr(net, cluster_params_dict['variable'], value)

    out = simulate(net, sim_params)

    out.result = trueposxtrueneg(net, out, sim_params)

    save_experiment(net, out, sim_params)

def launch_array_job():

    net_params_dict = load_dict(DEFAULT_NET_PARAMS_DICT_FILENAME)

    sim_params_dict = load_dict(DEFAULT_SIM_PARAMS_DICT_FILENAME)
    sim_params_dict['net_params_dict'] = net_params_dict

    #cluster_params_dict = load_dict(DEFAULT_CLUSTER_PARAMS_DICT_FILENAME)
    #cluster_params_dict['sim_params_dict'] = sim_params_dict

    cluster_params_dict = {
        'variable'      :   'fgi',
        'var_min'       :   0.0200,
        'var_max'       :   0.0224,
        'var_step'      :   0.0002,
        'repeats'       :   10,
        'job_name'      :   'lowfgi',
        'running_ntasks':   20,
        'sim_params_dict':  sim_params_dict,
    }

    tmp_cluster_filename = 'tmp_cluster_params.json'
    save_dict(tmp_cluster_filename, cluster_params_dict)

    ntasks = calc_ntasks(cluster_params_dict)
    running_tasks_max = cluster_params_dict.get('running_ntasks')
    job_name = cluster_params_dict.get('job_name')

    sbatch_params = {
        'job-name'  :   f'{job_name}',
        'array'     :   f'1-{ntasks}%{running_tasks_max}',
        #'time'      :   '01:00:00',
        #'nodelist' :   'r520-2',
        #'mem'       :   '3G',
        'partition' :   'batch',
        'ntasks'    :   '1',
        'output'    :   f'{job_name}_%A_%a.out',
        'mail-type' :   'end',
        'mail-user' :   'joshua.arnold1@uqconnect.edu.au',
    }

    ## Build and write sbatch script
    script = generate_header(tmp_cluster_filename, sbatch_params)

    sbatch_filename = f'{job_name}.sbatch'
    with open(sbatch_filename, "w") as f:
        f.write(script)

    # Launch new sbatch script
    proc = subprocess.Popen(f"sbatch {sbatch_filename}", shell=True)


def generate_header(tmp_exp_filename, sbatch_params):
    header = ['#!/bin/bash\n']
    for param, value in sbatch_params.items():
        header.append(f'#SBATCH --{param}={value}')

    script = '\n'.join(header)
    script += '\n' * 2
    script += '\n'.join(['echo \"sbatch started.\"',
            'OUTPUT_DIR=\"$SLURM_JOB_NAME\"_$SLURM_ARRAY_JOB_ID',
            'CLUSTER_PARAM_FILENAME=\"cluster_params_$SLURM_ARRAY_JOB_ID.json\"',
            'if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then',
            '  mkdir $OUTPUT_DIR',
            f'  mv {tmp_exp_filename} $CLUSTER_PARAM_FILENAME',
            'fi',
            '',
            '# wait until first task has moved params file',
            'counter=1',
            'while [ ! -e $CLUSTER_PARAM_FILENAME ]; do',
            '  if [ $counter -ge 10 ]; then',
            '    echo "Could not find param file after long sleep, exiting."',
            '    exit 1',
            '  fi',
            '  echo "Cannot find params, sleep for 5 seconds..."',
            '  sleep 5',
            '  counter=$(( $counter + 1 ))',
            'done',
            '',
            'srun python3 runcluster.py -t',
            '',
            'if [[ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]]; then',
            #f'  srun python3 runcluster.py {COMBINEFLAG} -f $OUTPUT_DIR',
            '  mv $SLURM_JOB_NAME.sbatch $OUTPUT_DIR/',
            '  mv $CLUSTER_PARAM_FILENAME $OUTPUT_DIR/',
            'fi',
            'mv "$OUTPUT_DIR"_"$SLURM_ARRAY_TASK_ID".out $OUTPUT_DIR/',
    ])

    return script

def calc_ntasks(cluster_params_dict):
    start = cluster_params_dict.get('var_min')
    end = cluster_params_dict.get('var_max')
    step_size = cluster_params_dict.get('var_step')
    repeats = cluster_params_dict.get('repeats')
    return int(round((end - start) / step_size)) * repeats

def exp_values_from_index(cluster_params_dict, index):
    start = cluster_params_dict.get('var_min')
    end = cluster_params_dict.get('var_max')
    step_size = cluster_params_dict.get('var_step')

    var_range = int(round((end - start) / step_size))
    value_idx = (index - 1) % var_range # Array index start at 1, make 0 indexed
    repeat = (index - 1) // var_range

    # Python float inaccuracies make this problematic
    # TODO : Consider fixing this issue
    value = round(value_idx * step_size + start, 4)
    
    return value, repeat


def combine(results_folder):
    slurm_id = results_folder.strip('/').split('_')[-1]
    cluster_params_dict = load_dict(f'{results_folder}/cluster_params_{slurm_id}.json')

    ntasks = calc_ntasks(cluster_params_dict)

    result = []
    for task_id in range(1, ntasks+1):
        filepath = f'{results_folder}/{cluster_params_dict["job_name"]}_{task_id}.npz'
        print('Open file to combine: ', filepath)
        out = load_experiment(filepath)
        result.append(out.result)
    
    result_matrix = np.array(result).reshape((cluster_params_dict['repeats'], -1))
    combine_filename = f'{results_folder}/combined_{cluster_params_dict["job_name"]}.npz'
    print('Save combined as: ', combine_filename)
    np.savez(combine_filename, result=result_matrix, cluster_params_dict=cluster_params_dict)


def runcluster():
    parser = argparse.ArgumentParser(description="Tools for running simulations on a slurm managed cluster.")
    parser.add_argument('-c', COMBINEFLAG, help='Combine npz files at filepath', action="store_true")
    parser.add_argument('-f', '--folder', help='The folder to combine', type=str)
    parser.add_argument('-r', '--runslurm', help='Run a full slurm array job', action="store_true")
    parser.add_argument('-t', '--task', help='Run a slurm array task', action="store_true")
    args = parser.parse_args()
    
    if args.combine:
        print('--------- Combine triggered, combining!')
        combine(args.folder)

    if args.task:
        print('--------- Run an array task, tasking')
        execute_array_task()
    
    if args.runslurm:
        print('--------- Run slurm triggered, slurming')
        launch_array_job()


if __name__ == '__main__':
    runcluster()
