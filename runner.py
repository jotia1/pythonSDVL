import json
import subprocess
import math

def make_sbatch_header(sbatch_params):
    header = ['#!/bin/bash\n']

    for param, value in sbatch_params.items():
        header.append(f'#SBATCH --{param}={value}')

    return '\n'.join(header)

def calc_ntasks(exp_params):
    start = exp_params.get('var_min')
    end = exp_params.get('var_max')
    step_size = exp_params.get('var_step')
    repeats = exp_params.get('repeats')

    return int(round((end - start) / step_size) * repeats)

def write_exp_param_file(filename, exp_params):
    with open(filename, 'w') as fp:
        json.dump(exp_params, fp, sort_keys=True, indent=4)

def load_exp_param_file(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def exp_values_from_index(exp_params, index):
    start = exp_params.get('var_min')
    end = exp_params.get('var_max')
    step_size = exp_params.get('var_step')

    var_range = int(round((end - start) / step_size))
    value = index % var_range
    repeat = index // var_range
    
    return value, repeat

def run_test_cluster():
    exp_params = {
        'variable'  :   'test',
        'var_min'   :   0,
        'var_max'   :   4,
        'var_step'  :   1,
        'repeats'   :   1,
        'job_name'  :   'test_job'
    }

    tmp_exp_filename = 'tmp_exp_params_file.json'
    write_exp_param_file(tmp_exp_filename, exp_params)

    ntasks = calc_ntasks(exp_params)
    running_tasks_max = 2
    job_name = exp_params.get('job_name')

    sbatch_params = {
        'job-name'  :   f'{job_name}',
        'array'     :   f'1-{ntasks}%{running_tasks_max}',
        'time'      :   '01:00:00',
        'partition' :   'batch',
        'ntasks'    :   '1',
    }

    ## Build and write sbatch script
    script = make_sbatch_header(sbatch_params)
    script += '\n' * 2
    script += '\n'.join([f'mv {tmp_exp_filename} exp_params_$SLURM_JOB_ID.json',
            'echo \"My SLURM_ARRAY_TASK_ID: \" $SLURM_ARRAY_TASK_ID',
            'srun python3 python_wait.py exp_params_$SLURM_JOB_ID.json $SLURM_ARRAY_TASK_ID'])

    sbatch_filename = f'{job_name}.sbatch'
    with open(sbatch_filename, "w") as f:
        f.write(script)

    # Launch new sbatch script
    proc = subprocess.Popen(f"sbatch {sbatch_filename}", shell=True)










class ExperimentRunner():
    def __init__(self, filename=None):
        self.variable = 'fgi'
        self.variable_min = 0.0224
        self.variable_max = 0.0228
        self.variable_step = 0.0001

        if filename:
            self.load_experiment_file(filename)

    def load_experiment_file(self, filename):
        # TODO 
        raise NotImplementedError()

    def save_experiment_file(self, filename):
        # TODO 
        raise NotImplementedError()

class ClusterRunner():
    def __init__(self, er=None):
        self.er = ExperimentRunner()

    def run(self):
        pass