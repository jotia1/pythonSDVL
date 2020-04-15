import json
import subprocess
import math
import numpy as np
import sys
from simtools import *

# JOBNAME_SLURMID/JOBNAME_SLURMID_TASKID

def run_slurm_array_job():
    exp_params = {
        'variable'      :   'frq',
        'var_min'       :   1,
        'var_max'       :   3,
        'var_step'      :   1,
        'repeats'       :   3,
        'job_name'      :   'frq_test',
        'running_ntasks':   6,
        'sim_time_sec'  :   11,
        'time_execution':   False,
        'Tp'            :   50,
        'Df'            :   10,
        'Pf'            :   5,
        'p_inp'         :   None,
        'p_ts'          :   None,
        'inp_idxs'      :   None,
        'inp_ts'        :   None,
        'inp_type'      :   STANDARDINPUT,
        'data_fcn'      :   None,
        'voltages_to_save': [2000],
        'delays_to_save':   [],
        'variances_to_save':[],    
    }

    tmp_exp_filename = 'tmp_exp_params.json'
    write_exp_param_file(tmp_exp_filename, exp_params)

    ntasks = calc_ntasks(exp_params)
    running_tasks_max = exp_params.get('running_ntasks')
    job_name = exp_params.get('job_name')

    sbatch_params = {
        'job-name'  :   f'{job_name}',
        'array'     :   f'1-{ntasks}%{running_tasks_max}',
        'time'      :   '01:00:00',
        'partition' :   'batch',
        'ntasks'    :   '1',
        'output'    :   f'{job_name}_%A_%a.out',
    }

    ## Build and write sbatch script
    script = generate_header(tmp_exp_filename, sbatch_params)

    sbatch_filename = f'{job_name}.sbatch'
    with open(sbatch_filename, "w") as f:
        f.write(script)

    # Launch new sbatch script
    proc = subprocess.Popen(f"sbatch {sbatch_filename}", shell=True)


def generate_header(tmp_exp_filename, sbatch_params):
    script = make_sbatch_header(sbatch_params)
    script += '\n' * 2
    script += '\n'.join(['echo \"script running\"',
            'OUTPUT_DIR=\"$SLURM_JOB_NAME\"_$SLURM_ARRAY_JOB_ID',
            'EXP_PARAM_FILENAME=\"exp_params_$SLURM_ARRAY_JOB_ID.json\"',
            'if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then',
            '  mkdir $OUTPUT_DIR',
            f'  mv {tmp_exp_filename} $EXP_PARAM_FILENAME',
            'fi',
            'counter=1',
            'while [ ! -e $EXP_PARAM_FILENAME ]; do',
            '  if [ $counter -ge 10 ]; then',
            '    echo "Could not find param file after long sleep, exiting."',
            '    exit 1',
            '  fi',
            '  echo "Cannot find params, sleep for 5 seconds..."',
            '  sleep 5',
            '  counter=$(( $counter + 1 ))',
            'done',
            'srun python3 runexperiment.py $EXP_PARAM_FILENAME $SLURM_ARRAY_TASK_ID',
            'if [[ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]]; then',
            #f'  srun python3 runcluster.py {COMBINEFLAG} $EXP_PARAM_FILENAME $SLURM_ARRAY_JOB_ID',
            '  mv $SLURM_JOB_NAME.sbatch $OUTPUT_DIR/',
            '  mv $EXP_PARAM_FILENAME $OUTPUT_DIR/',
            'fi',
            'mv "$OUTPUT_DIR"_"$SLURM_ARRAY_TASK_ID".out $OUTPUT_DIR/',
    ])

    return script



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
    value_idx = (index - 1) % var_range # Array index start at 1, make 0 indexed
    repeat = (index - 1) // var_range

    # Python float inaccuracies make this problematic
    # TODO : Consider fixing this issue
    value = round(value_idx * step_size + start, 4)
    
    return value, repeat

# def combine_results(exp_params_filename, slurmid):
#     output_folder = exp_params_filename.strip('.json')
#     exp_params = load_exp_param_file(exp_params_filename)
#     print(exp_params)

#     ntasks = calc_ntasks(exp_params);
#     results = []
#     for taskid in range(ntasks):
#         value, repeat = exp_values_from_index(exp_params, taskid)
#         file_str = get_sim_full_filepath(exp_params.get('job_name'), slurmid, taskid) + '.npz'
#         data = np.load(file_str)
#         results.append(data['results'])
    
#     return np.array(results).reshape((-1, exp_params.get('repeats')))


def main():
    print("run cluster argv: ", sys.argv)
    if len(sys.argv) == 4 and sys.argv[1] == COMBINEFLAG:
        #print(combine_results(sys.argv[2], sys.argv[3]))
        raise NotImplementedError()
    else:
        run_slurm_array_job()

if __name__ == '__main__':
    main()
