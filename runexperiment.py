import sys
from runner import *

def main():

    array_idx = 0
    if len(sys.argv) == 3:
        array_idx = int(sys.argv[2])
    
    assert len(sys.argv) > 1, "Must provide experiment file as argument"

    exp_filename = sys.argv[1]

    exp_params = load_exp_param_file(exp_filename)

    value, repeat = exp_values_from_index(exp_params, array_idx)

    print(f'(value, repeat) : ({value}, {repeat})')


if __name__ == '__main__':
    main()
