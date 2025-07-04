import subprocess
import argparse
import os
import json
import sys
import time
from kevin_test_perf import *

# for each kernel type,
# run the experiment and save to a folder

PYTHON_COMMAND = "python3"
DIR = 'workdir/trials'

# label: (torch2onnx command, flop command)
experiments = {'gemm': {'torch2onnx': ['gemm', '--bs', '4096', '--fp16'], 'flops': lambda: get_gemm_flops(dim=4096)},
               'dualgemm': {'torch2onnx': ['dualgemm', '--bs', '4096', '--fp16'], 'flops': lambda: get_dual_gemm_flops(dim=4096)},
               'batchedgemm': {'torch2onnx': ['batchedgemm', '--bs', '4096', '--fp16'], 'flops': lambda: get_batched_gemm_flops(dim=4096, L=4)},
               'selfattnvanilla': {'torch2onnx': ['selfattn', '--bs', '16', '--fp16'], 'flops': lambda: get_self_attn_flops(16, 1024, 16, 64)},
               'selfattneasy': {'torch2onnx': ['selfattneasy', '--bs', '16', '--fp16'], 'flops': lambda: get_self_attn_flops(16, 1024, 16, 64)}}


def run_experiment(label, save_dict):
    sys.stdout = sys.__stdout__
    print(f'Running {label}')
    save_dir = make_folder(label)
    output_log = os.path.join(save_dir, 'output.log')
    with open(output_log, 'w') as sys.stdout:
        # print('Running torch2onnx')
        run_torch2onnx(label, save_dir)
        # print('\nRunning engine')
        start = time.time()
        run_engine(save_dir)
        time_elapsed = time.time() - start
        print('\nRunning Benchmarking\n')
        output = run_benchmark(save_dir, label)
        output['compile_time_s'] = time_elapsed
        save_dict[label] = output


def make_folder(label):
    save_path = os.path.join(DIR, label)
    if not os.path.exists(save_path):
        print(f'making dir {save_path}')
        os.makedirs(save_path)
    return save_path


def run_torch2onnx(label, save_path):
    command = [PYTHON_COMMAND, './testing/torch2onnx.py'] + \
        experiments[label]['torch2onnx'] + ['--prefix', save_path]
    # print(command)
    subprocess.call(command, stdout=sys.stdout)


def run_engine(dir):
    subprocess.call([PYTHON_COMMAND, './testing/relay_test.py',
                    dir], stdout=sys.stdout, stderr=sys.stderr)


def run_benchmark(dir, label):
    prefix = dir
    inputs, outputs_ref = ref_output(os.path.join(prefix, "model.onnx"))

    rt_mod = setup_welder(prefix, inputs)
    rt_mod.run()

    bench = rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False)
    print(bench)
    median_s = bench.median

    flops = experiments[label]['flops']()
    print(f'Median time: {round(1e3 * median_s, 3)} ms')
    print(f'Median GFLOPs: {round((flops / median_s) * 1e-9, 2)}')

    outputs = []
    for i in range(rt_mod.get_num_outputs()):
        out = rt_mod.get_output(i).asnumpy()
        outputs.append(out)

    print('Correctness checking...')
    max_diff = get_max_diff(outputs, outputs_ref)

    outputs = np.array([output.flatten() for output in outputs]).flatten()
    outputs_ref = np.array([output.flatten()
                           for output in outputs_ref]).flatten()
    print('Sample outputs:')
    print(outputs[-8:])
    print(outputs_ref[-8:])
    print("Output max diff : ", max_diff)
    return {'max_diff': float(max_diff), 'gflops': float(flops/median_s * 1e-9)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--python-command", type=str, default=PYTHON_COMMAND,
                    help=f"Specify what the python command is. Default is {PYTHON_COMMAND}")
    ap.add_argument("--keepfiles", action="store_true",
                    help="Keep model.so etc. files after the run")
    args = ap.parse_args()
    PYTHON_COMMAND = args.python_command

    save_dict = dict()
    for workload in experiments.keys():
        run_experiment(workload, save_dict)

    # print(save_dict)

    with open(os.path.join(DIR, 'results.json'), 'w') as f:
        json.dump(save_dict, f, indent=4)
