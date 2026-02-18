# scripts/run_sweep.py
# Simple runner to execute a grid of training runs and collect final test metrics.
import subprocess
import shlex
import os
import sys

# Grid to run
lrs = [3e-4, 5e-4, 7e-4]
wds = [5e-4, 1e-3, 2e-3]

# Runtime args
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
mixup_alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
augment_flag = True
focal_gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

out_dir = 'sweep_logs'
os.makedirs(out_dir, exist_ok=True)

results = []

for lr in lrs:
    for wd in wds:
        name = f"lr{lr:.0e}_wd{wd:.0e}"
        log_path = os.path.join(out_dir, f"{name}.log")
        cmd = [
            sys.executable, '-u', 'src\main.py',
            '--data', 'data',
            '--multi-subject',
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--lr', str(lr),
            '--weight-decay', str(wd),
            '--mixup-alpha', str(mixup_alpha),
            '--focal-gamma', str(focal_gamma)
        ]
        if augment_flag:
            cmd.append('--augment')
        print('\nRunning:', ' '.join(shlex.quote(c) for c in cmd))
        with open(log_path, 'w', encoding='utf-8') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
            # stream to file and to stdout
            for line in proc.stdout:
                f.write(line)
                f.flush()
                print(line, end='')
            proc.wait()
            rc = proc.returncode
        # After run, try to parse final test results
        acc = None
        kappa = None
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                txt = f.read()
            # Find the FINAL TEST RESULTS block
            if 'FINAL TEST RESULTS' in txt:
                idx = txt.index('FINAL TEST RESULTS')
                tail = txt[idx:]
                # naive parse for Accuracy and Kappa
                for line in tail.splitlines():
                    if 'Accuracy:' in line:
                        # formats like: Accuracy: 0.7301 (73.01%)
                        parts = line.split()
                        try:
                            acc = float(parts[1])
                        except Exception:
                            pass
                    if "Cohen's Kappa:" in line:
                        parts = line.split()
                        try:
                            kappa = float(parts[2])
                        except Exception:
                            pass
                    if acc is not None and kappa is not None:
                        break
        except Exception as e:
            print('Failed to parse log:', e)
        results.append((name, lr, wd, rc, acc, kappa, log_path))

# Print summary
print('\nSweep summary:')
print('name, lr, wd, returncode, acc, kappa, log')
for row in results:
    print(row)

# Save summary
with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
    for row in results:
        f.write(','.join([str(x) for x in row]) + '\n')

print('\nDone. Logs in', out_dir)
