#!/usr/bin/env python
"""
Script to run training with accelerate launch.
This replaces the command:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 16663 \
--multi_gpu --num_processes 8 --num_machines 1 --mixed_precision fp16 \
train_t2i.py --config=configs/t2i_training_demo.py 2>&1 | tee log.txt
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    
    # Build the accelerate launch command
    cmd = [
        'accelerate', 'launch',
        '--main_process_port', '16663',
        '--multi_gpu',
        '--num_processes', '8',
        '--num_machines', '1',
        '--mixed_precision', 'fp16',
        'train_t2i.py',
        '--config=configs/t2i_training_demo.py'
    ]
    
    # Log file path
    log_file = '/storage/v-jinpewang/az_workspace/junchao/Crossflow_org/Crossflow/log.txt'
    
    print(f"Starting training at {datetime.now()}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Output will be saved to {log_file}")
    print("-" * 80)
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as log_f:
            # Run the command, streaming output to both console and log file
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output line by line
            for line in process.stdout:
                # Print to console
                print(line, end='')
                # Write to log file
                log_f.write(line)
                log_f.flush()
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print("-" * 80)
                print(f"Training completed successfully at {datetime.now()}")
            else:
                print("-" * 80)
                print(f"Training failed with return code {return_code} at {datetime.now()}")
                sys.exit(return_code)
                
    except KeyboardInterrupt:
        print("\n" + "-" * 80)
        print("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print("-" * 80)
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()