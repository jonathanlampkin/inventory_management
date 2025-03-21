import os
import subprocess
import time
import glob
from datetime import datetime

def get_script_priority(script_name):
    """Define execution priority for scripts based on numerical prefixes."""
    # Scripts are now named with numerical prefixes (01_, 02_, etc.)
    return os.path.basename(script_name)  # Just use the filename for sorting

def run_script(script_path, log_dir):
    """Run a Python script and log its output."""
    script_name = os.path.basename(script_path)
    log_file = os.path.join(log_dir, f"{os.path.splitext(script_name)[0]}_log.txt")
    
    print(f"Running: {script_path}")
    start_time = time.time()
    
    # Create the log file and write header
    with open(log_file, 'w') as f:
        f.write(f"=== EXECUTION LOG: {script_name} ===\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
    
    # Run the script and capture output
    try:
        process = subprocess.Popen(
            ['python', script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        end_time = time.time()
        execution_time = end_time - start_time
        return_code = process.returncode
        
        # Write output to log file
        with open(log_file, 'a') as f:
            f.write("STANDARD OUTPUT:\n")
            f.write(stdout)
            f.write("\n\n")
            f.write("STANDARD ERROR:\n")
            f.write(stderr)
            f.write("\n\n")
            f.write(f"Return Code: {return_code}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Return execution results
        return {
            'script': script_name,
            'return_code': return_code,
            'execution_time': execution_time,
            'stdout': stdout,
            'stderr': stderr
        }
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log the exception
        with open(log_file, 'a') as f:
            f.write("EXCEPTION:\n")
            f.write(str(e))
            f.write("\n\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Return execution results with error
        return {
            'script': script_name,
            'return_code': 1,
            'execution_time': execution_time,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    """Main function to run all scripts in the scripts directory."""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('output', 'execution_logs', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Master log file
    master_log = os.path.join(log_dir, 'master_log.txt')
    
    # Find all Python scripts in the scripts directory
    scripts = glob.glob(os.path.join('scripts', '*.py'))
    
    # Filter out any unwanted scripts (like __init__.py)
    scripts = [s for s in scripts if not os.path.basename(s).startswith('__')]
    
    # Sort scripts by priority (numerical prefix)
    scripts.sort(key=get_script_priority)
    
    # Write header to master log
    with open(master_log, 'w') as f:
        f.write(f"=== MASTER EXECUTION LOG: {timestamp} ===\n")
        f.write(f"Discovered {len(scripts)} scripts to execute:\n")
        for i, script in enumerate(scripts, 1):
            f.write(f"{i}. {os.path.basename(script)}\n")
        f.write("\n")
        f.write("="*50 + "\n\n")
    
    # Run each script and log results
    results = []
    for script in scripts:
        result = run_script(script, log_dir)
        results.append(result)
        
        # Append to master log
        with open(master_log, 'a') as f:
            f.write(f"Script: {result['script']}\n")
            f.write(f"Status: {'SUCCESS' if result['return_code'] == 0 else 'FAILED'}\n")
            f.write(f"Return Code: {result['return_code']}\n")
            f.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
            
            if result['return_code'] != 0:
                f.write("Error Output:\n")
                f.write(result['stderr'])
            
            f.write("\n" + "-"*50 + "\n\n")
    
    # Generate summary
    success_count = sum(1 for r in results if r['return_code'] == 0)
    total_time = sum(r['execution_time'] for r in results)
    
    summary = f"""
=== EXECUTION SUMMARY ===
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Scripts: {len(scripts)}
Successful: {success_count}
Failed: {len(scripts) - success_count}
Total Execution Time: {total_time:.2f} seconds

Individual Results:
"""
    
    for result in results:
        status = "SUCCESS" if result['return_code'] == 0 else f"FAILED (code {result['return_code']})"
        summary += f"- {result['script']}: {status} in {result['execution_time']:.2f}s\n"
    
    # Print summary to console
    print(summary)
    
    # Save summary to master log
    with open(master_log, 'a') as f:
        f.write(summary)
    
    # Also save summary to a separate file for quick reference
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nExecution logs saved to: {log_dir}")
    
    # Return summary code - 0 if all passed, otherwise non-zero
    return 0 if success_count == len(scripts) else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 