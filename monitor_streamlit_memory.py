import subprocess
import psutil
import time

# start streamlit
process = subprocess.Popen(["streamlit", "run", "src/dashboard/streamlit_app.py"])

print("Monitoring memory usage (including child processes)...\n")

peak = 0

while process.poll() is None:
    try:
        parent = psutil.Process(process.pid)

        # get parent + child processes
        processes = [parent] + parent.children(recursive=True)

        mem = 0
        for p in processes:
            try:
                mem += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass

        mem_mb = mem / (1024 ** 2)

        if mem_mb > peak:
            peak = mem_mb

        print(f"Current RAM: {mem_mb:.2f} MB | Peak: {peak:.2f} MB")

    except psutil.NoSuchProcess:
        break

    time.sleep(2)

print(f"\nPeak memory usage: {peak:.2f} MB")