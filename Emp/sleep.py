import time
import os

# Sleep for 1 hour (3600 seconds)
time.sleep(1800)

# Command to put the computer to sleep
# For Windows
if os.name == 'nt':
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")