import keyboard
import time
import os

# Check if the script is running with sudo
if os.geteuid() != 0:
    print("This script requires elevated privileges. Please run with sudo.")
    exit()

while True:
    # Your other program logic here
    time.sleep(1);
    print("main task done")
    # Check for key press without blocking
    if keyboard.is_pressed('a'):
        print("Key 'a' pressed!")
    if keyboard.is_pressed('b'):
        print("Key 'b' pressed!")
