import tensorflow as tf

tf.__version__

# !nvidia-smi

# GPU info
# !pip install -q gputil
# !pip install -q psutil
# !pip install -q humanize
import psutil as ps  # library for retrieving information on running processes and system utilization
import humanize as hm  # library for turning a number into a fuzzy human-readable
import os  # library for operations with operating system
import GPUtil as GPU  # access to GPU subsystem

process = ps.Process(os.getpid())
print(
    f'Gen RAM Free: {hm.naturalsize(ps.virtual_memory().available)} | Proc size: {hm.naturalsize(process.memory_info().rss)}')

GPUs = GPU.getGPUs()  # get number of GPUs
if GPUs and len(GPUs) >= 1:
    gpu = GPUs[0]
    if (gpu):
        print(f'GPU Model: {gpu.name}')
        print('GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util: {2:3.0f}% | Total: {3:.0f}MB'.format(gpu.memoryFree,
                                                                                                      gpu.memoryUsed,
                                                                                                      gpu.memoryUtil * 100,
                                                                                                      gpu.memoryTotal))

# XXX: only one GPU on Colab and isn’t guaranteed
print(f'Number of GPUs: {GPUs}')

devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')

print(gpus)

if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    print("GPU details: ", details)
