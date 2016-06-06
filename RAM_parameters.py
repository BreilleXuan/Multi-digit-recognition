
save_dir = "save-3scales/"
save_prefix = "save"
start_step = 10000
load_path = None
# load_path = save_dir + "save.ckpt"

minRadius = 12 # zooms -> minRadius * 2**<depth_level>
sensorBandwidth =  15# fixed resolution of sensor
sensorArea = sensorBandwidth**2
depth = 3 # zooms
channels = 3 # grayscale
totalSensorBandwidth = depth * sensorBandwidth * sensorBandwidth * channels
batch_size = 2

hg_size = 128
hl_size = 128

# g_size = 256
g_size = 128
cell_size = 256
cell_out_size = cell_size

max_length = 5
glimpses = 5
total_step = max_length * glimpses
n_classes = 10

lr = 1e-2
max_iters = 1000000

width = 160
height = 64

loc_sd = 0.1

lmda = 1. #cost weights
