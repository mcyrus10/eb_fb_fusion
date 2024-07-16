import metavision_sdk_core
from metavision_sdk_base import EventCD
from metavision_core.event_io import RawReader
from metavision_sdk_core import BaseFrameGenerationAlgorithm

from pathlib import Path
from tifffile import imread
from tqdm import tqdm
from scipy.signal import find_peaks, convolve
from scipy import ndimage as ndimage_cpu
from scipy.optimize import least_squares
from cupyx.scipy import ndimage
from PIL import Image
from mpi4py import MPI


import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Evanescent Data Set Specifications
data_radius =  [51, 51, 51, 101, 101, 101, 122, 122, 122]
data_fps =     [82, 82, 163, 82, 203, 203,  82,  82,  82]
data_version = [1,  2,   3,  1 ,  2,   3,   1,  2,    3]

data_radius =  [ 101, 122, 122, 122]
data_fps =     [ 203,  82,  82,  82]
data_version = [  3,   1,  2,    3]




def fetch_data_sets(data_path, radius, fps, version) -> tuple:
    eb_regex = f"ps_{radius}_v{version}*.raw"
    eb_file = sorted(list((data_path / "event-based").glob(eb_regex)))
    assert len(eb_file) == 1, f"non unique event based file; {eb_file}"
    eb_file = eb_file[0]
    print(eb_file)
    eb = RawReader(eb_file._str)

    fb_regex = f"PS{radius}nm_{fps}fps_V{version}.tif"
    fb_file = sorted(list((data_path / "frame-based").glob(fb_regex)))
    assert len(fb_file) == 1, f"non unique frame based file; {fb_regex}; {fb_file}"
    fb_file = fb_file[0]
    print(fb_file)
    fb = imread(fb_file)
    fb = fb - np.median(fb, axis = 0)

    return eb, fb


def sum_event_based(event_based_data, n_steps, dt) -> np.array:
    eb.reset()
    image = np.zeros([height,width,3], dtype = np.uint8)
    sums = np.zeros([n_steps,3], dtype = np.float32)
    for j in tqdm(range(n_steps)):
        image[:] = 0
        temp = eb.load_delta_t(dt)
        BaseFrameGenerationAlgorithm.generate_frame(temp, image)
        image_cp = cp.array(image, dtype = cp.uint8)
        sums[j] = image_cp.sum(axis = (0,1)).get()
    image_cp = None

    return sums


def make_synchronization_plot(time_eb,
                              time_fb,
                              peaks_eb,
                              peaks_eb_edge,
                              peaks_fb,
                              dt,
                              radius,
                              fps,
                              version) -> None:
    """
    """
    fig,ax = plt.subplots(2,1)
    ax[0].plot(time_eb[:,None], sums)
    ax[0].twinx().plot(time_eb[:,None], conv_event, color = 'r')
    for p in peaks_eb[0]:
        ax[0].axvline(p*dt, color = 'k')
        print(p*dt/1e6, p*dt, p)
    for p in peaks_eb_edge[0]:
        ax[0].axvline(p*dt, color = 'k', linestyle = '--')
        print(p*dt/1e6, p*dt, p)

    ax[1].plot(time_fb, fb_shutter_edge)
    ax2 = ax[1].twinx()
    ax2.plot(time_fb, conv_frame, alpha = 0.25, color = 'r')

    for p in peaks_fb[0]:
        ax[1].axvline(p/fps, color = 'k')

    for a in ax:
        a.set_xlabel("time (s)")

    ax[0].set_ylabel("Event Camera Intensity")
    ax[1].set_ylabel("Frame Camera Intensity")
    ax2.set_ylabel("Conv Edge Find Mag")
        
    fig.tight_layout()
    path_str = Path("/home/mcd4/Documents/Experimentation/eb_fb_fusion/time_synced/evanescent")
    id_handle = f"ps_{radius}_fps_{fps}_v{version}"
    im_f_name = path_str / id_handle / f"{id_handle}_sync.png"
    fig.savefig(im_f_name)

    csv_f_name = path_str / id_handle / f"{id_handle}_sync.csv"
    with open(csv_f_name, mode = 'w') as file:
        file.write("Event, Frame\n")
        file.write(f"{peaks_eb_edge[0][-1]*dt}, {peaks_fb[0][-1]/fps}\n")


def write_synchronized_image_sets(eb, fb, peaks_eb_edge, peaks_fb, radius, fps, version):
    path_str = "/home/mcd4/Documents/Experimentation/eb_fb_fusion/time_synced/evanescent"
    path_ext = f"ps_{radius}_fps_{fps}_v{version}"
    base_path = Path(path_str) / path_ext
    if not base_path.is_dir():
        os.mkdir(base_path)
        os.mkdir(base_path/"event_based")
        os.mkdir(base_path/"frame_based")

    # Reset and step back up to the moment that they are synchronized
    dt_synced = int(round(1e6/fps))
    eb.reset()
    eb_time_init = int(round(peaks_eb_edge[0][-1]*dt))
    eb.seek_time(eb_time_init)
    image = np.zeros([height,width,3], dtype = np.uint8)
    med_kernel = (5,5,1)
    q = 0
    for j in tqdm(range(peaks_fb[0][-1]-1, fb.shape[0])):
        fb_local = cp.array(fb[j], dtype = cp.float32)
        fb_local = ndimage.median_filter(fb_local, (5,3)).get()
        f_name_fb = base_path / f"frame_based/image_{q:06d}.tif"
        Image.fromarray(fb_local).save(f_name_fb)
        image[:] = 0
        temp = eb.load_delta_t(dt_synced)
        BaseFrameGenerationAlgorithm.generate_frame(temp, image)
        image = ndimage.median_filter(cp.array(image, dtype = cp.uint8), med_kernel).get()
        f_name_eb = base_path / f"event_based/image_{q:06d}.tif"
        Image.fromarray(image).save(f_name_eb)
        q += 1


if __name__ == "__main__":
    data_path = Path("/home/mcd4/Data/FolderForMIDSCAN/evanescent_2024-05-24")
    print(data_path.is_dir())

    for i, (radius, fps, version) in tqdm(enumerate(zip(data_radius, data_fps, data_version))):
        if i % size != rank:
            continue
        print(radius, fps, version)
        eb, fb = fetch_data_sets(data_path, radius, fps, version)
        height, width = eb.get_size()
        n_steps = 1000
        dt = 5_000
        plotting = True
        sums = sum_event_based(eb, n_steps, dt)
        print("a")


        # Edge Finding Kernel
        kernel = np.array([-1,0,1])

        conv_event = np.abs(convolve(sums[:,0], kernel, mode = 'same'))
        peaks_eb = find_peaks(sums[:,0], height = 1e8, distance = 10)
        peaks_eb_edge = find_peaks(conv_event, height = 1e8, distance = 10)
        time_eb = np.arange(0,n_steps*dt,dt)
        time_fb = np.arange(0,fb.shape[0]*(1/fps),1/fps)

        fb_shutter_edge = fb.sum(axis = (1,2)).astype(np.float32)
        conv_frame = np.abs(convolve(fb_shutter_edge, kernel, mode = 'same'))

        peaks_fb = find_peaks(conv_frame, height = 1e8)

        write_synchronized_image_sets(eb, fb, peaks_eb_edge, peaks_fb, radius, fps, version)

        if plotting:
            make_synchronization_plot(time_eb,time_fb,peaks_eb,peaks_eb_edge,
                              peaks_fb,dt,radius,fps,version)

