#!/home/mcd4/miniconda3/envs/openeb/bin/python3
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
rank = 0
size = 1
print(f"rank {rank} of {size}")

import metavision_sdk_core
import subprocess
import argparse
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

import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os
from sys import path
path.append("/home/mcd4/Documents/Experimentation/event_camera_emulation")
from common.event_camera_emulation.emulator import EventCameraEmulator


def fetch_data_sets(data_path, radius, fps, version, mode: str) -> tuple:
    """

    Bespoke function for getting polystyrene data..

    """
    eb_regex = f"ps_{radius}_v{version}*.raw"
    eb_file = sorted(list((data_path / "event-based").glob(eb_regex)))
    assert len(eb_file) == 1, f"non unique event based file; {eb_file}"
    eb_file = eb_file[0]
    print(eb_file)
    eb = RawReader(eb_file._str)

    fb_regex = f"ps_{radius}_v{version}_{fps}fps.tif"
    fb_file = sorted(list((data_path / "frame-based").glob(fb_regex)))
    assert len(fb_file) == 1, f"non unique frame based file; {fb_regex}; {fb_file}"
    fb_file = fb_file
    print(fb_file)
    fb = imread(fb_file).astype(np.float32)
    fb = fb - np.median(fb, axis = 0)

    return eb, fb, eb_file


def sum_event_based(eb, n_steps, dt) -> np.array:
    """
    This takes the sum of all the polarities as the indicator of the shutter
    event
    """
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


def make_synchronization_plot(time_eb, time_fb, peaks_eb_edge,
                              peaks_fb, dt, radius, fps, version,mode) -> None:
    """
    Plotting and outputting the csv file
    """
    fig,ax = plt.subplots(2,1)
    ax[0].plot(time_eb[:,None], sums)
    ax[0].twinx().plot(time_eb[:,None], conv_event, color = 'r')
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
    path_str = Path(f"/home/mcd4/Documents/Experimentation/eb_fb_fusion/time_synced/{mode}")
    id_handle = f"ps_{radius}_fps_{fps}_v{version}"
    im_f_name = path_str / id_handle / f"{id_handle}_sync.png"
    fig.savefig(im_f_name)

    csv_f_name = path_str / id_handle / f"{id_handle}_sync.csv"
    with open(csv_f_name, mode = 'w') as file:
        file.write("Event, Frame\n")
        file.write(f"{peaks_eb_edge[0][-1]*dt}, {peaks_fb[0][-1]/fps}\n")


def write_synchronized_image_sets(eb, fb, eb_file_name, peaks_eb_edge,
                                  peaks_fb, radius, fps, version, mode
                                  ) -> None:
    """

    Path/Directory creation and file writing (with some minor filtering)

    """
    path_str = f"/home/mcd4/Documents/Experimentation/eb_fb_fusion/time_synced/{mode}"
    path_ext = f"ps_{radius}_fps_{fps}_v{version}"
    base_path = Path(path_str) / path_ext
    if not base_path.is_dir():
        os.mkdir(base_path)
        os.mkdir(base_path/"event_based")
        os.mkdir(base_path/"frame_based")
        os.mkdir(base_path/"emulated")

    # Reset and step back up to the moment that they are synchronized
    dt_synced = int(round(1e6/fps))
    eb.reset()
    eb_time_init = int(round(peaks_eb_edge[0][-1]*dt_synced))
    eb.seek_time(eb_time_init)
    image = np.zeros([height,width,3], dtype = np.uint8)
    med_kernel = (3,3,1)
    event_inst = EventCameraEmulator()
    print("Not Applying Any Filters to Data")
    desc = 'writing images'
    #                                    So event emulator can get next image
    #                                                             |
    #                                                             v
    for q, j in tqdm(enumerate(range(peaks_fb[0][-1], fb.shape[0]-1)), desc = desc):
        fb_local = fb[j]
        f_name_fb = base_path / f"frame_based/image_{q:06d}.tif"
        Image.fromarray(fb_local).save(f_name_fb)

        grayscale = event_inst.get_events_image_rgb(fb[j+1], fb[j])
        event_em = event_inst.get_visual_events_image(grayscale)
        f_name_em = base_path / f"emulated/image_{q:06d}.tif"
        Image.fromarray(event_em).save(f_name_em)

        image[:] = 0
        temp = eb.load_delta_t(dt_synced)
        BaseFrameGenerationAlgorithm.generate_frame(temp, image)
        f_name_eb = base_path / f"event_based/image_{q:06d}.tif"
        Image.fromarray(image).save(f_name_eb)

    eb_time_final = eb_time_init + q*dt_synced
    eb_time_interval = eb_time_final - eb_time_init
    expected_time_steps = eb_time_interval / dt_synced
    print("q = ", q, "\tqmark = ", expected_time_steps, "\tdt", dt_synced, "\ttime_interval", eb_time_interval)
    print("--->",  eb_time_final, eb_time_init)

    # Write the raw file cut to the start time....for dt fuckery
    #  TIME IS SUPPLIED IN SECONDS TO THIS
    cutter_args = " ".join([
                   "metavision_file_cutter",
                   "-i",str(eb_file_name),
                   "-o",str(base_path / f"{path_ext}_cut.raw"),
                   "-s",str(eb_time_init/1e6),
                   "-e",str(eb_time_final/1e6)
                   ])
    print(cutter_args)
    subprocess.run(cutter_args, shell = True)
    print("DONE")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--mode",
                        type=str,
                        default = "",
                        help = "cytovia or evanescent"
                        )
    parser.add_argument(
            "--condition_number",
            type = int,
            default = 0,
            help = "hack to override the data iterating..."
            )

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    mode = args.mode
    condition = args.condition_number
    print(args.mode)
    print(args.condition_number)
    if mode == 'evanescent':
        ## Evanescent Data Set Specifications
        data_radius =  [51, 51, 51, 101, 101, 101, 122, 122, 122]
        data_fps =     [82, 82, 163, 82, 203, 203,  82,  82,  82]
        data_version = [1,  2,   3,  1 ,  2,   3,   1,  2,    3]
        _t0_event_ = [1.0366, 0.7805, 0.6442, 1.3658,0.4778, 1.0000,0.4146,
                    0.4024, 1.1219]
        _t0_frame_ = [0.7683,0.3780,0.4233,1.1707,0.2759,0.2069,0.2439,0.3049,
                    0.8293]


    elif mode == 'cytovia':
        ## cytovia data set specs
        data_radius =  [ 51, 51, 101, 101, 122, 122, 202, 202]
        data_fps =     [ 82, 50,  40, 199,  40, 102,  40, 199]
        data_version = [  1,  2,   1,  2,    1,   2,   1,   2]
    else:
        assert False, "unknown mode"


    if rank == 0:
        print(f"--> mode: {mode} <--")
    data_path = Path(f"/home/mcd4/Data/FolderForMIDSCAN/{mode}_2024-05-24")
    cytovia_thresh_dict ={
        "ps_51_fps_82_v1":  {'event':0.3e7, 'frame':1e7},
        "ps_51_fps_50_v2":  {'event':0.7e7, 'frame':0.6e8},
        "ps_101_fps_40_v1": {'event':0.3e7, 'frame':1e8},
        "ps_101_fps_199_v2":{'event':1.5e6, 'frame':1e7},
        "ps_122_fps_40_v1": {'event':0.3e7, 'frame':1e8},
        "ps_122_fps_102_v2":{'event':1.5e6, 'frame':1e7},
        "ps_202_fps_40_v1": {'event':1.5e6, 'frame':1e9},
        "ps_202_fps_199_v2":{'event':1.5e6, 'frame':1e8},
    }

    iterator = zip(
                   [data_radius[condition]],
                   [data_fps[condition]],
                   [data_version[condition]]
                   )

    for i, (radius, fps, version) in tqdm(enumerate(iterator)):
        print(rank, size, radius, fps, version)
        thresh_key = f"ps_{radius}_fps_{fps}_v{version}"
        if mode == 'evanescent':
            event_thresh = 1e8
            frame_thresh = 1e8
        elif mode == "cytovia":
            event_thresh = cytovia_thresh_dict[thresh_key]['event']
            frame_thresh = cytovia_thresh_dict[thresh_key]['frame']
        else:
            assert False, f"unknown mode: {mode}"


        eb, fb, eb_file_name = fetch_data_sets(data_path, radius, fps, version, mode)
        print("dtype of frame = ",fb.dtype)
        height, width = eb.get_size()
        n_steps = 1000
        dt_blind = 5_000
        plotting = True
        sums = sum_event_based(eb, n_steps, dt_blind)

        # Edge Finding Kernel
        kernel = np.array([-1,0,1])

        conv_event = np.abs(convolve(sums[:,0], kernel, mode = 'same'))
        peaks_eb = find_peaks(sums[:,0], height = event_thresh, distance = 10)
        peaks_eb_edge = find_peaks(conv_event, height = event_thresh, distance = 10)
        time_eb = np.arange(0,n_steps*dt_blind,dt_blind)
        time_fb = np.arange(0,fb.shape[0]*(1/fps),1/fps)

        fb_shutter_edge = fb.sum(axis = (1,2)).astype(np.float32)
        conv_frame = np.abs(convolve(fb_shutter_edge, kernel, mode = 'same'))

        peaks_fb = find_peaks(conv_frame, height = frame_thresh)

        write_synchronized_image_sets(eb, fb, eb_file_name, peaks_eb_edge, 
                                      peaks_fb, radius, fps, version, mode)



        if plotting:
            make_synchronization_plot(time_eb,time_fb, peaks_eb_edge,
                              peaks_fb,dt_blind,radius,fps,version, mode)
