import csv
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
from pathlib import Path
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm

if __name__ == "__main__":
    configs = sorted(list(Path("time_synced").glob("*/*/*.csv")))
    header = ['parameter','value']
    for elem in configs:
        parent = elem.parent
        with open(elem, mode = 'r') as f:
            data = dict(csv.reader(f, delimiter = "\t"))
        # WRITE THE FRAME SYNCED
        frame_0 = int(data['frame_0'])
        path_to_frame = data['path_to_frame']
        frame = imread(path_to_frame)[frame_0:]
        f_name_frame = parent / "frame_temporal_sync.tif"
        imwrite(f_name_frame, frame)


        # WRITE THE EVENT SYNCED
        event_file = data['path_to_raw']
        start_ts = np.round(float(data['time_init']) * 1e6)
        fps = int(str(parent).split("_")[-1].split("fps")[0])
        delta_t = int(round(1e6 / fps))
        mv_iterator = EventsIterator(
                event_file,
                delta_t = delta_t,
                mode = "delta_t",
                start_ts = start_ts,
                )
        height, width = mv_iterator.get_size()
        image_buffer = np.zeros([height, width, 3], dtype = np.uint8)
        images = []
        for q,ev in tqdm(enumerate(mv_iterator), desc = "raw -> numpy"):
            if q >= len(frame):
                continue
            BaseFrameGenerationAlgorithm.generate_frame(ev, image_buffer)
            images.append(image_buffer.copy())
        images = np.stack(images)
        f_name_event = parent / "event_temporal_sync.tif"
        imwrite(f_name_event, images)
        print(f"finished {elem}")
