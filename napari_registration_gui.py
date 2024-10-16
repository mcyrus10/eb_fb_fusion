#!/home/mcd4/miniconda3/envs/openeb/bin/python
from PIL import Image 
from dask_image.imread import imread
from magicgui import magicgui
from napari.qt.threading import thread_worker
from tqdm import tqdm
from os import mkdir
from pathlib import Path
from scipy.optimize import least_squares
from tifffile import imwrite
import dask.array as da
import trackpy as tp
import napari
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import affine_transform, median_filter
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm


def cp_free_mem() -> None:
    """
    Frees cupy's memory pool...
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def residual(mat: list, input_pair: np.array, target_pair: np.array) -> np.array:
    """
    This function is used by scipy.optimize.least_squares to fit the affine
    transform matrix

    Note that the conventional x,y cartesian coordinates are switched when
    looking at plt.imshow so the y coordinate is horizontal and x coordinate is
    vertical...

    Parameters:
    -----------
        - mat: array-like; fitted parameters for scale y, horizontal shear, y
          translation, vertical shear, scale x, translation x
        - input_pair: array of event coordinates (y,x,1)
        - target_pair: array of frame coordinates (y,x,1) that map 1-1 to the
          event coordinates
    """
    scale_y = mat[0]
    horz_shear = mat[1]
    translation_y = mat[2]
    vert_shear = mat[3]
    scale_x = mat[4]
    translation_x = mat[5]
    tform_array = np.array([
        [scale_y, horz_shear, translation_y],
        [vert_shear, scale_x, translation_x],
        [0, 0, 1]])
    out = np.dot(tform_array, input_pair)
    diff = out - target_pair
    return np.abs(diff).flatten()


class spatio_temporal_registration_gui:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.title = "Event-Frame Registration GUI"
        self.pull_affine_matrix = None
        self.affine_matrix = None
        self.frame_0 = 0
        self.track_bool = False

        dock_widgets = {
                'Registration Ops': [self._load_data_(),
                              self._flip_ud_(),
                              self._flip_lr_(),
                              self._diff_frame_(),
                              self._reset_affine_(),
                              self._fit_affine_(),
                              self._apply_transform_(),
                              self._shift_event_(),
                              self._set_frame_after_shutter_(),
                              self._write_transforms_(),
                              ],
                'Tracking Ops':[
                            self._preview_track_centroids_(),
                            self._track_()
                              ]
                }
        tabs = []
        for j,(key,val) in enumerate(dock_widgets.items()):
            handle = self.viewer.window.add_dock_widget(val,
                                               name = key,
                                               add_vertical_stretch = False,
                                               area = 'right'
                                               )
            tabs.append(handle)
            if j > 0:
                self.viewer.window._qt_window.tabifyDockWidget(
                        tabs[0],
                        handle
                        )
        self.total_shift = 0

    def _fit_affine_(self):
        @magicgui(call_button="Fit/Refit Affine Params")
        def inner():
            handle = self.viewer.layers[-1]
            _points_type_ = napari.layers.points.points.Points
            assert isinstance(handle, _points_type_), "not a points layer"
            data_handle = handle.data
            event_pairs = data_handle[::2,1:]
            frame_pairs = data_handle[1::2,1:]
            assert event_pairs.shape == frame_pairs.shape, "shape mismatch between pairs"
            n_points = event_pairs.shape[0]
            event_pairs = np.hstack([event_pairs, np.ones(n_points)[:,None]]).T
            frame_pairs = np.hstack([frame_pairs, np.ones(n_points)[:,None]]).T
            x0 = [1,0,0,1,0,0]
            out = least_squares(residual, x0, args = (event_pairs, frame_pairs))
            composed_mat = np.array([
                [out.x[0], out.x[1], out.x[2]],
                [out.x[3], out.x[4], out.x[5]],
                [0,0,1]
                ])

            if self.affine_matrix is None and self.pull_affine_matrix is None:
                self.pull_affine_matrix = composed_mat
                self.affine_matrix = np.linalg.inv(composed_mat)
            else:
                print("refining affine matrix")
                self.affine_matrix = self.affine_matrix @ np.linalg.inv(composed_mat)
                self.pull_affine_matrix = np.linalg.inv(self.affine_matrix)

        return inner

    def _apply_transform_(self) -> None:
        """
        still a work in progress, this returns the widget that enables the
        transform operations
        """
        @magicgui(call_button = "Transform")
        def inner(batch_size: int = 50):
            event_handle = self.__fetch_layer__("event").data
            frame_handle = self.__fetch_layer__("frame").data
            if frame_handle.ndim == 4:
                nz, nx, ny, _ = frame_handle.shape
                frame_ndim = 4
            elif frame_handle.ndim == 3:
                nz, nx, ny = frame_handle.shape
                frame_ndim = 3

            _, nx_event, ny_event, _ = event_handle.shape

            data_type = event_handle.dtype

            tform_pull = cp.eye(4).astype(cp.float32)
            tform_pull[1:,1:] = cp.array(self.pull_affine_matrix, dtype = cp.float32)

            tform_push = cp.eye(4).astype(cp.float32)
            tform_push[1:,1:] = cp.array(self.affine_matrix)

            event_raw_transformed = np.zeros([nz,nx,ny,3], dtype = np.uint8)
            if frame_ndim == 3:
                frame_transformed = np.zeros([nz, nx_event, ny_event], dtype = np.float32)
            elif frame_ndim == 4:
                frame_transformed = np.zeros([nz, nx_event, ny_event, 3], dtype = np.uint8)

            n_batch = nz // batch_size
            remainder = nz % batch_size
            if remainder > 0:
                n_batch += 1
            cp_free_mem()
            print("potential speedup for 4D affine transform...")
            for q in tqdm(range(n_batch)):
                upper_lim = min((q+1)*batch_size, nz)
                local_batch_size = upper_lim - batch_size*q
                slice_ = slice(q*batch_size, upper_lim)
                for j in range(3):
                    event_raw_transformed[slice_,:,:,j] = affine_transform(
                        cp.array(event_handle[slice_,:,:,j], dtype = cp.float32),
                        tform_push,
                        output_shape = (local_batch_size,nx,ny),
                        order = 0
                        ).get()
                    cp_free_mem()
                    if frame_ndim == 4:
                        frame_transformed[slice_,:,:,j] = affine_transform(
                            cp.array(frame_handle[slice_,:,:,j], dtype = cp.float32),
                            tform_pull,
                            output_shape = (local_batch_size, nx_event, ny_event),
                            order = 3
                                    ).get()
                        cp_free_mem()
                if frame_ndim == 3:
                    frame_transformed[slice_] = affine_transform(
                            cp.array(frame_handle[slice_], dtype = cp.float32),
                            tform_pull,
                            output_shape = (local_batch_size, nx_event, ny_event),
                            order = 3
                            ).get()
                cp_free_mem()

            self.viewer.add_image(event_raw_transformed, 
                                  opacity = 0.4,
                                  name = "event -> frame")
            self.viewer.add_image(frame_transformed, 
                                  visible = False,
                                  name = "frame -> event",
                                  colormap = "hsv"
                                  )
            self.__fetch_layer__("event").visible = False
            self.__fetch_layer__("frame").opacity = 1.0
        return inner

    def _reset_affine_(self):
        @magicgui(call_button="reset affine")
        def inner():
            self.pull_affine_matrix = None
            self.affine_matrix = None
            print("Affine Transform set to None")
        return inner

    def _flip_ud_(self):
        @magicgui(call_button="flip event ud")
        def inner():
            handle = self.__fetch_layer__("event")
            handle.data = handle.data[:,::-1]
            print("Flipped Event UD")
        return inner

    def _flip_lr_(self):
        @magicgui(call_button="flip event lr")
        def inner():
            handle = self.__fetch_layer__("event")
            handle.data = handle.data[:,:,::-1]
            print("Flipped event LR")
        return inner

    def _load_data_(self):
        @magicgui(call_button="load data sets",
              main_window = True,
              persist = True,
              layout = 'vertical',
              frame_dir = {"label": "Select Frame File (.tif)"},
              event_dir = {"label": "Select Event File (.raw)"},
                  )
        def inner(
                frame_dir = Path.home(),
                event_dir = Path.home(),
                fps: int = 0
                ):
            self.frame_dir = frame_dir
            self.event_dir = event_dir
            self.delta_t = int(np.round(1e6/fps))
            print(f"delta_t = {self.delta_t}")
            frame_files = frame_dir.as_posix()
            event_files = event_dir.as_posix()
            self.__extract_folder_metadata__(frame_dir)
            print(event_files)
            print(frame_files)
            frame_stack = imread(frame_files)
            event_stack = self._load_raw_to_numpy_(str(event_dir), self.delta_t)
            inst.viewer.add_image(event_stack, colormap = 'gray', name = 'event')
            inst.viewer.add_image(frame_stack, colormap = 'hsv',
                                    name = 'frame', opacity = 0.4)
            self.affine_matrix = None
            self.pull_affine_matrix = None
        return inner

    def _load_raw_to_numpy_(self, event_file, delta_t):
        mv_iterator = EventsIterator(
                event_file,
                delta_t = delta_t,
                mode = "delta_t",
                start_ts = 0,
                )
        height,width = mv_iterator.get_size()
        image_buffer = np.zeros([height, width, 3], dtype = np.uint8)
        print(delta_t, height, width, image_buffer.shape)
        images = []
        for q, ev in tqdm(enumerate(mv_iterator), desc = "raw --> numpy"):
            BaseFrameGenerationAlgorithm.generate_frame(ev, image_buffer)
            images.append(image_buffer.copy())
        return np.stack(images)

    def _diff_frame_(self):
        @magicgui(call_button="Diff Frame")
        def inner(median_kernel: int = 3):
            frame_handle = self.__fetch_layer__("frame").data.copy()
            cp_free_mem()
            temp = cp.diff(cp.array(frame_handle, dtype = cp.float32), axis = 0)
            temp = median_filter(temp, (1,median_kernel, median_kernel)).get()
            cp_free_mem()
            inst.viewer.add_image(temp, name = "diff frame", colormap = "cividis")
        return inner

    def _set_frame_after_shutter_(self):
        """
        This is a slightly subjective point, but is just for finding the point
        where the decay is sufficiently significant that the tracking can work
        properly...
        """
        @magicgui(call_button="Set Frame After Shutter",
                frame_0={'label':'Frame after Shutter','min':0,'max':1e16})
        def inner(frame_0: int = 0):
            self.frame_0 = frame_0
            print(f"first frame set to {self.frame_0}")
        return inner

    def _shift_event_(self):
        @magicgui(call_button="Shift Event Temporal",
                shift={'label':'shift','min':-10_000,'max':10_000})
        def inner(shift: int = 0):
            transformed = self.__fetch_layer__("event")
            transformed.data = np.roll(transformed.data,
                                       shift = shift, 
                                       axis = 0)
            self.total_shift += shift
        return inner

    def _write_transforms_(self):
        @magicgui(call_button="Write Transformed Images and data")
        def inner():
            frame_transformed_handle = self.__fetch_layer__("frame -> event").data
            event_transformed_handle = self.__fetch_layer__("event -> frame").data
            frame_raw_handle = self.__fetch_layer__("frame").data
            event_raw_handle = self.__fetch_layer__("event").data
            out_dir = Path("./time_synced") / self.microscope/ self.dataset
            if not out_dir.is_dir():
                mkdir(out_dir)
            self.__write_transform_info__(out_dir)
            # Write The Transformed Image Stacks
            f_name_frame_tformed = out_dir / f"frame_transformed.tif"
            f_name_event_tformed = out_dir / f"event_transformed.tif"
            imwrite(f_name_frame_tformed, frame_transformed_handle[self.frame_0:])
            imwrite(f_name_event_tformed, event_transformed_handle[self.frame_0:])

            # Write The Untransformed Image Stacks at the synchronization time
            f_name_frame_raw = out_dir / f"frame_temporal_sync.tif"
            f_name_event_raw = out_dir / f"event_temporal_sync.tif"
            end_index = frame_transformed.shape[0]
            imwrite(f_name_frame_raw, frame_raw_handle[self.frame_0:])
            imwrite(f_name_event_raw, event_raw_handle[self.frame_0:end_index])
            print("Finished Writing Images and Transform Data")

        return inner

    def __extract_folder_metadata__(self, folder_name) -> None:
        """
        this method extracts some of the experiment meta data from the file
        name
        """
        components = folder_name.parts
        for elem in components:
            if "cytovia" in elem.lower():
                microscope = 'cytovia'
                break
            elif "evanescent" in elem.lower():
                microscope = 'evanescent'
                break
        else:
            #assert False, "No Microscope in File name"
            microscope = "unknown_instrument"
            data_set = "data_set_temp"

        data_set = str(folder_name.name).split(".tif")[0]
        fps = int(data_set.split("_")[3].split("fps")[0])
        print("fps = ",fps)


        self.microscope = microscope
        self.dataset = data_set
        self.fps = fps
        self.delta_t = int(round(1e6/fps))

    def __fetch_layer__(self, layer_name: str):
        """
        Helper function so that the layers can be re-ordered
        """
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        else:
            print(f"Layer {layer_name} not found")
            return 

    def __write_transform_info__(self, out_directory) -> None:
        """
        Writes the synchronization info to a csv file (tab delimited)
        """
        file_name_prefix = f"{self.microscope}_{self.dataset}"
        points = self.__fetch_layer__("Points").data
        frame_shape  = inst.__fetch_layer__("frame").data.shape
        affine_mat = self.affine_matrix
        horz_scale = affine_mat[0,0]
        horz_shear = affine_mat[0,1]
        horz_translation = affine_mat[0,2]
        vert_shear = affine_mat[1,0]
        vert_scale = affine_mat[1,1]
        vert_translation = affine_mat[1,2]
        # This is the alignment time could be + or -
        time_alignment = -1 * self.total_shift * self.delta_t
        time_0 = time_alignment + self.frame_0 * self.delta_t
        duration = (frame_shape[0] - self.frame_0) * self.delta_t
        time_end = time_0 + duration
        params = {
                  "path_to_raw":str(self.event_dir),
                  "path_to_frame":str(self.frame_dir),
                  "time_init":time_0/1e6,
                  "time_end":time_end/1e6,
                  "frame_0":self.frame_0,
                  "horz_scale":horz_scale,
                  "horz_shear":horz_shear,
                  "horz_translation":horz_translation,
                  "vert_shear":vert_scale,
                  "vert_scale":vert_shear,
                  "vert_translation":vert_translation
                  }

        with open(out_directory / f"{file_name_prefix}_sync.csv", 'w') as f:
            for param, val in params.items():
                f.write(f"{param}\t{val}\n")

    def _preview_track_centroids_(self):
        @magicgui(call_button="Locate Centroids",
                layer_name = {'label':'Layer Name'},
                minmass = {'label':'minmass', 'max': 1e16},
                diameter = {'label':'Diameter', 'max': 1e16},
                test_frame = {'label':'Test Frame', 'max': 1e16}
                )
        def inner(
                layer_name: str,
                minmass: float,
                diameter: int,
                test_frame: int
                ):
            track_handle = self.__fetch_layer__(layer_name).data[test_frame].copy()
            if track_handle.ndim == 3:
                print("--> not sure what to do for 4d images?")
                track_handle = np.sum(track_handle.astype(np.float32), axis = -1)
                pass
            elif track_handle.ndim == 2:
                pass
            self.track_dict = {
                    "minmass":minmass,
                    "diameter": diameter,
                    }
            self.track_bool = True
            print(type(track_handle))
            f = tp.locate(
                          np.array(track_handle),
                          diameter,
                          minmass = minmass,
                          invert = False)
            points_array = f[['y','x']].values
            self.viewer.add_points(
                    points_array,
                    name = f'tracked centroids {test_frame}',
                    symbol = 'x',
                    face_color = 'b'
                    )
        return inner

    def _track_(self):
        @magicgui(call_button="Track Particles",
                layer_name = {'label':'Layer Name'},
                min_length = {'label':'Filter Stub Length', 'max': 1e16},
                search_range = {'label':'Search Range', 'max': 1e16},
                )
        def inner(
                layer_name: str,
                min_length: int,
                search_range: int,
                ):
            assert self.track_bool, ("Set the tracking parameters with "
                                     "'preview' before tracking")
            minmass = self.track_dict['minmass']
            diameter = self.track_dict['diameter']
            track_handle = self.__fetch_layer__(layer_name).data.copy()
            if track_handle.ndim == 4:
                print("--> not sure what to do for 4d images?")
                track_handle = np.sum(track_handle.astype(np.float32), axis = -1)
                pass
            elif track_handle.ndim == 3:
                pass
            f = tp.batch(np.array(track_handle, dtype = np.float32), 
                         diameter = diameter,
                         minmass = minmass)
            t = tp.link(f, search_range = search_range)
            print(t.head)
            t1 = tp.filter_stubs(t, min_length)
            self.tracks = t1.copy()
            # Add Track Centroids to viewer as points layer
            slice_handle = ['frame','y','x']
            self.viewer.add_points(t1[slice_handle], 
                                   name = "tracked centroids",
                                   face_color = 'k')

            # Add Tracks to viewer as path (frame agnostic)
            tracks = []
            for particle in self.tracks['particle'].unique():
                slice_ = self.tracks['particle'] == particle
                tracks.append(self.tracks[['y','x']][slice_])
            self.viewer.add_shapes(tracks, name = "Tracks", shape_type = "path")

        return inner

    def _calc_msd_(self):
        pass


if __name__ == "__main__":
    inst = spatio_temporal_registration_gui()
    napari.run()
