from PIL import Image 
from cupyx.scipy.ndimage import affine_transform, median_filter
from dask_image.imread import imread
from magicgui import magicgui
from magicgui.tqdm import tqdm
from pathlib import Path
from scipy.optimize import least_squares
import cupy as cp
import dask.array as da
import napari
import numpy as np


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


class registration_gui:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.title = "Event-Frame Registration GUI"
        dock_widgets = {
                'Operations': [self._load_data_(),
                              self._flip_ud_(),
                              self._flip_lr_(),
                              self._diff_frame_(),
                              self._fit_affine_(),
                              self._apply_transform_(),
                              self._shift_event_()]
                        }
        for key,val in dock_widgets.items():
            self.viewer.window.add_dock_widget(val,
                                               name = key,
                                               add_vertical_stretch = False,
                                               area = 'right'
                                               )

    def _fit_affine_(self):
        @magicgui(call_button="Fit Affine Params")
        def inner():
            data_handle = self.viewer.layers[-1].data
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
            self.pull_affine_matrix = composed_mat
            self.affine_matrix = np.linalg.inv(composed_mat)
        return inner

    def _apply_transform_(self) -> None:
        """
        still a work in progress, this returns the widget that enables the
        transform operations
        """
        @magicgui(call_button = "Transform Event")
        def inner(batch_size: int = 50):
            event_handle = self.__fetch_layer__("event").data
            frame_handle = self.__fetch_layer__("frame").data
            nz, nx, ny = frame_handle.shape
            _, nx_event, ny_event, _ = event_handle.shape

            data_type = event_handle.dtype

            tform_pull = cp.eye(4).astype(cp.float32)
            tform_pull[1:,1:] = cp.array(self.pull_affine_matrix, dtype = cp.float32)

            tform_push = cp.eye(4).astype(cp.float32)
            tform_push[1:,1:] = cp.array(self.affine_matrix)

            event_raw_transformed = np.zeros([nz,nx,ny,3], dtype = np.uint8)
            frame_transformed = np.zeros([nz, nx_event, ny_event], dtype = np.float32)

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
                frame_transformed[slice_] = affine_transform(
                            cp.array(frame_handle[slice_], dtype = cp.float32),
                            tform_pull,
                            output_shape = (local_batch_size, nx_event, ny_event),
                            order = 3
                            ).get()
                cp_free_mem()

            self.viewer.add_image(event_raw_transformed, 
                                  visible = False,
                                  name = "event -> frame")
            self.viewer.add_image(frame_transformed, 
                                  opacity = 0.4,
                                  name = "frame -> event",
                                  colormap = "hsv"
                                  )
            self.__fetch_layer__("frame").visible = False
        return inner

    def _flip_ud_(self):
        @magicgui(call_button="flip event ud")
        def inner():
            for elem in self.viewer.layers:
                if elem.name == 'event':
                    elem.data = elem.data[:,::-1]
                if elem.name == 'event 3 chan':
                    elem.data = elem.data[:,::-1]
        return inner

    def _flip_lr_(self):
        @magicgui(call_button="flip event lr")
        def inner():
            for elem in self.viewer.layers:
                if elem.name == 'event':
                    elem.data = elem.data[:,:,::-1]
                if elem.name == 'event 3 chan':
                    elem.data = elem.data[:,:,::-1]
        return inner

    def _load_data_(self):
        @magicgui(call_button="load data sets",
              main_window = True,
              persist = True,
              layout = 'vertical',
              frame_dir = {"label": "Select Frame Based Directory", 'mode': 'd'},
              event_dir = {"label": "Select Event Based Directory", 'mode': 'd'},
                  )
        def inner(
                frame_dir = Path.home(),
                event_dir = Path.home(),
                ):
            frame_files = (frame_dir / "*.tif").as_posix()
            event_files = (event_dir / "*.tif").as_posix()
            print(event_files)
            print(frame_files)
            frame_stack = imread(frame_files)
            event_stack = imread(event_files)
            inst.viewer.add_image(event_stack, colormap = 'gray', name = 'event')
            inst.viewer.add_image(frame_stack, colormap = 'hsv', name = 'frame', opacity = 0.4)
        return inner

    def _diff_frame_(self):
        @magicgui(call_button="Diff Frame")
        def inner(median_kernel: int):
            frame_handle = self.__fetch_layer__("frame").data.copy()
            cp_free_mem()
            temp = cp.diff(cp.array(frame_handle, dtype = cp.float32), axis = 0)
            temp = median_filter(temp, (1,median_kernel, median_kernel)).get()
            cp_free_mem()
            inst.viewer.add_image(temp, name = "diff frame", colormap = "cividis")
        return inner

    def _shift_event_(self):
        @magicgui(call_button="Shift (Transformed) Events Temporal")
        def inner(shift: int):
            transformed = self.__fetch_layer__("transformed")
            transformed.data = np.roll(transformed.data,
                                       shift = shift, 
                                       axis = 0)
        return inner

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

    def _write_transformed_event_(self):
        @magicgui(call_button="Write Transformed Event")
        def inner(shift: int):
            transformed = self.__fetch_layer__("transformed")
            transformed.data = np.roll(transformed.data,
                                       shift = shift, 
                                       axis = 0)
        return inner



if __name__ == "__main__":
    inst = registration_gui()
    napari.run()
