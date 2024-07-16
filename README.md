# Event-Based - Frame-Based registration

This repo is for tools employed in the registration of the event-based data and
the frame-based.


## Installation

With the conda package manager installed, install the environment:

(without a CUDA device)

    $ conda env create --file im_proc_NO_GPU.yml

(With a CUDA device)

    $ conda env create --file im_proc.yml


If you do not have conda, you will need to install the dependencies manually 

## Running the GUI

To run the napari-based gui (activate the environment):

    $ conda activate im_proc
    $ python napari_registration_gui.py

## Registering Datasets

- To register two datasets use the widget on the right to specify the path to the
directory with the .tif  files for the frame and event data respectively.
- load the datasets (this will put them onto a single canvas with 1 coordinate system)
- adjust the contrast and opacity so that you can see both the event data and
  the frame data
- create a "points layer"
- alternate adding corresponding points between the Event Data then the Frame Data (in that order)!
- Fit the Affine Transform Parameters
- Transform the Dataset
