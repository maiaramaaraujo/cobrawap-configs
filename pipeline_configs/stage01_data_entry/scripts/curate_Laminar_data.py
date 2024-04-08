import numpy as np
import argparse
import neo
from pathlib import Path
import warnings
from snakemake.logging import logger
from neo import utils as neo_utils
import nixio
import h5py
import quantities as pq
import scipy.io as spio
from utils.neo_utils import add_empty_sites_to_analogsignal, time_slice, remove_annotations, time_slice
from utils.parse import parse_string2dict, none_or_float, none_or_str, none_or_int
from utils.io_utils import write_neo, load_neo


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=Path, required=True,
                     help="path to input data directory")
    CLI.add_argument("--output", nargs='?', type=Path, required=True,
                     help="path of output file")
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float,
                     default=None, help="sampling rate in Hz")
    CLI.add_argument("--spatial_scale", nargs='?', type=float, required=True,
                     help="distance between electrodes or pixels in mm")
    CLI.add_argument("--data_name", nargs='?', type=str, default='None',
                     help="chosen name of the dataset")
    CLI.add_argument("--annotations", nargs='+', type=none_or_str, default=None,
                     help="metadata of the dataset")
    CLI.add_argument("--array_annotations", nargs='?', type=none_or_str,
                     default=None, help="channel-wise metadata")
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str, default=None,
                     help="additional optional arguments")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=None,
                     help="start time in seconds")
    CLI.add_argument("--t_stop", nargs='?', type=none_or_float, default=None,
                     help="stop time in seconds")
    CLI.add_argument("--orientation_top", nargs='?', type=str, required=True,
                     help="upward orientation of the recorded cortical region")
    CLI.add_argument("--orientation_right", nargs='?', type=str, required=True,
                     help="right-facing orientation of the recorded cortical region")
    CLI.add_argument("--trial", nargs='?', type=none_or_int, default=None,
                     help="Trial number (optional), if None select all")
    args, unknown = CLI.parse_known_args()

    data_file = h5py.File(args.data, 'r')
    data = np.array(data_file['LFP'])
    del data_file
    dim_t, dim_channel = data.shape
        
    asig = neo.AnalogSignal(data,
                            units = pq.uV,
                            t_start = 0*pq.s,
                            sampling_rate = args.sampling_rate*pq.Hz, 
                            spatial_scale = args.spatial_scale*pq.mm,
                            orientation_top = args.orientation_top,
                            orientation_right = args.orientation_right,
                            trial = args.trial
                            )
    
    annotations = parse_string2dict(args.annotations)
    asig.annotations.update(annotations)

    kwargs = parse_string2dict(args.kwargs)

    # cut trial
    t_start, t_stop = kwargs['trial_times'][str(args.trial)] # simulation data
    asig = time_slice(asig, t_start = t_start*pq.s, 
                            t_stop = t_stop*pq.s)

    # remove channels outside of cortical column
    channels = np.delete(np.arange(dim_channel), kwargs['remove_channel_ids'])
    asig = asig[:,channels]
    dim_t, dim_channel = asig.shape

    # annotate channel coordinates
    asig.array_annotations.update(x_coords=np.zeros(dim_channel, dtype=int))
    asig.array_annotations.update(y_coords=np.arange(dim_channel, dtype=int))

    # asig = add_empty_sites_to_analogsignal(asig) 

    # Save data
    block = neo.Block(name=args.data_name)
    seg = neo.Segment('Segment 1')
    block.segments.append(seg)
    if asig.description is None:
        asig.description = ''
    block.segments[0].analogsignals = [asig]

    # Save data
    if len(block.segments[0].analogsignals) > 1:
        logger.warning('Additional AnalogSignal found. The pipeline can yet '
                       'only process single AnalogSignals.')

    with neo.NixIO(args.output) as io:
        io.write(block)