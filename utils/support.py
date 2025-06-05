import inspect
from pathlib import Path
from scipy.io import loadmat
from functools import cache

support_dir = Path(inspect.getfile(inspect.currentframe())).parent / 'SupportData'

@cache
def get_rsvp_fix_stim():
    '''
    Load the RSVP fix stim mat file
    '''
    return loadmat(support_dir / 'rsvpFixStim.mat')

@cache
def get_face_library():
    '''
    Load the marmo face library
    '''
    return loadmat(support_dir / 'MarmosetFaceLibrary.mat')

def get_backimage_directory():
    '''
    Get the backimage directory
    '''
    return support_dir / 'Backgrounds'


