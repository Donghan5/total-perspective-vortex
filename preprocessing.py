import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne import events_from_annotations, Epochs, pick_types

def get_data(subject, runs):
    raw_fnames = eegbci.load_data(subject, runs, verbose=False)
    raw_list = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = concatenate_raws(raw_list)
    
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
    
    event_id = dict(T1=0, T2=1) 
    events, _ = events_from_annotations(raw, event_id=event_id, verbose=False)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    tmin, tmax = -1.0, 4.0 
    epochs = Epochs(
        raw, 
        events, 
        event_id, 
        tmin, tmax, 
        proj=True, 
        picks=picks, 
        baseline=None, 
        preload=True,
        verbose=False
    )
    

    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    
    X = epochs_train.get_data() 
    y = epochs_train.events[:, -1] 
    
    return X, y