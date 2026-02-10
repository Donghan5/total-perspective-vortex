import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne import Epochs, pick_types

subject = 1
runs = [4, 8, 12]
tmin, tmax = -1.0, 4.0

print("Loading data from PhysioNet...")
raw_fnames = eegbci.load_data(subject, runs)
raw_list = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raw_list)

print("Standardizing data...")
eegbci.standardize(raw)
montage = make_standard_montage('standard_1020')
raw.set_montage(montage)

print("Renaming annotations...")
raw.annotations.rename(dict(T1='hands', T2='feet'))

print("Setting EEG reference... Setting average reference")
raw.set_eeg_reference(projection=True)

print("Filtering data...")
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

print("Extracting epochs...")
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

epochs = Epochs(
    raw,
    event_id=["hands", "feet"],
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)

epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)

X = epochs_train.get_data()
y = epochs.events[:, -1] - 2

print("X shape:", X.shape)
print("y shape:", y.shape)
print("label values:", set(y))