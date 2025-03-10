import argparse
import mne
from mne.datasets import eegbci
from display import plot_channels, psd_plot, intensity_per_channels, each_channel
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
from pipeline import create_pipeline
from mne import Epochs, pick_types

MONTAGE = "Biosemi64"

def train(data, tmin=-1.0, tmax=4.0, seed=None):
    pipe =  create_pipeline()
    picks = pick_types(data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    events, events_id = mne.events_from_annotations(data)

    epochs = Epochs(raw=data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    epochs_data_train = epochs_train.get_data(copy=False)
    cv = ShuffleSplit(10, test_size=0.2, random_state=seed)
    labels = epochs.events[:, -1]
    print("Labels: ", labels)

    # add the min value to prevent negative values (which is not allowed for bincount)
    min_value = np.abs(np.min(labels)) if np.min(labels) < 0 else -np.min(labels)
    class_repartition = np.bincount(labels + min_value)
    class_repartition = class_repartition / np.sum(class_repartition)
    scores = cross_val_score(pipe, epochs_data_train, labels, cv=cv, n_jobs=None)
    print("Class repartition: ", class_repartition)
    print(scores)
    print("Classification accuracy: %0.1f (+/- %0.1f)" % (100 * scores.mean(), 100 * scores.std()))

def main(dataset, subject, runs, visual=False):
    ret = eegbci.load_data(subject=subject, runs=runs, path=dataset) 
    if len(ret) > 1:
        raise Exception("More than one file found")
    ret = str(ret[-1])
    raw = mne.io.read_raw_edf(ret, preload=True)
    montages = mne.channels.get_builtin_montages()

    data_montage = next((montage for montage in montages if MONTAGE in montage.title()), None)

    if data_montage is None:
        print("Could not find montage")
        exit(1)
    raw.set_montage(data_montage, on_missing='ignore')

    if visual is True:
        plot_channels(raw)
        intensity_per_channels(raw)
        each_channel(raw)
        psd_plot(raw)
    train(raw)


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--dataset", "-d", help="Path to the dataset", default='../datasets/'
    )
    params.add_argument(
        "--subject", "-s", help="Subject use to plots", default=1, type=int
    )
    params.add_argument(
        "--runs", "-r", help="Expemerient used to display", default=1, type=int
    )

    params.add_argument(
        '--visual', default=False, action=argparse.BooleanOptionalAction
    )
    args = params.parse_args()
    main(args.dataset, args.subject, args.runs, args.visual)
