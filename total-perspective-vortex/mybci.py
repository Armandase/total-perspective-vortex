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
    data.filter(7.0, 30.0, fir_design='firwin', skip_by_annotation='edge')
    pipe =  create_pipeline()
    picks = pick_types(data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    events, events_id = mne.events_from_annotations(data, event_id={"T1": 0, "T2": 1})

    epochs = Epochs(raw=data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    epochs_data_train = epochs_train.get_data(copy=False)

    cv = ShuffleSplit(10, test_size=0.2, random_state=seed)
    labels = epochs.events[:, -1]

    class_repartition = np.bincount(labels)
    class_repartition = class_repartition / np.sum(class_repartition)

    scores = cross_val_score(pipe, epochs_data_train, labels, cv=cv, n_jobs=None)

    print("Class repartition: ", class_repartition)
    print("Classification accuracy: %0.1f (+/- %0.1f)" % (100 * scores.mean(), 100 * scores.std()))

    # for train_idx, test_idx in cv.split(epochs_data_train):
    #     y_train, y_test = labels[train_idx], labels[test_idx]

    #     X_train = epochs_data_train[train_idx]
    #     X_test = epochs_data_train[test_idx]
    #     pipe.fit(X_train, y_train)
    #     score = pipe.score(X_test, y_test)
    #     print("Score: ", score)

def plots(raw):
    # plot_channels(raw)
    intensity_per_channels(raw)
    each_channel(raw)
    psd_plot(raw)

def set_montage(raw, montage_name=MONTAGE):
    montages = mne.channels.get_builtin_montages()
    data_montage = next((montage for montage in montages if montage_name in montage.title()), None)
    if data_montage is None:
        print("Could not find montage")
        exit(1)
    raw.set_montage(data_montage, on_missing='ignore')
    return raw

def main(dataset, subject, runs, visual=False):
    file_names = eegbci.load_data(subject=subject, runs=runs, path=dataset) 
    if len(file_names) != len(runs):
        raise Exception(f"Fail to open {len(runs)} runs")

    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in file_names])
    raw = set_montage(raw)

    if visual is True:
        plots(raw)
    train(raw, seed=42)


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--dataset", "-d", help="Path to the dataset", default='../datasets/'
    )
    params.add_argument(
        "--subject", "-s", help="Subject use to plots", default=1, type=int
    )
    params.add_argument('-r','--runs', nargs='+', help='Expemerients used', required=True, type=int)
    params.add_argument(
        '--visual', default=False, action=argparse.BooleanOptionalAction
    )
    args = params.parse_args()
    main(args.dataset, args.subject, args.runs, args.visual)
