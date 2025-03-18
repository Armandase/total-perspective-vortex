import argparse
import mne
from mne.datasets import eegbci
from display import plot_channels, custom_psd_plot, intensity_per_channels, each_channel, plot_montage, plot_matrix
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedKFold
import numpy as np
from pipeline import create_pipeline
from mne import Epochs, pick_types
import joblib

MONTAGE = "Biosemi64"
SEED=42
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (7.0, 13.0)
BETA_BAND = (13.0, 30.0)
GAMMA_BAND = (30.0, 100.0)

def class_repartition(labels):
    class_repartition = np.bincount(labels)
    class_repartition = class_repartition / np.sum(class_repartition)
    return class_repartition


def train(data, pipeline_names, tmin=1.0, tmax=4.0, seed=None):
    data.filter(ALPHA_BAND[0], BETA_BAND[1], fir_design='firwin', skip_by_annotation='edge')
    pipe =  create_pipeline(pipeline_names)
    picks = pick_types(data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    events, events_id = mne.events_from_annotations(data, event_id={"T1": 0, "T2": 1})

    epochs = Epochs(raw=data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    epochs_data_train = epochs_train.get_data(copy=False)

    # K-fold but still keep the dataset balanced(repartition) between the classes regardless of the fold
    cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    
    labels = epochs.events[:, -1]

    print("Class repartition: ", class_repartition(labels))

    # compute time taken
    # scores = cross_val_score(pipe, epochs_data_train, labels, cv=cv, n_jobs=None)
    # print("Classification accuracy: %0.1f (+/- %0.1f)" % (100 * scores.mean(), 100 * scores.std()))

    avg_val_scores = []
    avg_scores = []
    for train_idx, test_idx in cv.split(epochs_data_train, labels):
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = epochs_data_train[train_idx]
        X_test = epochs_data_train[test_idx]
        pipe.fit(X_train, y_train)
        score = pipe.score(X_train, y_train)
        val_score = pipe.score(X_test, y_test)
        print("Score: ", score)
        avg_scores.append(score)
        avg_val_scores.append(val_score)
    print(f"Average score: {np.mean(avg_scores):.2f} std: {np.std(avg_scores):.2f}")
    print(f"Val average score: {np.mean(avg_val_scores):.2f} std: {np.std(avg_val_scores):.2f}")

    # save the pipeline
    joblib.dump(pipe, 'pipeline.joblib')

    
def plots(raw, montage):
    plot_matrix(raw)
    intensity_per_channels(raw)
    each_channel(raw)
    custom_psd_plot(raw)
    plot_montage(montage)

def set_montage(raw, montage_name=MONTAGE):
    montages = mne.channels.get_builtin_montages()
    data_montage = next((montage for montage in montages if montage_name in montage.title()), None)
    if data_montage is None:
        print("Could not find montage")
        exit(1)
    raw.set_montage(data_montage, on_missing='ignore')
    return raw, data_montage

def main(dataset, subject, runs, visual=False, full=False, pipeline=['wavelet', 'custom_CSP_Whitening', 'LDA_shrinkage']):
    if full is False:
        file_names = eegbci.load_data(subjects=subject, runs=runs, path=dataset)
    else:
        print("Loading all runs")
        subjects = range(1, 110)
        runs = range(1, 15)
        file_names = eegbci.load_data(subjects=subjects, runs=runs, path=dataset)

    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in file_names])
    raw, montage_name = set_montage(raw)

    if visual is True:
        montage = mne.channels.make_standard_montage(montage_name)
        plots(raw, montage)
        exit(0)
    train(raw, pipeline, tmin=0., tmax=4., seed=SEED)


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--dataset", "-d", help="Path to the dataset", default=None, type=str
    )
    params.add_argument(
        "--subject", "-s", help="Subject use to plots", default=1, type=int
    )
    params.add_argument(
        '-r','--runs', nargs='+', help='Expemerients used', required=True, type=int
    )
    params.add_argument(
        '--visual', default=False, action=argparse.BooleanOptionalAction
    )
    params.add_argument(
        '--full', default=False, action=argparse.BooleanOptionalAction
    )
    params.add_argument(
        '--pipeline', '-p', nargs='+', help='Pipeline elements, last element should be the estimator and the other are transformers', required=False, type=str, default=['wavelet', 'custom_CSP_Whitening', 'LDA_shrinkage']
    )
    args = params.parse_args()
    main(args.dataset, args.subject, args.runs, args.visual, args.full, args.pipeline)
