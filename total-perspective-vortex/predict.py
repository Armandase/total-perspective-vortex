import argparse
import mne
from mne.datasets import eegbci
import numpy as np
import joblib
from mybci import set_montage, class_repartition, MONTAGE, ALPHA_BAND, BETA_BAND

MONTAGE = "Biosemi64"
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (7.0, 13.0)
BETA_BAND = (13.0, 30.0)
GAMMA_BAND = (30.0, 100.0)

def class_repartition(labels):
    class_repartition = np.bincount(labels)
    class_repartition = class_repartition / np.sum(class_repartition)
    return class_repartition

def predict(data, pipeline_path):
    pipe = joblib.load(pipeline_path)
    
    data.filter(ALPHA_BAND[0], BETA_BAND[1], fir_design='firwin', skip_by_annotation='edge')
    picks = mne.pick_types(data.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    events, events_id = mne.events_from_annotations(data, event_id={"T1": 0, "T2": 1})
    epochs = mne.Epochs(raw=data, events=events, event_id=events_id, tmin=1.0, tmax=4.0, proj=True, picks=picks, baseline=None, preload=True)
    epochs_data = epochs.get_data(copy=False)
    labels = epochs.events[:, -1]
    
    print('---------------------')
    print("Pipeline loaded: ", end="")
    for name in pipe.named_steps:
        print(name, end=" ")
    print()  
    
    print("Class repartition: ", class_repartition(labels))
    score = pipe.score(epochs_data, labels)
    print(f"Score: {score*100:.2f}%")
    

def set_montage(raw, montage_name=MONTAGE):
    montages = mne.channels.get_builtin_montages()
    data_montage = next((montage for montage in montages if montage_name in montage.title()), None)
    if data_montage is None:
        print("Could not find montage")
        exit(1)
    raw.set_montage(data_montage, on_missing='ignore')
    return raw, data_montage

def main(pipeline_path, dataset, subject, runs, full=False):
    if full is False:
        file_names = eegbci.load_data(subjects=subject, runs=runs, path=dataset)
    else:
        print("Loading all runs")
        subjects = range(1, 110)
        runs = range(1, 15)
        file_names = eegbci.load_data(subjects=subjects, runs=runs, path=dataset)

    list_raw = []
    for f in file_names:
        raw = mne.io.read_raw_edf(f, preload=True)
        # raw, _ = set_montage(raw)
        raw.resample(160)
        list_raw.append(raw)
    raw = mne.io.concatenate_raws(list_raw)
    raw, _ = set_montage(raw)

    predict(raw, pipeline_path)
    

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--pipeline", "-p", help="Pipeline path", type=str
    )
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
        '--full', default=False, action=argparse.BooleanOptionalAction
    )
    args = params.parse_args()
    main(args.pipeline, args.dataset, args.subject, args.runs, args.full)
