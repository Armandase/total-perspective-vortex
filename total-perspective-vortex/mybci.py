import argparse
import mne
from mne.datasets import eegbci
from display import plot_channels, psd_plot, intensity_per_channels, each_channel
import sklearn
from sklearn.model_selection import cross_val_score
from pipeline import create_pipeline
from mne import Epochs, pick_types

MONTAGE = "Biosemi64"

def train(data):
    pipe =  create_pipeline()
    
    scores = cross_val_score(pipe, epochs_data_train, labels, cv=cv, n_jobs=None)

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
