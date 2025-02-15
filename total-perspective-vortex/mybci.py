import argparse
import mne
import os
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import numpy as np

MONTAGE = "Biosemi64"



def psd_plot(raw):
    raw.compute_psd().plot(picks="data", exclude="bads", average=True)

    plt.show()


def each_channel(raw):
    """
    Display one window with each channel from the raw edf file

    Takes a raw (return of mne.io.read_raw_ed)
    :param raw:
    """
    raw.plot()
    plt.show()


def intensity_per_channels(raw):
    """
    Display the intensity of channels in the raw edf file.
    (red: intense activity, blue: low activity)
    :param raw:
    :return:
    """
    raw_data = raw.get_data()
    plt.jet()
    plt.figure(figsize=(20, 10))
    plt.imshow(raw_data[:, 0:1000])

    plt.show()


def plot_channels(raw):
    raw_data = np.array(raw.get_data())
    # raw_data = raw_data.mean(axis=-1)
    plt.xlabel("brain cell")
    plt.ylabel("cell")
    plt.plot(raw_data)
    plt.show()


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
        # plot_channels(raw)
        # intensity_per_channels(raw)
        # each_channel(raw)
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
