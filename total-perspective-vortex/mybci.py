import argparse
import mne
import matplotlib.pyplot as plt
import os
import numpy as np

MONTAGE = "Biosemi64"


def parse_args(datasets, subject, experiment):
    # nb_dirs = [ if os.path.isdir(dir) for dir in os.listdir(datasets)]
    if os.path.exists(datasets) is False:
        raise Exception("Wrong dataset path")

    nb_dirs = 0
    for dir in os.listdir(datasets):
        extended_path = os.path.join(datasets, dir)
        if os.path.isdir(extended_path):
            nb_dirs += 1

    if subject <= 0 or subject > nb_dirs:
        raise Exception("Wrong subject provided.")
    subject = f'S{str(subject).zfill(3)}'

    nb_experiment = 0
    sub_path = os.path.join(datasets, subject)
    for file in os.listdir(sub_path):
        extended_path = os.path.join(sub_path, file)
        if os.path.isfile(extended_path):
            nb_experiment += 1
    nb_experiment /= 2
    experiment = str(experiment).zfill(2)

    experiment_path = os.path.join(sub_path, f'{subject}R{experiment}.edf')
    if os.path.exists(experiment_path) is False:
        raise Exception(f"Experiment {experiment_path} doesn't exist.")
    return experiment_path


def every_channel(raw):
    """
    Display in one window the overlapping of every channels

    Takes a raw (raw with montage set)
    :param raw:
    """
    raw.compute_psd().plot(picks="data", exclude="bads", average=True)
    plt.plot()


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
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.plot(raw_data)
    plt.show()


def main(dataset, subject, experiment, visual=False):
    path = parse_args(dataset, subject, experiment)
    raw = mne.io.read_raw_edf(path, preload=True)
    montages = mne.channels.get_builtin_montages()

    data_montage = next((montage for montage in montages if MONTAGE in montage.title()), None)

    if data_montage is None:
        print("Could not find montage")
        exit(1)
    raw.set_montage(data_montage, on_missing='ignore')

    if visual is True:
        plot_channels(raw)
        # intensity_per_channels(raw)
        # each_channel(raw)
        # every_channel(raw)


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--dataset", "-d", help="Path to the dataset", default='../datasets/eggmmidb'
    )
    params.add_argument(
        "--subject", "-s", help="Subject use to plots", default=1, type=int
    )
    params.add_argument(
        "--experiment", "-exp", help="Expemerient used to display", default=1, type=int
    )

    params.add_argument(
        '--visual', default=False, action=argparse.BooleanOptionalAction
    )
    args = params.parse_args()
    main(args.dataset, args.subject, args.experiment, args.visual)
