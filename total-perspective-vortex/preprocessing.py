import argparse
import mne
import matplotlib.pyplot as plt

MONTAGE = "Biosemi64"


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
    raw_data = raw.get_data()
    plt.plot(raw_data)
    plt.show()


def main(dataset):
    raw = mne.io.read_raw_edf(dataset, preload=True)
    montages = mne.channels.get_builtin_montages()

    data_montage = next((montage for montage in montages if MONTAGE in montage.title()), None)

    if data_montage is None:
        print("Could not find montage")
        exit(1)
    raw.set_montage(data_montage, on_missing='ignore')

    # raw_data = raw.get_data()
    # plt.jet()
    # plt.figure(figsize=(20, 10))
    # # plt.subplot(121)
    # # plt.plot(raw_data)
    # # plt.subplot(122)
    # # plt.jet()
    # # plt.imshow(raw_data[:, :])
    # plt.imshow(raw_data[:, 0:1000])
    #
    # plt.show()
    # intensity_per_channels(raw)
    plot_channels(raw)


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--dataset", "-d", help="Path to the dataset", default='../datasets/eggmmidb'
    )
    args = params.parse_args()
    main(args.dataset)
