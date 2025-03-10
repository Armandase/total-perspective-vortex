import mne
import numpy as np
import matplotlib.pyplot as plt

def psd_plot(raw: mne.io.Raw):
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
    plt.imshow(raw_data[:, 0:1000])
    plt.show()


def plot_channels(raw):
    raw_data = np.array(raw.get_data())
    # raw_data = raw_data.mean(axis=-1)
    plt.xlabel("brain cell")
    plt.ylabel("cell")
    plt.plot(raw_data)
    plt.show()