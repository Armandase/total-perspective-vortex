import mne
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def custom_psd_plot(raw: mne.io.Raw):
    raw.compute_psd().plot(picks="data", exclude="bads", average=True, spatial_colors=True)
    plt.show()

def each_channel(raw):
    """
    Display one window with each channel from the raw edf file

    Takes a raw (return of mne.io.read_raw_ed)
    :param raw:
    """
    if isinstance(raw, np.ndarray):
        raw = raw[0]
        info = mne.create_info(ch_names=[f"ch{i}" for i in range(raw.shape[0])], sfreq=160, ch_types="eeg")
        raw = mne.io.RawArray(raw, info)
    raw.plot(n_channels=64, scalings="auto", title="Data from arrays", show=True, block=True)


def intensity_per_channels(raw):
    """
    Display the intensity of channels in the raw edf file.
    (red: intense activity, blue: low activity)
    :param raw:
    :return:
    """
    raw_data = raw.get_data()
    plt.jet()
    plt.imshow(raw_data)
    plt.show()


def plot_channels(raw):
    raw_data = np.array(raw.get_data())
    plt.xlabel("brain cell")
    plt.ylabel("cell")
    plt.plot(raw_data)
    plt.show()
    
def plot_montage(montage):
    montage.plot()
    montage.plot(kind='3d')
    plt.show()
    
def plot_matrix(raw):
    # (64, 161)
    df_raw = raw.to_data_frame()
    df_raw = df_raw.drop(columns=['time'])
    corr = df_raw.corr()
    # import pandas as pd
    # pd.plotting.scatter_matrix(df_raw, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
    sns.heatmap(corr, )
    plt.show()