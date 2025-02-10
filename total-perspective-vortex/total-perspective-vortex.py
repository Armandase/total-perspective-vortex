import argparse
import mne

def main(dataset):
    pass

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--dataset", "-d", help="Path to the dataset", default='../build/eggmmidb'
    )
    args = params.parse_args()
    main(args.dataset)
