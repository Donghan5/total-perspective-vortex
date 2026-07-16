import argparse
import matplotlib.pyplot as plt

from src.preprocessing import filter_raw_eeg, load_raw_eeg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw and filtered EEG signals."
    )
    parser.add_argument(
        "subject_id",
        type=int,
        help="Subject ID (1 to 109)"
    )
    parser.add_argument(
        "run_id",
        type=int,
        help="Run ID (1 to 14)"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Load raw EEG data
    raw = load_raw_eeg(args.subject_id, args.run_id)

    # Filter raw EEG data
    raw_filtered = filter_raw_eeg(raw)

    print(f"Subject {args.subject_id}")
    print(f"Run {args.run_id}: Visualizing raw and filtered EEG signals...")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"Filtering: 8-30 Hz")

    raw.plot(
        title=f"Raw EEG - S{args.subject_id:03d}R{args.run_id:02d}",
        duration=10,
        n_channels=20,
        scalings='auto',
        block=False
    )

    raw_filtered.plot(
        title=f"Filtered EEG (8-30 Hz) - S{args.subject_id:03d}R{args.run_id:02d}",
        duration=10,
        n_channels=20,
        scalings='auto',
        block=True
    )

    raw_filtered.compute_psd(fmin=8, fmax=30).plot(
        average=True,
        picks="data",
        show=False,
    )

    plt.suptitle(
        f"Filtered EEG PSD 8-30 Hz - S{args.subject_id:03d}R{args.run_id:02d}"
    )

    plt.show()

if __name__ == "__main__":
    main()