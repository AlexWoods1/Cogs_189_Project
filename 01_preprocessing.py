'''
Step 1.
Pre-process raw data.

Usage:
    01_preprocessing.py --eeg=FILE --photo=FILE --out=PATH

Options:
    -h --help     Show this screen and terminate script.
    --eeg=FILE    EEG signal: eeg_run-X.npy.
    --photo=FILE  Photo sensor signal: aux_run-X.npy.
    --out=PATH    Output file path.

'''

# imports
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt

def main():

    # get arguments
    vargs = docopt(__doc__)
    eeg_path = vargs['--eeg']
    photo_path = vargs['--photo']
    fname_out = vargs['--out']

    # global variables
    sampling_rate = 250 #Hz
    session = int(eeg_path.rstrip(".npy")[-1])

    # load data
    raw_eeg = np.load(eeg_path)
    aux = np.load(photo_path)

    # display total recording time
    recording_time = raw_eeg.shape[1] / sampling_rate
    minutes, seconds = divmod(recording_time, 60)
    print(f"Recording time: {int(minutes)} minutes and {int(seconds)} seconds")

    # binarize photo sensor data for behavioral labels
    photo = aux[1]
    mask = (photo >= 275)
    behavioral_state = mask.astype(float)

    all_channels = dict()
    for i in range(0,8):
        channel = i + 1

        # correct 0 value
        high_eeg = raw_eeg[i][1:]

        # perform discrete difference
        eeg = np.diff(high_eeg)

        #plot and save session-wide behaviorally labeled eeg
        time = np.linspace(0, len(eeg)/sampling_rate, len(eeg))

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(time, eeg, color='black', linewidth=1)
        for i in range(len(behavioral_state) - 1):
            if behavioral_state[i] == 0:
                ax.axvspan(time[i], time[i + 1], color='#ffe5ec', alpha=0.3)  # Light pink background if speaking

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(u"Microvolts (\u00b5"+"m)")
        plt.title(f"Session {session} EEG Signal")
        fig.savefig(f"figs/eeg_session{session}_channel{channel}.png")

        # save pre-processed eeg channel array
        all_channels[f'channel {channel}'] = eeg
   
    # save behavior labels
    np.save(fname_out + f"/behavior/behavior_{session}.npy", behavioral_state)

   # save the data to npz
    np.savez(fname_out + f"/eeg/eeg_{session}.npz", chan1=all_channels['channel 1'], chan2=all_channels['channel 2'], chan3=all_channels['channel 3'], chan4=all_channels['channel 4'], 
                    chan5=all_channels['channel 5'], chan6=all_channels['channel 6'], chan7=all_channels['channel 7'], chan8=all_channels['channel 8'])

if __name__ == "__main__":
    main()


