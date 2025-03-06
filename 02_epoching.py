'''
Step 2.
Create time windos and epoch the session into trials. This script should be run sequentially after 01.

Usage:
    02_epoching.py --sess=FILE --beh=FILE --out=PATH

Options:
    -h --help        Show this screen and terminate script.
    --sess=FILE      EEG signal: eeg_X.npz.
    --beh=FILE       Behavioral labels: behavior_X.npy
    --out=PATH       Output file path.

'''

# imports
from docopt import docopt
import numpy as np

def main():
    # get arguments
    vargs = docopt(__doc__)
    session_path = vargs['--sess']
    behavior_path = vargs['--beh']
    fname_out = vargs['--out']

    # global variables
    DURATION = 3 #sec
    sampling_rate = 250 #Hz
    session = int(session_path.rstrip(".npz")[-1])

    # load data
    all_channels = np.load(session_path)
    behavioral_state = np.load(behavior_path)

    # loop through channels
    for i_chan in range(1,9):
        eeg = all_channels[f'chan{i_chan}']

        # behavior
        messy_index = np.where(np.diff(behavioral_state))[0]

        clean_index = np.array([])
        differences = np.diff(messy_index)
        for i in np.where(differences >= 50)[0]:
            clean_index = np.append(clean_index, messy_index[int(i)])

        if len(clean_index)%2 == 1:
            clean_index = clean_index[:-1]              # we should have an even number, 32 indicates 16 press on/off instances
        else:
            continue

        #behavioral_state = behavioral_state[1:-1]       # account for values removed during eeg processing
       
        # make trials per channel
        trials = []
        for i in range(0,len(clean_index),2):
            start_i = clean_index[i] - (DURATION*sampling_rate)
            stop_i = clean_index[i+1] + (DURATION*sampling_rate)
           
            trial = eeg[int(start_i):int(stop_i + 1)]
            trials.append(trial)

        # remove first window (press to initiate gui)
        trials.pop(0)
        np.savez(fname_out + f"/eeg_{session}_chan{i_chan}.npz", trials)


if __name__ == "__main__":
    main()