lsl_out = False
save_dir = f'data/misc/' # Directory to save data to
run = 3 # Run number, it is used as the random seed for the trial sequence generation
save_file_aux = save_dir + f'aux_run-{run}.npy'
save_file_eeg = save_dir + f'eeg_run-{run}.npy'

import glob, sys, time, serial, os
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from threading import Thread, Event
from queue import Queue
from psychopy.hardware import keyboard
import numpy as np
sampling_rate = 250
CYTON_BOARD_ID = 0 # 0 if no daisy 2 if use daisy board, 6 if using daisy+wifi shield
BAUD_RATE = 115200
ANALOGUE_MODE = '/2' # Reads from analog pins A5(D11), A6(D12) and if no 
                    # wifi shield is present, then A7(D13) as well.
def find_openbci_port():
    """Finds the port to which the Cyton Dongle is connected to."""
    # Find serial port names per OS
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Error finding ports on your operating system')
    openbci_port = ''
    for port in ports:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            line = ''
            time.sleep(2)
            if s.inWaiting():
                line = ''
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass
    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
        exit()
    else:
        return openbci_port
    
print(BoardShim.get_board_descr(CYTON_BOARD_ID))
params = BrainFlowInputParams()
if CYTON_BOARD_ID != 6:
    params.serial_port = find_openbci_port()
elif CYTON_BOARD_ID == 6:
    params.ip_port = 9000
board = BoardShim(CYTON_BOARD_ID, params)
board.prepare_session()
res_query = board.config_board('/0')
print(res_query)
res_query = board.config_board('//')
print(res_query)
res_query = board.config_board(ANALOGUE_MODE)
print(res_query)
board.start_stream(45000)
stop_event = Event()

def get_data(queue_in, lsl_out=False):
    while not stop_event.is_set():
        data_in = board.get_board_data()
        timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
        eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
        aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
        if len(timestamp_in) > 0:
            print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
            queue_in.put((eeg_in, aux_in, timestamp_in))
        time.sleep(0.1)

queue_in = Queue()
cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
cyton_thread.daemon = True
cyton_thread.start()

keyboard = keyboard.Keyboard()
eeg = np.zeros((8, 0))
aux = np.zeros((3, 0))
while not stop_event.is_set():
    time.sleep(0.1)
    keys = keyboard.getKeys()
    if 'escape' in keys:
        stop_event.set()
        break
    while not queue_in.empty():
        eeg_in, aux_in, timestamp_in = queue_in.get()
        print('queue-out: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
        eeg = np.hstack((eeg, eeg_in))
        aux = np.hstack((aux, aux_in))
        print('total: ', eeg.shape, aux.shape)

os.makedirs(save_dir, exist_ok=True)
np.save(save_file_aux, aux)
np.save(save_file_eeg, eeg)