import time
from brainflow import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

# set up board parameters
params = BrainFlowInputParams()
board_id = BoardIds.CYTON_BOARD.value
params.serial_port = "COM6"

# create board object
board =  BoardShim(board_id, params)

try:
    #prepare the board
    board.prepare_session()

    # start streaming data
    board.start_stream()
    time.sleep(120)


