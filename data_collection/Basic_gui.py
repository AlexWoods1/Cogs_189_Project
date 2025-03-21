import tkinter as tk
import random
import time
import pandas as pd
import datetime
import os
topics = ['Animals','Fungi/Plants','Cognitive Science','Planets and the universe', 'Things you learn in class','Colors','Clothes','Math', 'Your job', 'Research', 'Dinosaurs']

past = pd.DataFrame()

trial=0

def update_trial_info(event=None):
    global past, trial
    truth_value = random.choice(["fact","lie"])
    topic = random.choice(topics)
    time.sleep(3)
    label.config(text=f"Please tell A {truth_value} about {topic}.")
    current =pd.DataFrame(  {'time':time.time(), 'topic':topic, 'truth_value':truth_value}, index=[0])
    past = pd.concat([past,current])
    trial +=1
    if trial == 15:
        end_trial()

def end_trial(event=None):
    global past
    past.to_csv(f"trial_{datetime.date.today()}_{time.time()}.csv")
    quit()

# Create the main window
root = tk.Tk()
root.title("Random Number Generator")
root.geometry("300x200")
root.attributes('-fullscreen', True)

# Create a label to display the number
label = tk.Label(root, text="Press Space", font=("Arial", 24))
label.pack(expand=True)


# Pre Speaking
root.bind("<KeyRelease-space>", update_trial_info)
root.bind("<Return>",end_trial)
# Post Speaking



# Run the Tkinter event loop
root.mainloop()
