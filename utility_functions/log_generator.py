import os
import sys
import time
import streamlit as st


log_dir = os.path.join(os.getcwd(),'logs/')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

def trim_logs():
    logs = os.listdir(log_dir)
    logs_sorted = sorted(logs, key=lambda x :int(x))
    os.remove(os.path.join(log_dir, logs_sorted[0]))



@st.cache_resource
def start_log():
    time_stamp = str(time.time()).split('.')[0]
    saved_stdout = sys.stdout
    f = open(log_dir + time_stamp, "w", encoding='utf-8')
    sys.stdout = f

    if len(os.listdir(log_dir)) > 10:
        trim_logs()
    return saved_stdout


