

#create a streamlit app to show how EMG can be analysed


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import glob
import plotly.io as pio
from scipy.signal import butter, filtfilt
import os
from PyQt5 import QtWidgets
import streamlit as st
import neurokit2 as nk
from datetime import datetime

st.title('Analyze electromyography (EMG) data')

"This program will allow the analysis of an EMG signal"