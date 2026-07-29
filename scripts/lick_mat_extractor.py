import numpy as np
import scipy.io

#if on mac, omit r and use forward slashes instead
filepath = r"C:\Users\assad\Documents\recording_files\DS15\DS15_20260630\HSOMBS_DS15_20260630.mat"

mat = scipy.io.loadmat(filepath, squeeze_me=True)

event_markers = mat['None'].MatlabOpaque
print(event_markers)