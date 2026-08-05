from pathlib import Path
from preprocessing import extract_photometry as ep
from preprocessing import parse_nidq as pni

base_folder = Path(r"\\research.files.med.harvard.edu\neurobio\Assad Lab\Rachele\Photometry\recording_files\DS48\DS48_20260729")

bs = pni.get_binary_signals(base_folder)

dFF = ep.get_photometry(base_folder)