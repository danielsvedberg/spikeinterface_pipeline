import parse_nidq as pni

def get_stimulation_frames(base_folder, dinmap=None, ainmap=None, optoidx=None, overwrite=False):
    # snippet for testing
    #from pathlib import Path
    #base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
    #bs = pni.get_binary_signals(base_folder, overwrite=False)
    bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx, overwrite=overwrite, plot=False)

    bs_md = bs.metadata
    # get all rows of bs_md where column 'event' contains 'chrimson_on' or 'chr2_on'
    on_events = bs_md[bs_md['event'].str.contains('chrimson_on|chr2_on|solenoid_on')]
    off_events = bs_md[bs_md['event'].str.contains('chrimson_off|chr2_off')]
    # for each row in on_events, make a list of time times
    on_event_times = []
    labels = []
    for index, row in on_events.iterrows():
        ontimes = bs[index].t
        on_event_times.append(ontimes)
        labels.append(row['event'])

    off_event_times = []
    for index, row in off_events.iterrows():
        offtimes = bs[index].t
        off_event_times.append(offtimes)

    #flatten the lists into a single list
    on_event_times = [time for sublist in on_event_times for time in sublist]
    off_event_times = [time for sublist in off_event_times for time in sublist]

    #sort the lists
    on_event_times.sort()
    off_event_times.sort()

    return on_event_times, off_event_times



