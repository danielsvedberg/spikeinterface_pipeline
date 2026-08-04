from pathlib import Path
import ast
import utils.file_structure as fs

default_dinmap = {
    'licks': 0,
    'timing_pulse': 1,
    'tone': 2,
}

default_ainmap = {
    'photometry': 0,
    'accelerometer_x': 1,
    'accelerometer_y': 2,
    'accelerometer_z': 3,
    'solenoid': 4,
    'orange': 5,
    'blue': 6
}

default_ints = [20, 40, 60, 80, 100]  # light intensities in microwatts
default_optoidx = {'orange': default_ints,
                   'blue': default_ints}


def get_maps_from_txt(base_folder):
    #get 'config.txt' from base_folder
    paths = fs.SessionPaths(base_folder)

    if paths.input_map is None:
        raise FileNotFoundError(f"No nidaq_map.txt file found in {base_folder}")

    vars_dict = {}
    with open(paths.input_map) as f:
        tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                name = node.targets[0].id
                value = ast.literal_eval(node.value)
                vars_dict[name] = value

    #a map file need not assign all three: whichever ones it leaves out come back as None, so a
    #caller can fall back to its own default for just the missing map rather than all of them
    dinmap = vars_dict.get("dinmap")
    ainmap = vars_dict.get("ainmap")
    optoidx = vars_dict.get("optoidx")

    return dinmap, ainmap, optoidx


def get_channel_maps(base_folder, dinmap=None, ainmap=None, optoidx=None, use_defaults=False):
    #if all of dinmap, ainmap, and optoidx are None, try to get them from nidaq_map.txt
    #dinmap should be a dictionary mapping digital channel ids to event names
    #ainmap should be a dictionary mapping analog channel ids to event names

    if (dinmap is not None) and (ainmap is not None) and (optoidx is not None):
        return dinmap, ainmap, optoidx
    elif (dinmap is None) and (ainmap is None) and (optoidx is None):
        #check that nidaq_map.txt exists in base_folder
        paths = fs.SessionPaths(base_folder)
        if paths.input_map is not None:
            return get_maps_from_txt(base_folder)
        elif use_defaults:
            print("Warning: No nidaq_map.txt file found in base folder, using default channel maps.")
            dinmap = default_dinmap
            ainmap = default_ainmap
            optoidx = default_optoidx
            return dinmap, ainmap, optoidx
        else:
            print("Warning: No nidaq_map.txt file found in base folder, channel maps not provided, and use_default channel maps disabled.")
            return None, None, None
    else:
        raise ValueError("Either provide all of dinmap, ainmap, and optoidx, or none to use nidaq_map.txt or defaults.")
