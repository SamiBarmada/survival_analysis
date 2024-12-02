import csv
import cv2
import os
import sys
import pickle
import operator
import numpy as np
import pandas as pd
import scipy.stats
from glob import glob
from collections import defaultdict, namedtuple
import functools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'fractionation'))
#import nuclear_fractionation
import nuclear_fractionation

from imgutils import tifffile

def merge_measurements_with_survival_data(exp_name, results_path):
    fname = exp_name + '_surv_data.csv'
    try: 
        surv_df = pd.read_csv(os.path.join(results_path, fname))

    except OSError:
        print('Could not find ' + fname + ' and so unable to merge with measurements CSV.')
        return

    meas_df = pd.read_csv(os.path.join(results_path, exp_name + '_measurements.csv'))
    try:
        key = 'well-id'
        merged_df = pd.merge(surv_df, meas_df, on=key)
        merged_df.to_csv(os.path.join(results_path, exp_name + '_measurements_with_surv_data.csv'), na_rep='NA', index=False)
    except KeyError:
        print("Unable to merge CSVs because one or both do not have the key 'well-id'.")

def _make_dataframe(to_be_measured, compounds_to_be_measured, total_timepoints):
    # Create a dict that is the union of both dicts.
    all_to_be_measured = {channel : ms | compounds_to_be_measured[channel] 
                          for channel, ms in to_be_measured.items()}
    columns = []
    for channel, measurements in all_to_be_measured.items():
        for measurement in measurements:
            columns += [channel + '-' + measurement + '-' + str(tp+1) for tp in range(total_timepoints)]

    return pd.DataFrame([], columns=columns)

def _get_roi_paths(workdir):
    paths = glob(workdir + '/results/rois/*.p')
    #Basename will be '[A-H][0-9+].p. First sort by row letter, then by column number.
    paths = sorted(paths, key=lambda x: os.path.basename(x)[0][0])
    paths = sorted(paths, key=lambda x: os.path.basename(x)[0][1:])
    if not paths:
        print('\nNo rois found in ' + os.path.join(workdir, 'results', 'rois') + '\n')
        sys.exit()
    return paths

def _empty_handler(func):
    '''Used by the measure functions. It intercepts an empty array, which will have size 
       0, and returns a suitable value instead. If the array is non-empty, the measurement
       function is called normally.'''
    def wrapper(*args, **kwargs):
        if args[0].size == 0: return np.NaN
        else: return func(*args, **kwargs)
    return wrapper

_eh = _empty_handler
def _intensity_measure(subimgs, submasks, func):
    return list(map(_eh(func), [img[mask != 0] for img, mask in zip(subimgs, submasks)]))

def _geometric_measure(contours, func):
    return list(map(_eh(func), contours))

def _mean(subimgs, submasks):
    return _intensity_measure(subimgs, submasks, np.mean)

def _std(subimgs, submasks):
    return _intensity_measure(subimgs, submasks, np.std)

def _median(subimgs, submasks):
    return _intensity_measure(subimgs, submasks, np.median)

def _skew(subimgs, submasks):
    return _intensity_measure(subimgs, submasks, scipy.stats.skew)

def _kurtosis(subimgs, submasks):
    return _intensity_measure(subimgs, submasks, scipy.stats.kurtosis)

def _gradient_mean(subimgs, submasks):
    f = lambda arr : np.mean(np.gradient(arr))
    return _intensity_measure(subimgs, submasks, f)

def _gradient_std(subimgs, submasks):
    f = lambda arr : np.std(np.gradient(arr))
    return _intensity_measure(subimgs, submasks, f)

_95th_percentile_helper = functools.partial(np.percentile, q=95)
def _95th_percentile(subimgs, submasks):
    return _intensity_measure(subimgs, submasks, _95th_percentile_helper)

def _area(contours):
    return _geometric_measure(contours, cv2.contourArea)

def _perimeter(contours):
    f = functools.partial(cv2.arcLength, closed=True)
    return _geometric_measure(contours, f)

def _centroid(contours):
    def compute_centroid(contour):
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return f'{str(cx)}; {str(cy)}'
    return _geometric_measure(contours, compute_centroid)

def _aspect_ratio_helper(contour):
    _, (w, h), _ = cv2.minAreaRect(contour)
    return w / h

def _aspect_ratio(contours):
    return _geometric_measure(contours, _aspect_ratio_helper)

def _load_nuclear_rois(workdir, well):
    path = os.path.join(workdir, 'results', 'nuclear_fractionation', 'nuclear_ROIs', 'rois', well +'.p')
    rois = pickle.load(open(path, 'rb'))
    nuclear_rois = {ID : roi['nuclear'] for ID, roi in rois.items()}
    unknown_rois = {ID : roi['unknown'] for ID, roi in rois.items()}
    return nuclear_rois, unknown_rois 

# Set of all geometric measurements. 
geometric = set('area aspect_ratio n_area perimeter n_perimeter centroid'.split())

def _take_measurements(to_be_measured, subimgs, submasks, contours):
    measurements = {}
    for m in to_be_measured:
        if m in geometric:
            measurements[m] = measurement_map[m](contours)
        else:
            measurements[m] = measurement_map[m](subimgs, submasks)
    return measurements

def rois_to_bound_rects_and_submasks(rois):
    bound_rects = defaultdict(list)
    submasks = defaultdict(list)
    for ID, roi in rois.items():
        # TMP
        if type(roi) is list: contours = roi
        else: contours = roi['contours']

        if type(contours) is dict:
            tps_with_contours = sorted([(tp, c) for tp, c in contours.items()], key=lambda x: x[0])
            contours = list(zip(*tps_with_contours))[1]

        for c in contours:
            # If in the middle of a contour time series a contour was not found, then an empty numpy
            # array placeholder exists. Its size will be 0. Continue to store an empty array to keep 
            # track of this.
            if c.size == 0:
                bound_rects[ID].append((0, 0, 0, 0))
                submasks[ID].append(np.empty(0))

            else:
                x, y, w, h = cv2.boundingRect(c)
                submask = np.zeros((h, w))
                cv2.drawContours(submask, [c - np.array([[x, y]])], 0, (255, 255, 255), -1)
                bound_rects[ID].append((x, y, w, h))
                submasks[ID].append(submask)

    return bound_rects, submasks

# Iterate at the stack level because want to load a single stack ONCE and keep that in memory. 
def load_stack(workdir, well, channel):
    # Load a stack to begin measurements, if possible.
    try: 
        path = os.path.join(workdir, 'processed_imgs', 'stacked', channel, well + '.tif')
        stack = tifffile.imread(path)

        # If loaded stack is 2D because it's a single image, reshape it to be a 3D array.
        if len(stack.shape) == 2:
            stack = stack.reshape(1, *stack.shape)

        return stack

    except FileNotFoundError:
        print('\nStack at ' + path + ' not found. No measurements computed for ' + channel + ' in well ' + well + '.\n')
        return None

# Can create two distinct measurement maps. But that also means that would need two compoud measurement maps. Possibly more.
measurement_map = {'mean'                    : _mean,
                   'std'                     : _std,
                   'median'                  : _median,
                   '95th_percentile'         : _95th_percentile,
                   'area'                    : _area,
                   'perimeter'               : _perimeter,
                   'centroid'                : _centroid,
                   'skew'                    : _skew,
                   'kurtosis'                : _kurtosis,
                   'gradient_mean'           : _gradient_mean,
                   'gradient_std'            : _gradient_std,
                   'aspect_ratio'            : _aspect_ratio,
                   'c_mean'                  : _mean,
                   'c_std'                   : _std,
                   'c_median'                : _median,
                   'c_95th_percentile'       : _95th_percentile,
                   'c_area'                  : _area,
                   'c_skew'                  : _skew,
                   'c_kurtosis'              : _kurtosis,
                   'c_gradient_mean'         : _gradient_mean,
                   'c_gradient_std'          : _gradient_std,
                   'n_mean'                  : _mean,
                   'n_std'                   : _std,
                   'n_median'                : _median,
                   'n_95th_percentile'       : _95th_percentile,
                   'n_area'                  : _area,
                   'n_perimeter'             : _perimeter,
                   'n_skew'                  : _skew,
                   'n_kurtosis'              : _kurtosis,
                   'n_gradient_mean'         : _gradient_mean,
                   'n_gradient_std'          : _gradient_std}

CompoundMeasurement = namedtuple('Compound', ['atomic_measurement_measurements', 'measure_func'])
CM = CompoundMeasurement

def circ(areas, perimeters):
    perimeters[perimeters == 0] = np.NaN
    return 4 * np.pi * areas / perimeters ** 2
#circ = lambda x, y : 4 * np.pi * x / y ** 2
compound_measurement_map = {'CV'            : CM(['std', 'mean'], operator.truediv),
                            'c_CV'          : CM(['c_std', 'c_mean'], operator.truediv),
                            'n_CV'          : CM(['n_std', 'n_mean'], operator.truediv),
                            'gradient_CV'   : CM(['gradient_mean', 'gradient_std'], operator.truediv),
                            'c_gradient_CV' : CM(['c_gradient_mean', 'c_gradient_std'], operator.truediv),
                            'n_gradient_CV' : CM(['n_gradient_mean', 'n_gradient_std'], operator.truediv),
                            'n_c_mean'      : CM(['n_mean', 'c_mean'], operator.truediv),
                            'c_area'        : CM(['area', 'n_area'], operator.sub),
                            'circularity'   : CM(['area', 'perimeter'], circ),
                            'n_circularity' : CM(['n_area', 'n_perimeter'], circ)}
                               
def setup_compound_measurements(to_be_measured):
    compounds_to_be_measured = defaultdict(set)
    # Check each channel for any compound measurements.
    for channel, measurements in to_be_measured.items():
        # Ensure iteration on a copy of the set as may modify the set during iteration.
        for m in measurements.copy():
            # If the measurement is a compound one.
            if m in compound_measurement_map:
                cm = compound_measurement_map[m]
                # Remove that measurement.
                measurements.remove(m)
                # Add in its requisite atomic measurements.
                for measurement in cm.atomic_measurement_measurements:
                    measurements.add(measurement)
                # Store the compound measurement for later use.
                compounds_to_be_measured[channel].add(m)
    
    return compounds_to_be_measured

def measure_rois(workdir, config, to_be_measured):
    '''measure is a dict (map) from channels to list of measurements'''
    exp_name = config['experiment']['name']
    primary_channel = config['experiment']['imaging']['primary_channel']

    total_timepoints = config['experiment']['time_data']['timepoint_num']

    results_path = os.path.join(workdir, 'results')
    outpath = os.path.join(results_path, exp_name + '_measurements.csv')

    compounds_to_be_measured = setup_compound_measurements(to_be_measured)

    df = _make_dataframe(to_be_measured, compounds_to_be_measured, total_timepoints)

    cell_to_be_measured, n_to_be_measured, c_to_be_measured = {}, {}, {}
    for channel, measurements in to_be_measured.items():
        cell_to_be_measured[channel] = {m for m in measurements if not (m.startswith('n_') or m.startswith('c_'))}
        n_to_be_measured[channel] = [m for m in measurements if m.startswith('n_')]
        c_to_be_measured[channel] = [m for m in measurements if m.startswith('c_')]

    roi_paths = _get_roi_paths(workdir)

    for roi_path in roi_paths:
        well = os.path.basename(roi_path).split('.p')[0]
        print(exp_name + ':: Measuring ROIs in well: ' + well)

        with open(roi_path, 'rb') as roi_file:
            rois = pickle.load(roi_file)

        # Sets the full well-id row to NA. Sort the ROIs first so the order of rows are sorted.
        for ID in sorted(rois):
            # Increase ID by 1 for humans.
            df.loc[well + '-' + str(ID+1)] = np.NaN

        bound_rects, submasks = rois_to_bound_rects_and_submasks(rois)

        # If any nuclear or cytoplasmic measurements are to be taken.
        if any(n_to_be_measured.values()) or any(c_to_be_measured.values()):
            try:
                nuclear_rois, unknown_rois = _load_nuclear_rois(workdir, well)

            except FileNotFoundError:
                # It would be useful if NF returned rois here.
                nuclear_fractionation.run(workdir, config, well)

                nuclear_rois_loaded = False
                while not nuclear_rois_loaded:
                    try: 
                        nuclear_rois, unknown_rois = _load_nuclear_rois(workdir, well)
                        nuclear_rois_loaded = True
                    except:
                        nuclear_rois_loaded = False

            n_bound_rects, n_submasks = rois_to_bound_rects_and_submasks(nuclear_rois)

            # Build cytoplasmic submasks if cytoplasmic measurements are to be taken.
            u_bound_rects, u_submasks = rois_to_bound_rects_and_submasks(unknown_rois)
            if any(c_to_be_measured.values()):
                # Find the difference between the upper-left corner of the unknown (nuclear or cytoplasmic) 
                # and cellular bounding rectangles.
                delta = lambda br, u_br: (u_br[0] - br[0], u_br[1] - br[1])

                c_submasks = defaultdict(list)
                # At this point, the nuclear 'roi' is just a list of contours.
                for ID, u_roi in unknown_rois.items():
                    for ix, submask in enumerate(submasks[ID]):
                        # If the middle of a time series is missing a contour, then it is also missing
                        # a submask. The submask size will be 0 in this case.
                        if submask.size == 0:
                            c_submasks[ID].append(np.empty(0))

                        # Each cell submask is used to create a cyto submask by setting the contents of 
                        # the corresponding nuclear submask to zero. The nuclear submask is first shifted.
                        else:
                            c_submask = np.copy(submask)
                            u_submask = u_submasks[ID][ix]
                            if u_submask.size == 0: 
                                c_submasks[ID].append(np.empty(0))

                            else:
                                dx, dy = delta(bound_rects[ID][ix], u_bound_rects[ID][ix])
                                c_submask[dy:dy+u_submask.shape[0], dx:dx+u_submask.shape[1]][u_submask != 0] = 0
                                c_submasks[ID].append(c_submask)

        for channel in to_be_measured:
            stack = load_stack(workdir, well, channel)

            if stack is None: continue
            
            for ID, roi in rois.items():
                contours = roi['contours']

                # Some older versions of ROIs may contain contours as a dict instead of list. Convert here.
                if type(contours) is dict:
                    tps_with_contours = sorted([(tp, c) for tp, c in contours.items()], key=lambda x: x[0])
                    contours = list(zip(*tps_with_contours))[1]

                well_id = well + '-' + str(ID+1)

                # Create subimgs. Iterate through length of contour list. This length varies throughout 
                # the dataset. If the stack length is N, the contour list may be less than N, as not each 
                # image may have a contour.
                subimgs = []
                for ix, contour in enumerate(contours):
                    if type(contour) is int:
                        continue
                    # If the length of the current stack is less than or equal to the index, break. The stack
                    # for this channel does not have as many timepoints as contours.
                    if ix >= len(stack): break
                    # If in the middle of a contour time series a contour was not found, then an empty numpy
                    # array placeholder exists. Its size will be 0. Continue to store an empty array to keep 
                    # track of this. 
                    elif contour.size == 0:
                        subimgs.append(np.empty(0))

                    else:
                        x, y, w, h = bound_rects[ID][ix]
                        subimgs.append(stack[ix][y:y+h, x:x+w])

                measured = _take_measurements(cell_to_be_measured[channel], subimgs, submasks[ID], contours)

                if any(n_to_be_measured.values()) or any(c_to_be_measured.values()):

                    # It would be best if NF returned 'NONE' for those IDs which did not exist.
                    try: nuclear_roi = nuclear_rois[ID]
                    except KeyError: nuclear_roi = None

                    if nuclear_roi is not None:
                        n_subimgs = []
                        for ix in range(len(nuclear_roi)):
                            # can potentially turn offets into slices.
                            x, y, w, h = n_bound_rects[ID][ix]
                            n_subimgs.append(stack[ix][y:y+h, x:x+w])

                        if any(n_to_be_measured.values()):
                            measured.update(_take_measurements(n_to_be_measured[channel], n_subimgs, n_submasks[ID], nuclear_roi))

                        if any(c_to_be_measured.values()):
                            measured.update(_take_measurements(c_to_be_measured[channel], subimgs, c_submasks[ID], None))

                for measurement, values in measured.items():
                    try:
                        if len(values) == 1: df.loc[well_id][f'{channel}-{measurement}-1'] = values[0]
                        else: df.loc[well_id][f'{channel}-{measurement}-1':f'{channel}-{measurement}-{len(values)}'] = values
                    except KeyError:
                        print('\n\nEnsure that the amount of timepoints in your Mfile match the stack timepoints.\n\n')
                        raise

        df.to_csv(outpath, index_label='well-id', na_rep='NA')

    tps = total_timepoints
    mk_cols = lambda prefix: [prefix + str(i) for i in range(1, tps+1)]
    for channel, compound_measure_keys in compounds_to_be_measured.items():
        mk_prefix = lambda l: channel + '-' + l + '-'
        for measurement in compound_measure_keys:
            cm = compound_measurement_map[measurement]

            cm_prefix = mk_prefix(measurement)

            atomic_prefixes = [mk_prefix(m) for m in cm.atomic_measurement_measurements]

            df[mk_cols(cm_prefix)] = cm.measure_func(*[df[mk_cols(p)].values for p in atomic_prefixes])

    df.to_csv(outpath, index_label='well-id', na_rep='NA')
        
    print(exp_name + ' measurements are complete.')
    merge_measurements_with_survival_data(exp_name, results_path)
