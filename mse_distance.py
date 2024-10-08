import numpy as np
import argparse
from scipy.io import wavfile
import pandas as pd
from scipy.spatial.distance import euclidean
from numpy.fft import fft
from numpy.linalg import norm
import pdb 
from scipy.signal import resample
import librosa
import matplotlib.pyplot as plt
import os 
import mir_eval
import csv
from sklearn.metrics import f1_score
# from utils import read_data

def my_f1_score(y_true, y_pred, ths=2):
    # Initialize counters for true positives, false positives, and false negatives
    TP = 0
    FP = 0
    FN = 0
    
    # Iterate over the true and predicted arrays
    for i in range(len(y_true)):
        true = y_true[i]
        pred = y_pred[i]
        if true == 1 and np.sum(y_pred[i-ths: i+ths]) >= 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and np.sum(y_pred[i-ths: i+ths]) == 0:
            FN += 1
    
    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision == 0 or recall == 0:
        return 0
    
    return 2 * precision * recall / (precision + recall)

def load_data(file_path):
    sr = None
    if file_path.endswith('.npy'):
        array = np.load(file_path, allow_pickle=True)
        if len(array.shape) == 0:
            array = np.array([array])
        return sr, array
    elif file_path.endswith('.wav'):
        # For wav files, we ignore the sample rate
        data, sr = librosa.load(file_path, sr=None)
        return sr, data
    else:
        raise ValueError("Unsupported file type. Please provide .npy or .wav file.")


def compute_mse_from_target(args, input_array):

    if 'synthesis' in args.target_file:
        # do F1 score instead
        array1 = np.int64(input_array)
        array2 = np.load(args.target_file, allow_pickle=True)

        max_n = max(array1.max(), array2.max())
        _array_1, _array_2 = np.zeros(max_n), np.zeros(max_n)
        
        for n in array1:
            _array_1[n-1] = 1
        for n in array2:
            _array_2[n-1] = 1
        return my_f1_score(_array_2, _array_1)

    if args.file in ('ppg', 'general'):
        # array2_max = max(np.load(args.input_file[0], allow_pickle=True))
        array2_max = max(load_data(args.input_file[0])[1])
        # convert back to the original scale
        input_array = input_array / 100 * array2_max
    elif args.file == 'ecg_data' and 'heartrate' not in args.target_file:
        input_array = input_array / 1000

    if len(input_array) == 0:
        return np.nan

    array1 = input_array
    _, array2 = load_data(args.target_file)
    
    if len(array1) > len(array2):
        array1 = array1[:len(array2)]
    elif len(array2) > len(array1):
        array1 = np.concatenate([array1, np.zeros(len(array2) - len(array1))])

    if 'speech' in args.input_file[0]:
        mse, _, _, _ = mir_eval.separation.bss_eval_sources(
            array2, array1
        )
        mse = mse[0]
    else:
        mse = ((array1 - array2)**2).mean()
    return mse

def compute_mse_from_str(type, string1, string2):
    array1 = string1.split(', ')
    array1 = np.array([float(a) for a in array1 if a.isnumeric()])

    array2 = string2.split(', ')
    array2 = np.array([float(a) for a in array2 if a.isnumeric()])

    if len(array1) > len(array2):
        array1 = array1[:len(array2)]
    elif len(array2) > len(array1):
        array1 = np.concatenate([array1, np.zeros(len(array2) - len(array1))])

    if type == 'ecg':
        array1 /= 1000
        array2 /= 1000 
        return ((array1 - array2)**2).mean()
    else:
        raise NotImplementedError("Other modalities haven't been implemeted")

def compute_mse(file_path1, file_path2, args=None):
    # file_path1: calculated signal
    # file_path2: GT signal
    # Load the arrays from the .npy files
    if not os.path.exists(file_path1):
        if 'wav' in file_path2:
            return -np.inf
        else:
            return np.inf

    sr1, array1 = load_data(file_path1)
    sr2, array2 = load_data(file_path2)

    if np.sum(array1**2) == 0 or np.isnan(np.sum(array1)):
        # this is an empty signal
        if 'wav' in file_path2:
            return np.nan
        # else:
        #     return np.nan

    if array1.size == 0 or array1 is None:
        return np.nan

    length = min(array1.shape[0], array2.shape[0])

    if 'wav' in file_path2:
        # use sdr
        if sr2 != sr1:
            num_samples1 = int(len(array1) * (sr2 / sr1))
            resampled_array1 = resample(array1, num_samples1)
            resampled_array1 = resampled_array1.astype(array1.dtype)
            array1 = resampled_array1
            length = min(array1.shape[0], array2.shape[0])

        # resampled_file1_path = ''.join(file_path1.split('.wav')[:-1])+'rs.wav'
        mse, _, _, _ = mir_eval.separation.bss_eval_sources(
            array2[:length], array1[:length]
        )

        mse = mse[0]
    elif 'synthesis' in file_path2:
        if array1.dtype.kind in 'fc':
            return np.inf

        max_n = max(array1.max(), array2.max())
        _array_1, _array_2 = np.zeros(max_n), np.zeros(max_n)
        for n in array1:
            _array_1[n-1] = 1
        for n in array2:
            _array_2[n-1] = 1
        # mse = f1_score(_array_2, _array_1)

        mse = my_f1_score(_array_2, _array_1)
        # pdb.set_trace()
    elif 'recover_respiration_trace' in file_path2:
        _, org_data = load_data(args.input_file[0])
        identified_trace = org_data[:, array1].squeeze()
        array1 = identified_trace
        array1 = (array1 - array1.min())/(array1.max() - identified_trace.min())
        array2 = (array2 - array2.min()) / (array2.max() - array2.min())
        # mse = np.sqrt(np.mean((identified_trace-array2)**2))
        correlation_matrix = np.corrcoef(array1, array2)
        mse = correlation_matrix[0, 1]
        mse = abs(mse)
    elif 'recover_respiration' in file_path2:
        mse = np.abs((array1[:length]))[0]
    else:
        
        # Compute the Mean Squared Error (MSE)
        mse = np.mean((array1[:length] - array2[:length]) ** 2)

    # if 'ppg-imputation' in file_path1 or \
    #     'ppg-extrapolation' in file_path1:
    #     mse /= 100*2

    return mse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--input", type=str, default=None, help="input file"
        )
    parser.add_argument(
            "--gt", type=str, default=None, help="ground truth"
        )
    parser.add_argument(
            "--output", type=str, default=None, help="output file"
        )
    args = parser.parse_args()

    # Example usage
    file1 = args.input

    mse1 = compute_mse(file1, args.gt)
    # pdb.set_trace()
    print(mse1)
    # # df = pd.DataFrame({'{}'.format(file1.split('/')[-1]): [mse1], '{}'.format(file2.split('/')[-1]): [mse2]})
    # dirs = file1.split('/')[:-1]
    # dirs = '/'.join(dirs)
    # result_file = args.gt.split('/')[-1].split('.')[0]
    # # df.to_csv(dirs+'/{}.csv'.format(result_file), index=False)
    # rows = ['mse', mse1]
    # with open(dirs+'/result.csv', 'a', newline='') as csvfile:
    #     # Creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     # Appending the data
    #     csvwriter.writerow(rows)