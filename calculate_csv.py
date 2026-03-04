import os
import pandas as pd
import numpy as np 
import pdb
from scipy.stats import trim_mean

def trimmed_std(data, proportion_to_cut=0.15):

    data = data.dropna()

    # Ensure the data is a numpy array
    data = np.asarray(data)
    
    # Sort the data
    sorted_data = np.sort(data)
    
    # Calculate indices for trimming
    n = len(sorted_data)
    lower_idx = int(np.floor(proportion_to_cut * n))
    upper_idx = int(np.ceil((1 - proportion_to_cut) * n))
    
    # Select the middle portion of the data
    trimmed_data = sorted_data[lower_idx:upper_idx]
    
    # Compute the standard deviation of the trimmed data
    trimmed_std_value = np.std(trimmed_data, ddof=1)  # Use ddof=1 for sample standard deviation

    trimmed_std_value /= np.sqrt(len(trimmed_data))
    
    return trimmed_std_value

order = [
    "ecg_data-extrapolation",
    "ecg_data-gaussian",
    "ecg_data-heartrate",
    "ecg_data-imputation",
    "ecg_data-motion",
    "ecg_data-powerline_1",
    "ecg_data-powerline_2",
    "gait-delay_detection",
    "gait-period_detection",
    "ppg-extrapolation",
    "ppg-imputation",
    "resampling",
    "speech-Siren",
    "speech-TelephoneRing1",
    "speech-TelephoneRing2",
    "speech-TelephoneRing3",
    "speech-echo",
    "change_point_detect_1",
    "change_point_detect_2",
    "change_point_detect_3",
    "outlier_detect_1",
    "outlier_detect_2",
    "outlier_detect_3",
    "change_point_detect_4",
    "outlier_detect_4"
]

def insert (source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]

# _subject = 'ziqi'
_subject = 'api'
# subject = 'jason_new_results'
# subject = 'AddtionalBenchmark-ziqi'
# Specify the directory to search for files
# csv_file_path = './llm_response/no_api.csv'
# csv_file_path = './llm_response/api_Llama-3-70b_2.csv'
# csv_file_path = './conv_history/gpt-4o_api_2_#trial_5.csv'
csv_file_paths = [
                # './conv_history/human.csv',
                # './ziqi.csv',
                # './conv_history/gpt-4-turbo_base_self_coding_#trial_1.csv',
                # './conv_history/gpt-3.5-turbo_base_self_coding_#trial_1.csv',
                # './conv_history/meta-llama/Llama-3-70b-chat-hf_base_self_coding_#trial_1.csv',
                
                # # './conv_history/gpt-4o_base_self_coding_#trial_1.csv', \

                # #   './conv_history/gpt-4o_base_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-4o_CoT_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-4o_api_2_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-4o_api_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-4o_api_self_coding_#trial_3.csv', \
                # #   './conv_history/gpt-4o_api_self_coding_#trial_4.csv', \
                # #   './conv_history/gpt-4o_api_self_coding_#trial_5.csv', \

                # #   './conv_history/gpt-4o_api_self_coding_#trial_1_s78.csv', \
                # #   './conv_history/gpt-4o_api_2_self_coding_#trial_1_s78.csv', \
                # #   './conv_history/gpt-4o_base_self_coding_#trial_1_s78.csv', \
                # #   './conv_history/gpt-4o_api_self_coding_#trial_5_s78.csv', \

                # # './conv_history/gpt-4o_api_self_coding_#trial_5_bk12_synthesis.csv', \
                # # './conv_history/gpt-4o_api_self_coding_#trial_1_bk12_synthesis.csv', \
                # # './conv_history/gpt-4o_base_self_coding_#trial_1_bk12_synthesis.csv', \
                # './conv_history/gpt-4o_base_self_coding_#trial_1.csv',
                
                # './conv_history/gpt-4o_no_api_self_coding_#trial_1.csv',
                # './conv_history/gpt-4o_text_self_coding_#trial_1.csv',

                # './conv_history/gpt-4-turbo_no_api_self_coding_#trial_1.csv',
                # './conv_history/gpt-4-turbo_text_self_coding_#trial_1.csv',

                # './conv_history/gpt-3.5-turbo_no_api_self_coding_#trial_1.csv',
                # './conv_history/gpt-3.5-turbo_text_self_coding_#trial_1.csv',

                # './conv_history/meta-llama/Llama-3-70b-chat-hf_no_api_self_coding_#trial_1.csv',
                # './conv_history/meta-llama/Llama-3-70b-chat-hf_text_self_coding_#trial_1.csv',

                # # './conv_history/Qwen/Qwen2-72B-Instruct_base_self_coding_#trial_1.csv',

                # './results/gpt-4o_base_self_coding_#trial_1_bk_12_all_task.csv', \
                # './results/gpt-4o_CoT_self_coding_#trial_1_bk_12_all_task.csv', \
                # './results/gpt-4o_api_2_self_coding_#trial_1_bk_12_all_task.csv', \
                # './results/gpt-4o_api_self_coding_#trial_1_bk_12_all_task.csv', \
                # './results/gpt-4o_api_self_coding_#trial_3_bk_12_all_task.csv', \
                # './results/gpt-4o_api_self_coding_#trial_4_bk_12_all_task.csv', \
                # './results/gpt-4o_api_self_coding_#trial_5_bk_12_all_task.csv', \

                # #   './conv_history/gpt-3.5-turbo_base_self_coding_#trial_1.csv',\
                # #   './conv_history/gpt-3.5-turbo_CoT_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-3.5-turbo_api_2_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-3.5-turbo_api_self_coding_#trial_1.csv', \
                # #   './conv_history/gpt-3.5-turbo_api_self_coding_#trial_5.csv',

                # './conv_history/org-llama-70B_api_2_self_coding_#trial_1_full.csv',
                # './conv_history/org-llama-70B_api_2_self_coding_#trial_1_full_new.csv',
                # './conv_history/finetuned-llama-70B_api_2_self_coding_#trial_1_full_new.csv',
                # './conv_history/finetuned-llama-70B_api_2_self_coding_#trial_1_full.csv',
                # # './conv_history/org-llama-3.0-70B_api_2_self_coding_#trial_1.csv',
                # './conv_history/oliver.csv',
                # './conv_history/gpt-4o_api_2_self_coding_#trial_1_bk_12_all_task.csv',
                # './conv_history/gpt-4-turbo_base_self_coding_#trial_1.csv',
                # './conv_history/gpt-3.5-turbo_api_2_self_coding_#trial_1-1.csv',
                # './conv_history/Llama-3-70b-chat-hf_base_self_coding_#trial_1.csv',

                # './conv_history/org-llama-3B_api_2_self_coding_#trial_1_full.csv',
                './conv_history/o1_base_self_coding_#trial_1.csv',
]
# index_list = [1, 2, 3]
index_list = [1, 2, 3]
# index_list = [a for a in range(20, 39)]
# index_list = [4, 5, 6]
for i, csv_file_path in enumerate(csv_file_paths):
    # pdb.set_trace()
    if not os.path.exists(csv_file_path):
        continue
    
    if 'api_2' in csv_file_path:
        subject = 'api_2'
    elif 'base' in csv_file_path:
        subject = 'base'
    elif 'CoT' in csv_file_path:
        subject = 'CoT'
    elif 'text' in csv_file_path:
        subject = 'text'
    elif 'no_api' in csv_file_path:
        subject = 'no_api'
    elif 'oliver' in csv_file_path:
        subject = 'oliver'
    elif 'ziqi' in csv_file_path:
        subject = 'ziqi'
    elif 'human' in csv_file_path:
        subject = 'human'
    else:
        subject = _subject
    
    new_csv_file_path = insert(csv_file_path, '_avg', -4)

    df = pd.read_csv(csv_file_path)

    filtered_task = ['ecg_data-powerline_3', 'ecg_data-heartrate-2', 'ecg_data-noise',\
                     'ecg_data-resampling']

    # Group the data by 'task' and 'index', then filter out indexes that are not 1, 2, or 3
    # filtered_df = df[df['index'].isin([1, 2, 3])]

    # filtered_df = df[df['mode']=='api']
    filtered_df = df[df['mode']==f'{subject}']
    filtered_df = filtered_df[~filtered_df['task'].isin(filtered_task)]
    # filtered_df = df[df['mode']=='human']
    # pdb.set_trace()
    filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    total_nan = filtered_df.isna().sum().sum()
    
    print("Total NaN values:", total_nan, " fail rate: {:.4f}".format(total_nan/len(filtered_df)))
    
    nan_percentage = df.groupby('task')['score'].apply(lambda x: x.isna().mean() * 100).reset_index()
    nan_percentage.columns = ['task', 'nan_percentage']

    # Condition for NaN scores and task containing 'synthesis'
    condition = filtered_df['score'].isna() & df['task'].str.contains('synthesis')

    # Update the score to 0 for rows meeting the condition
    filtered_df.loc[condition, 'score'] = 0
    # filtered_df = filtered_df.dropna().reset_index(drop=True)
    
    # filtered_df = filtered_df.fillna(0)
    
    # Calculate the average score for each task with index from 1 to 3
    # average_scores = filtered_df.groupby('task')['score'].mean().reset_index()
    
    # Get scores for [1, 2, 3]
    filtered_df = filtered_df[filtered_df['index'].isin(index_list)]
    
    filtered_df = filtered_df[~filtered_df['task'].str.contains('4s')]

    # Define a function to calculate the trimmed mean
    def calculate_trimmed_mean(group, proportion_to_cut):
        group = group.dropna()
        return trim_mean(group, proportion_to_cut)

    # Calculate the task-wise trimmed mean by excluding the top and bottom 10%
    proportion_to_cut = 0.15
    # pdb.set_trace()
    filtered_df.loc[(filtered_df['task']=='ppg-imputation') & (filtered_df['index']==3), 'score'] /= 100**2
    filtered_df.loc[(filtered_df['task']=='ppg-extrapolation') & (filtered_df['index']==3), 'score'] /= 100**2

    # filtered_df['score'] = filtered_df['score'].abs()
    average_scores = filtered_df.groupby('task')['score'].apply(calculate_trimmed_mean, proportion_to_cut).reset_index()
    # Deal with the large value in ppg
    # pdb.set_trace()

    # Calculate standard deviation scores
    # average_scores = filtered_df.groupby('task')['score'].mean().reset_index()


    # Calculate standard deviation scores
    # std_scores = filtered_df.groupby('task')['score'].std().reset_index()
    std_scores = filtered_df.groupby('task')['score'].apply(trimmed_std, proportion_to_cut).reset_index() 

    # Rename the columns to distinguish them
    average_scores.rename(columns={'score': 'mean_score'}, inplace=True)
    std_scores.rename(columns={'score': 'std_score'}, inplace=True)

    # Merge the average_scores and std_scores DataFrames
    average_scores = pd.merge(average_scores, std_scores, on='task')
    average_scores = pd.merge(average_scores, nan_percentage, on='task')
    if 'change_point_detect_1' in average_scores['task'].tolist():
        # pdb.set_trace()
        average_scores['task'] = pd.Categorical(average_scores['task'], categories=order, ordered=True)
        average_scores = average_scores.sort_values(by='task')

    average_scores.to_csv(new_csv_file_path, index=False) 

    average_scores._append({'task':'NaN rate', 'score': str(total_nan/len(filtered_df))}, ignore_index=True)

    # pd.options.display.float_format = '{:.6f}'.format
    # Print the average scores
    print(csv_file_path)
    print(average_scores)
