def handle_input_file_format(dataset_dir, args):
    if 'speech' in args.query:
        args.input_file = [dataset_dir + args.query + '/' + args.index + '.wav']
    elif 'imputation' in args.query or 'extrapolation' in args.query:
        args.input_file = [dataset_dir + args.query + '/' + args.index + '_50.npy']
    elif 'gait-delay_detection' in args.query:
        args.input_file = [dataset_dir + args.query + '/' + args.index + '_1.npy', \
                           dataset_dir + args.query + '/' + args.index + '_2.npy']
    elif 'gait-period_detection' in args.query:
        args.input_file = [dataset_dir + args.query + '/' + args.index + '_1.npy']
    else:
        args.input_file = [dataset_dir + args.query + '/' + args.index + '.npy']

def handle_target_and_output_file(args):
    if 'ecg' in args.input_file[0]:
        args.file = 'ecg_data'
    elif 'ppg' in args.input_file[0]:
        args.file = 'ppg'
    else:
        args.file = 'general'