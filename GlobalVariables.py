class GlobalVariables :

    #options for running different experiments
    use_samples = 0
    use_pitch =0
    use_spectrogram = 1
    use_raw_data = 0

    use_dense=0
    use_CNN_1D=0
    use_CNN_2D=1
    #Grid Size

    nRow = 3
    nCol = 3

    start=1  #0 for fixed start position and 1 for random start position
    goal=1 #0 for fixed goal position and 1 for random goal position

    #parameters
    sample_state_size = 100
    pitch_state_size= 87#114
    spectrogram_length=129
    spectrogram_state_size= 259
    raw_data_state_size= 100
    action_size = 4
    batch_size = 32
    Number_of_episodes=100
    timesteps=50#50#(nRow+nCol+nRow)
    how_many_times = 2 #How many times to run the same experiment

