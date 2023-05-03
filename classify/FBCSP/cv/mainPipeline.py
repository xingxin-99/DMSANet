from MLEngine import MLEngine

if __name__ == "__main__":

    '''Example for loading BCI Competition IV Dataset 2a'''
    # dataset_details = {
    #     'data_path': "C:\\Users\\BCIgroup\\Desktop\\star\\Coding\\DMSANet\\data\\bci42s\\originalData",
    #     'file_to_load': 'A01T.gdf',
    #     'file_to_label':'A01T.mat',
    #     'ntimes': 1,
    #     'kfold': 10,
    #     'm_filters': 4,
    #     'window_details': {'tmin': 0, 'tmax': 4}
    # }

    '''Example for loading HGD Dataset'''
    dataset_details={
        'data_path' : "C:\\Users\\BCIgroup\\Desktop\\star\\Coding\\DMSANet\\data\\hgd\\originalData",
        'file_to_load': 'A14T.mat',
        'ntimes': 1,
        'kfold':10,
        'm_filters':4,
        'window_details':{'tmin':0,'tmax':4}
    }

    ML_experiment = MLEngine(**dataset_details)
    ML_experiment.experiment()
