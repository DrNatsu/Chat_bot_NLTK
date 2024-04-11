# for future you, you can call this function to download and load all packages for the same usecase

def ImportPack():
    import nltk
    import json
    import pickle
    import numpy as np
    import random
    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.models import load_model

    nltk.download('punkt')
    nltk.download('wordnet')
