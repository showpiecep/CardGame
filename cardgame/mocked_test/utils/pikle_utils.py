import pickle


def save_obj2pkl(obj, filepath):
    with open(filepath, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_pkl_obj(filepath):
    with open(filepath, "rb") as file:  # 'rb' = read binary
        loaded_data = pickle.load(file)
    return loaded_data
