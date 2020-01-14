import h5py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

from lib import load_data
from lib import model_analysis as aa


def main():

    root_path = os.getcwd()
    tds_file = os.path.join(root_path, "modis_artificial_tds_8cat.hdf")

    # # folder = (os.path.join(wv_tds_folder, '*JUL*tds.hdf'))
    features, labels = load_data.load_tds_file(tds_file)

    # Whether or not to do a test/train split
    tts = False
    features_test = []
    labels_test = []
    if tts:
        # Create a test set as a split from the main set.
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                    test_size=0.3, random_state=51)
        labels_train = np.array(labels_train, dtype=np.float32)
        labels_test = np.array(labels_test, dtype=np.float32)
        features_train = np.array(features_train, dtype=np.float32)
        features_test = np.array(features_test, dtype=np.float32)
    else:
        features_train = np.array(features, dtype=np.float32)
        labels_train = np.array(labels, dtype=np.float32)

    # Display model and test input information
    print("Size of training set: {}".format(len(features_train)))
    if tts:
        print("Size of test set:     {}".format(len(features_test)))

    print(np.shape(features_train))
    print(np.shape(labels_train))

    # sklearn Random Forest
    rfc = RandomForestRegressor(n_estimators=100, oob_score=True)
    rfc.fit(features_train, labels_train)

    # Save the model
    root_path = os.getcwd()
    model_filename = os.path.join(root_path, "rfc_8cat.p")
    with open(model_filename, 'wb') as mf:
        pickle.dump(rfc, mf)

    # To test an existing model:
    #model_filename = '/Users/nicholas/Documents/Dartmouth/Projects/modis_unmixing/ANN/artificial_tds/rfc_model_8cat.p'
    #with open(model_filename, 'rb') as mf:
    #    rfc = pickle.load(mf)

    print(rfc.oob_score_)

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    feature_names_sorted = []
    feature_names = range(1,7)
    for f in range(6):
        print("%d. feature %s (%f)" % (f+1, feature_names[indices[f]], importances[indices[f]]))
        feature_names_sorted.append(feature_names[indices[f]])
    #
    # If we split a test set, use it for some tests!
    if tts:
        labels_predict = rfc.predict(features_test)
        aa.print_stats(labels_test, labels_predict)
        aa.plot_results(labels_test, labels_predict)


def write_hdf(fname, features, labels, weights):
    tds_file = h5py.File(fname, 'w')
    tds_file.create_dataset("features", data=features)
    tds_file.create_dataset("labels", data=labels)
    tds_file.create_dataset("weights", data=weights)
    tds_file.close()


if __name__ == '__main__':
    main()