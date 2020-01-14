import os
import glob
import h5py
import csv
import numpy as np


def load_tds_file(filename):

    # Define the master training feature set and associated labels as the first tds file
    with h5py.File(filename, 'r') as infile:
        features = infile['features'][:]
        labels = infile['labels'][:]

    # if np.shape(labels)[1] == 4:
    #     labels = four_to_three(labels)

    # features = add_feature(features)

    return features, labels


def load_oib_csv(filename, dst_file=None):

    features = []
    labels = []
    with open(filename, 'r') as cf:
        csv_reader = csv.reader(cf)
        # Skip header
        next(cf)
        for row in csv_reader:
            features.append([float(row[9]), float(row[10]), float(row[11]),
                             float(row[12]), float(row[13]), float(row[14]),
                             float(row[15])])
            labels.append([float(row[2]), float(row[4]), float(row[6])])


    # For any duplicate modis reflectances, average all the associated icebridge obs
    n = 1
    i = 0
    while i < len(features)-1:
        if features[i] == features[i+1]:
            next_label = labels.pop(i+1)
            labels[i][0] += next_label[0]
            labels[i][1] += next_label[1]
            labels[i][2] += next_label[2]
            n += 1
            features.pop(i+1)
        # IF the next one is different, divide the running sum by n and reset n,
        # then increment the counter
        else:
            labels[i][0] = (labels[i][0] / n)
            labels[i][1] = (labels[i][1] / n)
            labels[i][2] = (labels[i][2] / n)
            n = 1
            i += 1

    labels = np.array(labels)
    labels_new = np.zeros(np.shape(labels))
    labels_new[:, 0] = np.convolve(labels[:, 0], np.ones(7)/7, mode='same')
    labels_new[:, 1] = np.convolve(labels[:, 1], np.ones(7)/7, mode='same')
    labels_new[:, 2] = np.convolve(labels[:, 2], np.ones(7)/7, mode='same')

    labels_new = labels_new[4:-4]
    features = features[4:-4]

    # features = add_feature(features)

    # w = 5
    # for i in range(len(labels)):
    #     if i < w:
    #         labels_new[i, :] = np.mean(labels[i: i+w, :])
    #     elif i > len(labels)-w:
    #         labels_new[i, :] = np.mean(labels[i-w : i, :])
    #     else:
    #         labels_new[i, :] = np.mean(labels[i-w:i+w, :])

    if dst_file:
        with open(dst_file, 'w') as cf:
            csv_writer = csv.writer(cf)
            for i in range(len(labels_new)):
                csv_writer.writerow(np.append(labels_new[i], features[i], axis=0))

    return features, labels_new


def load_tds_folder(folder, to_skip=None):
    '''
    Loads a whole folder worth of training data
    :param folder: folder with training data
    :param to_skip: list of indices to not load
    :return: training features and labels
    '''
    if to_skip is None:
        to_skip = []
    skipped = []

    tds_file_list = glob.glob(os.path.join(folder, '*tds.hdf'))

    tds_file1 = tds_file_list[0]
    print("Loading {}".format(tds_file1))
    # Define the training feature set and associated labels as the first tds file
    with h5py.File(tds_file1, 'r') as infile:
        features = infile['features'][:]
        labels = infile['labels'][:]

    #Load each training dataset and append the contained information to the master feature and label lists
    for i in range(1,len(tds_file_list)):
        if i in to_skip:
            skipped.append(tds_file_list[i])
            continue
        tds_file = tds_file_list[i]
        print("Loading {}".format(tds_file))
        with h5py.File(tds_file, 'r') as infile:
            features_new = infile['features'][:]
            features = np.append(features, features_new, axis=0)
            labels_new = infile['labels'][:]
            labels = np.append(labels, labels_new, axis=0)

    if np.shape(labels)[1] == 4:
        labels = four_to_three(labels)

    features = add_feature(features)

    return features, labels, skipped


def four_to_three(labels):
    '''
    Converts data with 4 categories (snow/light/dark/ocean) to
    3 categories (snow/pond/ocean)
    :param labels:
    :return:
    '''
    labels_reorg = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        labels_reorg[i, 0] = labels[i, 0]
        labels_reorg[i, 1] = labels[i, 1] + labels[i, 2]
        labels_reorg[i, 2] = labels[i, 3]

    return labels_reorg


def add_feature(features):

    features_list = []
    for i in range(len(features)):
        if (features[i][1] + features[i][2]) != 0:
            ratio1 = (features[i][2] - features[i][1]) / (features[i][2] + features[i][1])
        else:
            ratio1 = 0.

        new_feature = np.append(features[i], ratio1)
        new_feature = np.append(new_feature, np.random.rand())

        features_list.append(new_feature)

    return features_list