import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import kde


def print_stats(labels_test, labels_predict):
    ''''
    Calculate the following statistics from machine learning tests.
    RMSE, Bias, linregress results
    '''
    # labels_predict = regr.predict(features_test)
    sqerr_sum_i = 0
    sqerr_sum_pl = 0
    sqerr_sum_w = 0
    b_i = 0
    b_pl = 0
    b_w = 0
    rmse_counter = 0
    average_sum = 0

    for i in range(len(labels_test)):

        b_i += np.subtract(labels_predict[i][0], labels_test[i][0])
        b_pl += np.subtract(labels_predict[i][1], labels_test[i][1])
        b_w += np.subtract(labels_predict[i][2], labels_test[i][2])

        sqerr_sum_i += np.square(np.subtract(labels_predict[i][0], labels_test[i][0]))
        sqerr_sum_pl += np.square(np.subtract(labels_predict[i][1], labels_test[i][1]))
        sqerr_sum_w += np.square(np.subtract(labels_predict[i][2], labels_test[i][2]))
        rmse_counter += 1.

        average_sum += np.sum(labels_predict[i,:])

    rmse_i = np.sqrt(sqerr_sum_i / rmse_counter)
    rmse_pl = np.sqrt(sqerr_sum_pl / rmse_counter)
    rmse_w = np.sqrt(sqerr_sum_w / rmse_counter)

    bias_i = b_i / len(labels_test)
    bias_pl = b_pl / len(labels_test)
    bias_w = b_w / len(labels_test)

    print("RMSE: I: {0:0.3f} | P: {1:0.3f} | W: {2:0.3f}".format(rmse_i, rmse_pl, rmse_w))

    print("Bias: I: {0:0.3f} | P: {1:0.3f} | W: {2:0.3f}".format(bias_i, bias_pl, bias_w))

    ice_stats = stats.linregress(labels_test[:, 0],labels_predict[:, 0])
    print("ice: Slope: {0:0.3f} | Int: {1:0.3f} | R: {2:0.3f}".format(ice_stats[0],
                                                                      ice_stats[1],
                                                                      ice_stats[2]))
    pnd_stats = stats.linregress(labels_test[:, 1], labels_predict[:, 1])
    print("pnd: Slope: {0:0.3f} | Int: {1:0.3f} | R: {2:0.3f}".format(pnd_stats[0],
                                                                      pnd_stats[1],
                                                                      pnd_stats[2]))
    ocn_stats = stats.linregress(labels_test[:, 2], labels_predict[:, 2])
    print("ocn: Slope: {0:0.3f} | Int: {1:0.3f} | R: {2:0.3f}".format(ocn_stats[0],
                                                                      ocn_stats[1],
                                                                      ocn_stats[2]))

    print("Average sum: {}".format(average_sum/rmse_counter))

    # r_score = regr.score(features_test,labels_test)
    # print("Score: {}".format(r_score))


def plot_results(labels_test, labels_predict):

    fig, axs = plt.subplots(ncols=3, sharey=True)
    ax = axs[0]
    ax.hist2d(labels_test[:, 0], labels_predict[:, 0],
              bins=90,
              norm=colors.LogNorm(),
              cmap='PuBu')
    ax.axis([0, 1, 0, 1])
    ax.set_title("Ice")

    ax = axs[1]
    ax.hist2d(labels_test[:, 1], labels_predict[:, 1],
              bins=90,
              norm=colors.LogNorm(),
              cmap='PuBu')
    ax.axis([0, 1, 0, 1])
    ax.set_title("Pond")
    ax.set_xlabel("True")

    ax = axs[2]
    ax.hist2d(labels_test[:, 2], labels_predict[:, 2],
              bins=90,
              norm=colors.LogNorm(),
              cmap='PuBu')
    ax.axis([0, 1, 0, 1])
    ax.set_title("Ocean")

    fig.tight_layout()

    plt.show()
    # x = labels_test[:, 0]
    # y = labels_predict[:, 0]
    # nbins = 300
    # k = kde.gaussian_kde([x, y])
    # xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    # xi, yi = np.mgrid[0:1:nbins * 1j, 0:1:nbins * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Reds_r)