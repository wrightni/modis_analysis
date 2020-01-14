import os
import sys
import csv
import pickle
import numpy as np
import argparse
import colorsys
from tqdm import tqdm
import gdal
import matplotlib
matplotlib.use('tkagg')
import h5py

from monte_carlo_simulation import monte_carlo_simulation, monte_carlo_ml
import block_surface_fraction as bsf
import rosel_alg
sys.path.append('/Users/nicholas/Documents/Dartmouth/Github/OSSP')
import preprocess as pp
from hist_test import plot_histograms

sys.path.append('/Users/nicholas/Documents/Dartmouth/Projects/modis_unmixing/ANN/artificial_tds/')
from load_data import load_tds_file


def block_albedo(optic_img, clsf_image, factor, pmean=None):
    # Given an input image, this function divides it into blocks based on factor,
    # and calculates the average surface color within each block

    nbands, ydim, xdim = np.shape(optic_img)
    ice_albedo = np.zeros((nbands, int(ydim / factor), int(xdim / factor)))
    pnd_albedo = np.zeros((nbands, int(ydim / factor), int(xdim / factor)))
    ocn_albedo = np.zeros((nbands, int(ydim / factor), int(xdim / factor)))
    pratio_list = []
    total_albedo = np.zeros((nbands, int(ydim / factor), int(xdim / factor)))

    for y in range(0, ydim, factor):
        for x in range(0, xdim, factor):
            for b in range(nbands):
                ice_sum, pnd_sum, ocn_sum = 0, 0, 0
                num_light_ponds, num_dark_ponds = 0, 0
                # Slice the input data block and calculate the surface histogram
                ih, ph, oh = bsf.surface_histograms(optic_img[b, y:y + factor, x:x + factor],
                                                    clsf_image[y:y + factor, x:x + factor],
                                                    factor, factor)
                # Find Average from histogram
                # Multiply bin number (i:pixel value) times the bin count (hist[i])
                #   and sum these to get the total sum from that category
                for i in range(len(ih)):
                    ice_sum += ih[i] * i
                    pnd_sum += ph[i] * i
                    ocn_sum += oh[i] * i
                    # Count the total number of pond pixels that are above and below the image mean
                    if pmean is not None and b == 2:
                        if i > pmean:
                            num_light_ponds += ph[i]
                        else: # i < pmean
                            num_dark_ponds += ph[i]

                if pmean is not None and b == 2:
                    pratio = num_light_ponds / (num_dark_ponds + 1)
                    pratio_list.append(pratio)

                if (np.sum(ih) + np.sum(ph) + np.sum(oh)) < factor*factor*.25:
                    continue
                else:
                    total_albedo[b, int(y / factor), int(x / factor)] = ((ice_sum + pnd_sum + ocn_sum)
                                                                         / (np.sum(ih) + np.sum(ph) + np.sum(oh)))

                # Divide the sum by the total number of pixels (sum of bin counts)
                if np.sum(ih) > factor*factor*.05:
                    ice_average = ice_sum / (np.sum(ih))
                else:
                    ice_average = find_default('ice', b) * 255
                if np.sum(ph) > factor*factor*.05:
                    pnd_average = pnd_sum / (np.sum(ph))
                else:
                    pnd_average = find_default('pnd', b) * 255
                if np.sum(oh) > factor*factor*.05:
                    ocn_average = ocn_sum / (np.sum(oh))
                else:
                    ocn_average = find_default('ocn', b) * 255

                # Convert to float from uint8
                ice_albedo[b, int(y / factor), int(x / factor)] = ice_average / 255.
                pnd_albedo[b, int(y / factor), int(x / factor)] = pnd_average / 255.
                ocn_albedo[b, int(y / factor), int(x / factor)] = ocn_average / 255.

                # if b == 2:
                #     print(ice_average, pnd_average, ocn_average)
                #     print(pmean)
                #     avgs = [ice_average, pmean, ocn_average]
                #     plot_histograms(range(256), ih, ph, oh, avgs)
                #     a = input("Cont?")
                #     if str(a) == 'n':
                #         quit()

    return ice_albedo, pnd_albedo, ocn_albedo, total_albedo, pratio_list


def block_classification(clsf_image, factor):

    ydim, xdim = np.shape(clsf_image)
    blk_fractions = np.zeros((int(ydim/factor), int(xdim/factor), 3))
    for y in range(0, ydim, factor):
        for x in range(0, xdim, factor):
            fs, fp, fo = bsf.block_surface_fraction(clsf_image[y:y+factor, x:x+factor], factor, factor)
            blk_fractions[int(y/factor), int(x/factor), :] = [fs, fp, fo]

    return blk_fractions


def find_image_pmean(src_ds, clsf_ds, dark_ref, white_pt):
    y_dim = src_ds.RasterYSize
    x_dim = src_ds.RasterXSize
    block_size = 2500

    y_blocks = range(0, y_dim, block_size)
    x_blocks = range(0, x_dim, block_size)

    ice_sum = [0 for _ in range(8)]
    pnd_sum = [0 for _ in range(8)]
    ocn_sum = [0 for _ in range(8)]
    ice_count = [0 for _ in range(8)]
    pnd_count = [0 for _ in range(8)]
    ocn_count = [0 for _ in range(8)]

    pbar = tqdm(total=len(y_blocks)*len(x_blocks), unit='blocks')

    for y in y_blocks:
        for x in x_blocks:
            read_size_y = check_read_size(y, block_size, y_dim)
            read_size_x = check_read_size(x, block_size, x_dim)
            if read_size_y != block_size or read_size_x != block_size:
                continue

            optic_data = src_ds.ReadAsArray(x, y, read_size_x, read_size_y)
            clsf_data = clsf_ds.ReadAsArray(x, y, read_size_x, read_size_y)

            for b in range(8):
                if not valid_block(optic_data):
                    continue

                # Apply a dark point reference based on the image histogram (skip band 8)
                if b != 7:
                    optic_data[b, :, :] = pp.rescale_band(optic_data[b, :, :], dark_ref[b], white_pt)

                ih, ph, oh = bsf.surface_histograms(optic_data[b, :, :], clsf_data, block_size, block_size)

                # Find Average from histogram
                # Multiply bin number (i:pixel value) times the bin count (hist[i])
                #   and sum these to get the total sum from that category
                for i in range(len(ih)):
                    ice_sum[b] += ih[i] * i
                    pnd_sum[b] += ph[i] * i
                    ocn_sum[b] += oh[i] * i

                # Record the total number of pixels in the pond category
                ice_count[b] += np.sum(ih)
                pnd_count[b] += np.sum(ph)
                ocn_count[b] += np.sum(oh)
            pbar.update()
    pbar.close()

    ice_albedo = np.divide(ice_sum, ice_count)
    pnd_albedo = np.divide(pnd_sum, pnd_count)
    ocn_albedo = np.divide(ocn_sum, ocn_count)

    # quit()
    srm = [[ocn_albedo[4]/255, pnd_albedo[4]/255, ice_albedo[4]/255],
           [ocn_albedo[6]/255, pnd_albedo[6]/255, ice_albedo[6]/255],
           [ocn_albedo[1]/255, pnd_albedo[1]/255, ice_albedo[1]/255],
           [1, 1, 1]]

    # Divide the sum by the count to get mean
    return (pnd_sum[2] / pnd_count[2]), np.array(srm)


def analyze_imagery(optic_image, clsf_image, ds_clsf_data, f, pmean=None):
    '''
    Reads through the input imagery to extract the variables needed for later processing
    '''

    # Output variables
    srm_list = []
    refl_list = []
    pond_hsv_list = []
    ocean_hsv_list = []
    true_fraction_list = []
    pond_ocean_diff_list = []
    # Find the average reflectance of each surface within the chosen block size.
    ice_albedo, pnd_albedo, ocn_albedo, total_albedo, pratio = block_albedo(optic_image, clsf_image, f, pmean=pmean)

    # Change all of the nan values back to 0
    ice_albedo = np.nan_to_num(ice_albedo)
    pnd_albedo = np.nan_to_num(pnd_albedo)
    ocn_albedo = np.nan_to_num(ocn_albedo)

    # Loop through each pseudo_modis pixel
    for i in range(np.shape(total_albedo)[1]):
        for j in range(np.shape(total_albedo)[2]):

            # Store the true fraction for this pseudo modis pixel
            if np.sum(ds_clsf_data[i, j]) == 0:
                continue
            true_fraction_list.append(ds_clsf_data[i, j])

            ## Calculate the best spectral reflectance matrix for the current pixel
            srm = [[ocn_albedo[4, i, j], pnd_albedo[4, i, j], ice_albedo[4, i, j]],
                   [ocn_albedo[6, i, j], pnd_albedo[6, i, j], ice_albedo[6, i, j]],
                   [ocn_albedo[1, i, j], pnd_albedo[1, i, j], ice_albedo[1, i, j]],
                   [1, 1, 1]]

            # srm = assert_defaults(srm)

            ## Find the reflectance of the current pixel
            # refl = [total_albedo[4, i, j], total_albedo[6, i, j], total_albedo[1, i, j]]
            refl = total_albedo[:, i, j]            # ML needs the full array
            refl = np.divide(refl, 255.)

            # Store the srm and reflectance for output
            srm_list.append(srm)
            refl_list.append(refl)

            ## Save the color of ponds and ocean as HSV
            pond_hsv = colorsys.rgb_to_hsv(pnd_albedo[6, i, j], pnd_albedo[4, i, j], pnd_albedo[2, i, j])
            ocean_hsv = colorsys.rgb_to_hsv(ocn_albedo[6, i, j], ocn_albedo[4, i, j], ocn_albedo[2, i, j])

            pond_ocean_diff = np.sqrt((pnd_albedo[6, i, j] - ocn_albedo[6, i, j]) ** 2 +
                                      (pnd_albedo[4, i, j] - ocn_albedo[4, i, j]) ** 2 +
                                      (pnd_albedo[2, i, j] - ocn_albedo[2, i, j]) ** 2)
            pond_hsv_list.append(pond_hsv)
            ocean_hsv_list.append(ocean_hsv)
            pond_ocean_diff_list.append(pond_ocean_diff)


    # Convert these lists to numpy arrays
    srm_list = np.array(srm_list)
    refl_list = np.array(refl_list)
    pond_hsv_list = np.array(pond_hsv_list)
    ocean_hsv_list = np.array(ocean_hsv_list)
    true_fraction_list = np.array(true_fraction_list)

    return srm_list, refl_list, pond_hsv_list, ocean_hsv_list, true_fraction_list, pond_ocean_diff_list, pratio


def stage_one_unmixing(refl_list, srm_list, true_list):
    '''
    Perform stage 1 unmixing, where each entry in reflectance list is unmixed by
        a unique spectral reflectance matrix

    :param refl_list: List of pixel reflectance triplets
    :param srm_list: SRM for each associated reflectance triplet
    :return unmix_fraction: Unmixed fraction for each entry in refl_list
    :return unmix_error: Montecarlo error approx for each unmix attempt
    '''
    # Output data
    unmix_fraction = []
    unmix_error = []

    num_obs = len(refl_list)

    # Unmix each reflectance value with its associated srm
    for i in range(num_obs):

        refl = refl_list[i]
        srm = srm_list[i]

        # refl might be more than 3 entries if we stored extras for the ML process
        if len(refl) != 3:
            refl = [refl[4], refl[6], refl[1]]

        ## Approximate the surface distribution with unmixing
        unmix_fraction_i = rosel_alg.spec_unmix(refl, 1, srm)
        unmix_fraction_i = np.flip(unmix_fraction_i)
        unmix_fraction.append(np.divide(unmix_fraction_i, 1000.))

        # Apply a MonteCarlo error propagation
        mc_errors = monte_carlo_simulation(refl, srm)
        unmix_error.append(np.mean(mc_errors))

    return np.array(unmix_fraction), np.array(unmix_error)


def stage_two_unmixing(refl_list, srm):
    '''
    Perform stage 2 unmixing, where each entry in reflectance list is unmixed by
        the same spectral reflectance matrix

    :param refl_list: List of pixel reflectance triplets
    :param srm_list: Single (average) SRM
    :return unmix_fraction: Unmixed fraction for each entry in refl_list
    :return unmix_error: Montecarlo error approx for each unmix attempt
    '''
    # Output data
    unmix_fraction = []
    unmix_error = []

    i = 0

    # Unmix each reflectance value with its associated srm
    for refl in refl_list:

        # refl might be more than 3 entries if we stored extras for the ML process
        if len(refl) != 3:
            refl = [refl[4], refl[6], refl[1]]

        ## Approximate the surface distribution with unmixing
        unmix_fraction_i = rosel_alg.spec_unmix(refl, 1, srm)
        unmix_fraction_i = np.flip(unmix_fraction_i)
        unmix_fraction.append(np.divide(unmix_fraction_i, 1000.))

        if i == 0:
            print("Stage2 SRM:")
            print_srm(srm)
            i+=1

        # Apply a MonteCarlo error propagation
        mc_errors = monte_carlo_simulation(refl, srm)
        unmix_error.append(np.mean(mc_errors))

    return np.array(unmix_fraction), np.array(unmix_error)


def stage_three_unmixing(refl_list, srm):
    # Output data
    unmix_fraction = []
    unmix_error = []

    # Unmix each reflectance value with its associated srm
    for refl in refl_list:

        # refl might be more than 3 entries if we stored extras for the ML process
        if len(refl) != 3:
            refl = [refl[4], refl[6], refl[1]]

        ## Approximate the surface distribution with unmixing
        unmix_fraction_i = rosel_alg.spec_unmix(refl, 1, srm)
        unmix_fraction_i = np.flip(unmix_fraction_i)
        unmix_fraction.append(np.divide(unmix_fraction_i, 1000.))

        # Apply a MonteCarlo error propagation
        mc_errors = monte_carlo_simulation(refl, srm)
        unmix_error.append(np.mean(mc_errors))

    return np.array(unmix_fraction), np.array(unmix_error)


def ml_estimation(refl_list, model):

    # Output data
    unmix_fraction = []
    unmix_error = []

    # Unmix each reflectance value with its associated srm
    for refl in refl_list:

        ## Approximate the surface distribution with unmixing
        unmix_fraction_i = rosel_alg.ml_estimation(refl, model)
        # unmix_fraction_i = np.flip(unmix_fraction_i)
        unmix_fraction.append(np.divide(unmix_fraction_i, 1000.))

        # Apply a MonteCarlo error propagation
        mc_errors = monte_carlo_ml(refl, model)
        # mc_errors = 0
        unmix_error.append(np.mean(mc_errors))

    return np.array(unmix_fraction), np.array(unmix_error)


def calculate_rmse(true_fraction_all, s1_fraction_all, s2_fraction_all,
                   s3_fraction_all, s4_fraction_all):

    s1_rmse_all = []
    s2_rmse_all = []
    s3_rmse_all = []
    s4_rmse_all = []

    for i in range(len(true_fraction_all)):

        tf = true_fraction_all[i]
        s1f = s1_fraction_all[i]
        s2f = s2_fraction_all[i]
        s3f = s3_fraction_all[i]
        s4f = s4_fraction_all[i]

        s1_rmse = np.sqrt(np.mean(np.square(np.subtract(s1f, tf))))
        s2_rmse = np.sqrt(np.mean(np.square(np.subtract(s2f, tf))))
        s3_rmse = np.sqrt(np.mean(np.square(np.subtract(s3f, tf))))
        s4_rmse = np.sqrt(np.mean(np.square(np.subtract(s4f, tf))))

        s1_rmse_all.append(s1_rmse)
        s2_rmse_all.append(s2_rmse)
        s3_rmse_all.append(s3_rmse)
        s4_rmse_all.append(s4_rmse)

    return s1_rmse_all, s2_rmse_all, s3_rmse_all, s4_rmse_all


def calculate_srm_diff(srm_list, srm_avg):
    srm_diff_all = []
    for srm in srm_list:
        # Euclidian distance between mean color and current color
        srm_diff = np.sqrt((srm[0, 1] - srm_avg[0, 1]) ** 2 +  # pond - pond avg, band 4
                           (srm[1, 1] - srm_avg[1, 1]) ** 2 +  # pond - pond avg, band 6
                           (srm[2, 1] - srm_avg[2, 1]) ** 2   # pond - pond avg, band 2
                          )

        # srm_diff = colorsys.rgb_to_hsv(srm[1, 1], srm[0, 1], srm[2, 1])[2] - colorsys.rgb_to_hsv(srm_avg[1, 1], srm_avg[0, 1], srm_avg[2, 1])[2]
        # srm_diff = np.sum(np.abs(np.subtract([srm[1, 1], srm[0, 1], srm[2, 1]], [srm_avg[1, 1], srm_avg[0, 1], srm_avg[2, 1]])))
        # srm_diff = np.average(np.square(np.subtract(srm, srm_avg)))
        # srm_diff = np.sum(np.abs(np.subtract(srm, srm_avg)))
        srm_diff_all.append(srm_diff)

    return srm_diff_all


def assert_defaults(srm):
    # If any of the categories in SRM are zeros, change that category to the default
    # Based on this default matrix:
    # srm = np.array([[0.050, 0.21, 0.64], [0.026, 0.05, 0.569], [0.07, 0.39, 0.691], [1, 1, 1]])
    srm = np.array(srm)
    # Default for Ocean
    if np.sum(srm[:, 0]) == 1:
        srm[:, 0] = [0.05, 0.026, 0.15, 1]
    # Default for Pond
    if np.sum(srm[:, 1]) == 1:
        srm[:, 1] = [0.21, 0.05, 0.39, 1]
    # Default for Ice
    if np.sum(srm[:, 2]) == 1:
        srm[:, 2] = [0.64, 0.569, 0.691, 1]

    return srm


def find_default(category, band):
    '''
    Returns the default reflectance value for a surface type / band combination
    '''
    if category == 'ice':
        ice_defaults = [0.8, 0.765, 0.70, 0.642, 0.5, 0.4, 0.3, 0.3]
        return ice_defaults[band]
    if category == 'pnd':
        pnd_defaults = [0.456, 0.431, 0.335, 0.197, 0.1, 0.065, 0.027, 0.027]
        return pnd_defaults[band]
    if category == 'ocn':
        ocn_defaults = [0.06, 0.06, 0.054, 0.065, 0.031, 0.029, 0.027, 0.027]
        return ocn_defaults[band]


def valid_block(image_data):
    '''
    Checks all image edges. If any are entirely zeros, skip the block
    '''

    # if np.sum(image_data[1,10,:]) == 0:
    #     return False
    # else:
    #     return True
    if image_data[1, 0, 0] == 0:
        return False
    elif image_data[1, -1, 0] == 0:
        return False
    elif image_data[1, 0, -1] == 0:
        return False
    elif image_data[1, -1, -1] == 0:
        return False
    else:
        return True


def check_read_size(y, block_size_y, y_dim):
    if y + block_size_y < y_dim:
        return block_size_y
    else:
        return y_dim - y


def write_results(dst_name, true_fraction,
                  s1_fraction, s1_error, s1_rmse,
                  s2_fraction, s2_error, s2_rmse,
                  s3_fraction, s3_error, s3_rmse,
                  s4_fraction, s4_error, s4_rmse,
                  pond_hsv, ocean_hsv, pond_ocean_diff, srm_diff_all, pratio_all):

    header = ["true_ice", "true_pond", "true_ocean",
              "s1_ice", "s1_pond", "s1_ocean", "s1_mce", "s1_rmse",
              "s2_ice", "s2_pond", "s2_ocean", "s2_mce", "s2_rmse",
              "s3_ice", "s3_pond", "s3_ocean", "s3_mce", "s3_rmse",
              "s4_ice", "s4_pond", "s4_ocean", "s4_mce", "s4_rmse",
              "pond_h", "pond_s", "pond_v",
              "ocean_h", "ocean_s", "ocean_v",
              "pond_ocean_diff", "pond_srm_diff", "pratio"]

    with open(os.path.join(dst_name), "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i in range(len(true_fraction)):
            writer.writerow([true_fraction[i][0], true_fraction[i][1], true_fraction[i][2],
                             s1_fraction[i][0], s1_fraction[i][1], s1_fraction[i][2], s1_error[i], s1_rmse[i],
                             s2_fraction[i][0], s2_fraction[i][1], s2_fraction[i][2], s2_error[i], s2_rmse[i],
                             s3_fraction[i][0], s3_fraction[i][1], s3_fraction[i][2], s3_error[i], s3_rmse[i],
                             s4_fraction[i][0], s4_fraction[i][1], s4_fraction[i][2], s4_error[i], s4_rmse[i],
                             pond_hsv[i][0], pond_hsv[i][1], pond_hsv[i][2],
                             ocean_hsv[i][0], ocean_hsv[i][1], ocean_hsv[i][2],
                             pond_ocean_diff[i], srm_diff_all[i], pratio_all[i]])

def write_to_tds(new_features, new_labels):

    fname = '/Volumes/research/active_projects/MODIS/unmixing/unmixing_data/wv_tds.hdf'
    if os.path.isfile(fname):
        features, labels = load_tds_file(fname)

        features = np.append(features, new_features, axis=0)
        labels = np.append(labels, new_labels, axis=0)

    else:
        features = np.array(new_features)
        labels = np.array(new_labels)

    tds_file = h5py.File(fname, 'w')
    tds_file.create_dataset("features", data=features)
    tds_file.create_dataset("labels", data=labels)
    tds_file.close()


def print_srm(srm):
    print("{0:0.2f} {1:0.2f} {2:0.2f}".format(*srm[0, :]))
    print("{0:0.2f} {1:0.2f} {2:0.2f}".format(*srm[1, :]))
    print("{0:0.2f} {1:0.2f} {2:0.2f}".format(*srm[2, :]))
    print("-" * 80)
    # print(srm)


def unmix_image(full_image_name, clsf_file, output_name):

    # Create the output folder if it does not exist
    output_folder = os.path.dirname(output_name)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # scale factor (f): 0.5m WV pixel * 1000 = 500m pseudo-MODIS
    f = 1000
    block_size = f*5        # size of each chunk to load into memory

    src_ds = gdal.Open(full_image_name, gdal.GA_ReadOnly)
    clsf_ds = gdal.Open(clsf_file, gdal.GA_ReadOnly)

    y_dim = src_ds.RasterYSize
    x_dim = src_ds.RasterXSize

    y_blocks = range(0, y_dim, block_size)
    x_blocks = range(0, x_dim, block_size)

    # Find the dark point reference
    src_dtype = gdal.GetDataTypeSize(src_ds.GetRasterBand(1).DataType)
    stretch_params = pp.histogram_threshold(src_ds, src_dtype)
    dark_ref = stretch_params[3]
    # Calculate the dark point and white point for a
    #       2pt correction based on the image histogram
    for b in range(len(dark_ref)):
        # Add a ceiling to the amount of correction per band (b)
        offset = ((b+1)**2)
        if dark_ref[b] > 65 - offset and offset < 62:
            dark_ref[b] = 65 - offset
    white_pt = stretch_params[1]
    # Add a const. ceiling to the white correction
    if white_pt < 235:
        white_pt = 235

    print(stretch_params)
    # We have to read the entire image ahead of time to calculate the whole image
    #   melt pond mean, which will be used in each block read below
    image_pmean, stage_two_srm = find_image_pmean(src_ds, clsf_ds, dark_ref, white_pt)

    # Create an RFC Model from existing training data
    model_filename = './rfc_model_worldview.p'
    with open(model_filename, 'rb') as mf:
        model = pickle.load(mf)

    # Flag for setting variables on first iteration
    initial = True
    pbar = tqdm(total=len(y_blocks)*len(x_blocks), unit='block')

    for y in y_blocks:
        for x in x_blocks:

            read_size_y = check_read_size(y, block_size, y_dim)
            read_size_x = check_read_size(x, block_size, x_dim)
            if read_size_y != block_size or read_size_x != block_size:
                pbar.update()
                continue

            optic_data = src_ds.ReadAsArray(x, y, read_size_x, read_size_y)
            clsf_data = clsf_ds.ReadAsArray(x, y, read_size_x, read_size_y)

            # Skip mostly empty blocks
            if not valid_block(optic_data):
                pbar.update()
                continue

            print("Rescaling bands...")
            # Apply the 2pt correction (simple histogram stretch)
            for b in range(7):
                optic_data[b, :, :] = pp.rescale_band(optic_data[b, :, :], dark_ref[b], white_pt)
            print("Done")

            # "Downsample" the classified image. This finds the percentage of each
            #   pseudo-MODIS pixel that is each surface type.
            ds_clsf_data = block_classification(clsf_data, f)

            # Analyze the imagery to get relevant data
            (srm_list, refl_list, pond_hsv, ocean_hsv,
             true_fractions, pond_ocean_diff, pratio) = analyze_imagery(optic_data,
                                                                        clsf_data, ds_clsf_data,
                                                                        f, pmean=image_pmean)

            # refl_list is a list of all pseudo-MODIS pixels [[b1, b2, ...], [b1, b2, ...], ...]
            # srm_list is a list of all the stage 1 reflectance matrices
            # Apply stage 1 unmixing
            s1_fract, s1_error = stage_one_unmixing(refl_list, srm_list, true_fractions)

            # Apply stage 2 unmixing
            s2_fract, s2_error = stage_two_unmixing(refl_list, stage_two_srm)

            # Unmix using a 'global' srm (generally Rosel's)
            global_srm_rosel = np.array([[.08, .16, .95], [.08, .07, .87], [.08, .22, .95], [1, 1, 1]])
            # Average of a bunch of WV images, calculated elsewhere
            global_srm_wv = np.array([[0.024, 0.201, 0.651], [0.024, 0.152, 0.557], [0.043, 0.279, 0.735], [1, 1, 1]])

            s3_fract, s3_error = stage_three_unmixing(refl_list, global_srm_wv)

            # Unmix with a machine learning method
            s4_fract, s4_error = ml_estimation(refl_list, model)

            # Find the difference between the pond color of each individual SRM and the average of all of them
            srm_diff = calculate_srm_diff(srm_list, stage_two_srm)
            srm_diff2 = calculate_srm_diff(srm_list, global_srm_rosel)
            srm_diff3 = calculate_srm_diff(srm_list, global_srm_wv)

            # write_to_tds(refl_list, true_fractions)

            if initial == True:
                initial = False
                # Initialize the data output lists
                true_fraction_all = true_fractions
                s1_fraction_all = s1_fract
                s1_error_all = s1_error
                s2_fraction_all = s2_fract
                s2_error_all = s2_error
                s3_fraction_all = s3_fract
                s3_error_all = s3_error
                s4_fraction_all = s4_fract
                s4_error_all = s4_error
                pond_hsv_all = pond_hsv
                ocean_hsv_all = ocean_hsv
                pond_ocean_diff_all = pond_ocean_diff
                srm_diff_all = srm_diff
                pratio_all = pratio

                srm_diff2_all = srm_diff2
                srm_diff3_all = srm_diff3
            else:
                # Append the new data to the master lists
                true_fraction_all = np.append(true_fraction_all, true_fractions, axis=0)
                s1_fraction_all = np.append(s1_fraction_all, s1_fract, axis=0)
                s1_error_all = np.append(s1_error_all, s1_error, axis=0)
                s2_fraction_all = np.append(s2_fraction_all, s2_fract, axis=0)
                s2_error_all = np.append(s2_error_all, s2_error, axis=0)
                s3_fraction_all = np.append(s3_fraction_all, s3_fract, axis=0)
                s3_error_all = np.append(s3_error_all, s3_error, axis=0)
                s4_fraction_all = np.append(s4_fraction_all, s4_fract, axis=0)
                s4_error_all = np.append(s4_error_all, s4_error, axis=0)
                pond_hsv_all = np.append(pond_hsv_all, pond_hsv, axis=0)
                ocean_hsv_all = np.append(ocean_hsv_all, ocean_hsv, axis=0)
                pond_ocean_diff_all = np.append(pond_ocean_diff_all, pond_ocean_diff, axis=0)
                srm_diff_all = np.append(srm_diff_all, srm_diff, axis=0)
                pratio_all = np.append(pratio_all, pratio, axis=0)
                srm_diff2_all = np.append(srm_diff2_all, srm_diff2, axis=0)
                srm_diff3_all = np.append(srm_diff3_all, srm_diff3, axis=0)

            pbar.update()

    pbar.close()

    src_ds = None
    clsf_ds = None

    s1_rmse, s2_rmse, s3_rmse, s4_rmse = calculate_rmse(true_fraction_all, s1_fraction_all,
                                                        s2_fraction_all, s3_fraction_all, s4_fraction_all)

    write_results(output_name, true_fraction_all,
                  s1_fraction_all, s1_error_all, s1_rmse,
                  s2_fraction_all, s2_error_all, s2_rmse,
                  s3_fraction_all, s3_error_all, s3_rmse,
                  s4_fraction_all, s4_error_all, s4_rmse,
                  pond_hsv_all, ocean_hsv_all, pond_ocean_diff_all, srm_diff_all, pratio_all)

    print("Stage 1 RMSE: {}".format(np.average(s1_rmse)))
    print("Stage 2 RMSE: {}".format(np.average(s2_rmse)))
    print("Stage 3 RMSE: {}".format(np.average(s3_rmse)))
    print("Stage 4 RMSE: {}".format(np.average(s4_rmse)))

    print("Pond differences:")
    print("Diff from image mean")
    print("{0:0.4f}, {1:0.4f}".format(np.mean(srm_diff_all), np.std(srm_diff_all)))
    print("Diff from Rosel")
    print("{0:0.4f}, {1:0.4f}".format(np.mean(srm_diff2_all), np.std(srm_diff2_all)))
    print("Diff from WV global")
    print("{0:0.4f}, {1:0.4f}".format(np.mean(srm_diff3_all), np.std(srm_diff3_all)))


def main():
    # Set Up Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_im",
                        help='''file path of raw image''')
    parser.add_argument("clsf_im",
                        help='''file path of classified image''')
    parser.add_argument("dst_dir",
                        help="directory to write output files")

    # Parse Arguments
    args = parser.parse_args()

    # System filepath that contains the directories or files for batch processing
    full_image_name = args.raw_im
    clsf_file = args.clsf_im
    dst_dir = args.dst_dir

    # Run the unmixing process given the input raw/clsf pair
    unmix_image(full_image_name, clsf_file, dst_dir)


if __name__ == '__main__':
    main()
