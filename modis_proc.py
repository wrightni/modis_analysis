### Calculates the melt pond fraction from MODIS imagery using the algorithm
###  developed by Rosel (and implemented in rosel_alg.py)
## Nick Wright
# 9/18/17

import pickle
import csv
import os
import sys
import datetime
import calendar
import numpy as np
import gdal

from sklearn.ensemble import RandomForestRegressor

# Custom Functions
from lib import rosel_alg
from lib import alter_image as ai
from lib import modis_dl_reproj as mdr


# Reads the modis file given by a filename and returns a numpy array with 
# the raster data. Need to read band 1, 2, and 3 for the algorithm,
# and maybe the cloud mask band for later use. 
#  Mask is a dataset in the shape of the other bands where values of 1
#  are masked in the output dataset.
def read_modis(modis_band_list, mask):
    
    # Make a list of the input filenames
    modis_filenames = modis_band_list #[:3]
    num_bands = len(modis_band_list)

    # Cycle through the list of inputs and read each file into an array
    band_collection = []
    for file in modis_filenames:
        print(file)
        # Open the image with gdal
        dataset = gdal.Open(file)

        # Get the geotransformation
        geotransform = list(dataset.GetGeoTransform())
        proj = dataset.GetProjection()

        # Read the data as a raster band and convert to a numpy array
        #  No data pixels (-28672) will get set to 36864. This is ok, as 
        #  everything above 10000 will be ignored in further processing.
        band = dataset.GetRasterBand(1)
        band = np.array(band.ReadAsArray(),dtype=np.int16)
        # Zeros are invalid data points
        band[band==0] = 65535
        band[band<1] = 1
        band_collection.append(band)

        # Clear the gdal datasets out of memory
        dataset = None
        band = None

    # Combine the bands read above into a single 3-dim stack
    # Layer 0 = red(b4), layer 1 = green(b3) layer 2 = blue(b1)
    img_dim = band_collection[0].shape
    modis_img = np.zeros((img_dim[0], img_dim[1], num_bands), dtype=np.uint16)
    for i in range(num_bands):
        modis_img[:,:,i] = band_collection[i]

    if mask == None:
        return modis_img, geotransform, proj
    else:
        mask_dim = np.shape(mask)

        # Delete or pad mask to be the same size as the modis image
        #  We dont particularly care about the edges of the image anyways,
        #  since the areas of interest are in the middle.
        if mask_dim[0] > img_dim[0]:
            mask = mask[0:img_dim[0],:]
        else:
            modis_img = modis_img[0:mask_dim[0],:,:]

        if mask_dim[1] > img_dim[1]:
            mask = mask[:,0:img_dim[1]]
        else:
            modis_img = modis_img[:,0:mask_dim[1],:]

        # Set the mask values to the no data value
        modis_img[mask==True] = 65535

        return modis_img, geotransform, proj


# Reads the modis QC band and parses out the cloud and land mask values.
# Returns a boolean dataset in the shape of a Modis image where 1 is a 
# pixel that is either a cloud or land, and 0 is not. 
def create_cloudland_mask(modis_state_band, latlon=None):
    
    # Load the modis dataset containing the cloud and land flags.
    dataset = gdal.Open(modis_state_band)

    band = dataset.GetRasterBand(1)
    band = np.array(band.ReadAsArray())

    # Initialize the output mask as a binary image (based on 
    # dimensions of the input band.
    if latlon is None:
        dims = np.shape(band)
        cl_mask = np.zeros((dims[0] * 2, dims[1] * 2), dtype=bool)
    else:
        geotransform = list(dataset.GetGeoTransform())
        proj = dataset.GetProjection()
        x, y = ai.coord_to_index(latlon[0], latlon[1], geotransform, dst_proj=proj)
        ulx = int(x-50)
        uly = int(y-50)
        lrx = int(x+50)
        lry = int(y+50)
        band = band[uly:lry, ulx:lrx]

        dims = np.shape(band)
        cl_mask = np.zeros((dims[0] * 2, dims[1] * 2), dtype=bool)

    dataset = None

    # Read the binary of every pixel in the state band
    for x in range(dims[0]):
        for y in range(dims[1]):

            # Skip no value pixels
            if band[x,y] == -28672:
                continue
            if band[x,y] == 32767:
                continue
            if band[x,y] == 0:
                continue

            # Read the binary value of the pixel
            binary = "{0:016b}".format(band[x,y])

            # We're going to go ahead and upscale the image
            # by a factor of 2, since the quality band is in 1kmx1km
            # resolution, but we want 500x500m. 
            # Variable names: x_left, x_right, y_left, y_right
            xl = x*2
            xr = (x+1)*2
            yl = y*2
            yr = (y+1)*2

            # Parse the binary value
            # Cloud or partially cloudy flag
            if binary[14:16] == '01': # or binary[14:16] == '10':
                cl_mask[xl:xr,yl:yr] = True
            # Cloud shadow flag
            elif binary[13] == '1':
                cl_mask[xl:xr,yl:yr] = True
            # Land flag
            elif binary[10:13] == '001':
                cl_mask[xl:xr,yl:yr] = True

    # Save the mask as a tif for testing / inspection
    # driver = gdal.GetDriverByName('GTiff')
    # dst_dataset = driver.Create('/Volumes/research/MODIS/MOD09GA/2014.06.13/cl_mask.tif', dims[1]*2, dims[0]*2, 1, gdal.GDT_Byte)
    # dst_dataset.GetRasterBand(1).WriteArray(cl_mask)
    # dst_dataset = None

    return cl_mask

# Runs the optimization algorithm on a single image block
#   Using the unmixing algorithm
def process_block_unm(image, srm, noise=0):

    y_dim, x_dim, nbands = np.shape(image)
    # modis 09ga scale factor: 0.0001.
    scale_factor = 0.0001

    output_image = np.zeros((y_dim, x_dim, 3))

    obs_reshape = np.reshape(image, (-1,7))
    obs_reshape = obs_reshape * scale_factor

    if noise != 0:
        for i in range(len(obs_reshape)):
            obs_reshape[i] = np.random.normal(obs_reshape[i], noise)

    print(np.shape(obs_reshape))
    print("Predicting values...")
    results = np.zeros((len(obs_reshape),3))
    for i in range(len(obs_reshape)):
        refl = [obs_reshape[i, 0], obs_reshape[i, 2], obs_reshape[i, 3]]
        results[i] = rosel_alg.spec_unmix(refl, 1, srm)

    print("Done...")

    # output_image = np.reshape(results, (y_dim, x_dim, 3))

    i = 0
    for y in range(y_dim):
        for x in range(x_dim):
            output_image[y, x, 0] = results[i, 0]
            output_image[y, x, 1] = results[i, 1]
            output_image[y, x, 2] = results[i, 2]
            i += 1

    return output_image

# Runs the optimization algorithm on a single image block
#   Using the random forest classifier
def process_block_rfc(image, model, noise=0.0):

    y_dim, x_dim, nbands = np.shape(image)
    # modis 09ga scale factor: 0.0001.
    scale_factor = 0.0001

    output_image = np.zeros((y_dim, x_dim, 3))

    obs_reshape = np.reshape(image, (-1, nbands))
    obs_reshape = obs_reshape * scale_factor

    if noise != 0:
        for i in range(len(obs_reshape)):
            obs_reshape[i] = np.random.normal(obs_reshape[i], noise)

    print(np.shape(obs_reshape))
    print("Predicting values...")
    model.n_jobs = 7
    results = model.predict(np.array(obs_reshape))
    print("Done...")

    # output_image = np.reshape(results, (y_dim, x_dim, 3))

    i = 0
    for y in range(y_dim):
        for x in range(x_dim):
            output_image[y, x, 0] = results[i, 2]
            output_image[y, x, 1] = results[i, 1]
            output_image[y, x, 2] = results[i, 0]
            i += 1

    return output_image


# Converts the reflectance data returned by read_modis() into areal
# fraction using the rosel algorithm on each pixel.
# Returns a three band geotiff where
#   layer 1: Area of water
#   layer 2: Area of melt ponds
#   layer 3: Area of snow and ice
def convert_to_areal(modis_image, geotransform, proj,
                     method=process_block_unm, model=None, latlon=None):
    '''
    If method is umn, model is a spectral reflectance matrix (srm)
    If method is rfc, model is a random forest model
    '''

    # If a specifc coordinate is given, only process an area around that point.
    # Otherwise process the whole image
    if latlon is None:
        block = modis_image
        gt_new = None
    else:
        x, y = ai.coord_to_index(latlon[0],latlon[1],geotransform,dst_proj=proj)
        ulx = int(x-latlon[2])
        uly = int(y-latlon[2])
        lrx = int(x+latlon[2] + 1)   #Add 1 for python indexing
        lry = int(y+latlon[2] + 1)
        block = modis_image[uly:lry, ulx:lrx, :]
        # New geotransform for the crop (shifted corner coord)
        print(geotransform)
        gt_new = [geotransform[0] + ulx*geotransform[1], geotransform[1], 0,
                  geotransform[3] + uly*geotransform[5], 0, geotransform[5]]
        print(gt_new)

    # Clear the modis image from memory
    modis_image = None

    output_im = method(block, model)

    ## Add some noise
    stdev = 0.05
    block_list = []
    for _ in range(1):
        block_noise = method(block, model, noise=stdev)
        block_list.append(block_noise)

    block_noise = np.std(block_list, axis=0)

    return output_im, block, block_noise, gt_new


# Checks that the given pixel is within the valid data range
def is_valid_pixel(pixel):
    if (pixel[0] > 10000 or 
        pixel[1] > 10000 or
        pixel[2] > 10000):
        return False
    elif (pixel[0] == 0 or 
        pixel[1] == 0 or
        pixel[2] == 0):
        return False
    else:
        return True


# Checks if the given pixel is land
def is_land_pixel(pixel):
    # Flag land mask points
    if (pixel[0] == 216 and
        pixel[1] == 216 and
        pixel[2] == 216):
        return True
    # Flag coastal points
    elif (pixel[0] == 3 and
        pixel[1] == 3 and
        pixel[2] == 3):
        return True
    else:
        return False


# Saves a three band geotiff
def save_geotiff(filename, raster_data):

    out_dims = np.shape(raster_data)
    # Save the mask as a tif for testing / inspection
    driver = gdal.GetDriverByName('GTiff')
    dst_dataset = driver.Create(filename, out_dims[1], out_dims[0], out_dims[2], gdal.GDT_Float32,
                                options=["TILED=YES", "COMPRESS=LZW"])

    for b in range(out_dims[2]):
        dst_dataset.GetRasterBand(b+1).WriteArray(raster_data[:,:,b])

    dst_dataset.FlushCache()
    dst_dataset = None


def write_to_csv(output_file, output_data):

    if not os.path.isfile(output_file):
        with open(output_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image", "date", "hrsnow", "msnow", "hrmp", "mmp", "hrow", "mow", "rmse", "mc_error",
                             "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
            writer.writerow(output_data)

    else:
        with open(output_file, "a+") as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow([output_data[0], output_data[1], output_data[2], output_data[3],
            #                  output_data[4], output_data[5], output_data[6], output_data[7],
            #                  output_data[8]])
            writer.writerow(output_data)


# Asks the user a yes or no question
# Adapted from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
# http://code.activestate.com/recipes/577058/
def query_yes_no(question):
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    while True:
        sys.stdout.write(question)
        choice = raw_input().lower()
        if choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'"
                             "(or 'y' or 'n').\n")


def drift_correct(fraction_image, raw_image, true_fraction, window_size, var_image=None):

    snow, mp, ow = true_fraction

    ## Pseudo drift correction. Find the best window within the given area.
    x_dim, y_dim, z = np.shape(fraction_image)
    x_blocks = range(0, x_dim)
    y_blocks = range(0, y_dim)
    rmse = 10
    ulcoord = [0, 0]
    for x in x_blocks:
        for y in y_blocks:
            if x + window_size > x_dim or y + window_size > y_dim:
                continue
            window = fraction_image[x:x + window_size, y:y + window_size, :]
            fract_avg = [np.nanmean(window[:, :, 0]), #/1000,
                         np.nanmean(window[:, :, 1]), #/1000,
                         np.nanmean(window[:, :, 2])] #/1000]
            rmse_i = np.sqrt((snow-fract_avg[2])**2 + (mp-fract_avg[1])**2 + (ow-fract_avg[0])**2)
            if rmse_i < rmse:
                rmse = rmse_i
                best_window = fract_avg
                ulcoord = [y, x]

                if var_image is not None:
                    window_var = var_image[x:x + window_size, y:y + window_size, :]
                    var_average = np.nanmean(window_var)
                else:
                    var_average = 0

                cloud_prcnt = 0
                raw_avg = []
                for b in range(np.shape(raw_image)[2]):
                    raw_avg.append(np.average(raw_image[x:x + window_size, y:y + window_size, b] / 10000.))

    return rmse, best_window, raw_avg, var_average, ulcoord


def window_average(fraction_image, raw_image, true_fraction, var_image=None):

    snow, mp, ow = true_fraction

    # Mask out clouds (high band 7 reflectance)
    fraction_image[raw_image[:, :, 6] > 600] = np.nan

    fract_avg = [np.nanmean(fraction_image[:, :, 0]), #/1000,
                 np.nanmean(fraction_image[:, :, 1]), #/1000,
                 np.nanmean(fraction_image[:, :, 2])] #/1000]

    rmse = np.sqrt((snow-fract_avg[2])**2 + (mp-fract_avg[1])**2 + (ow-fract_avg[0])**2)

    raw_avg = []
    for b in range(np.shape(raw_image)[2]):
        raw_avg.append(np.average(raw_image[:, :, b] / 10000.))

    return rmse, fract_avg, raw_avg


def construct_filestrings(date, dst_name, product='MYD09GA'):
    '''
    Creates the necessary filenames and paths
    :param date: MODIS image date (mmm dd yyyy)
    :param dst_name: name to append to modis filenames for output file
    :return:
    '''
    day = int(date.split()[1])
    month = int(list(calendar.month_abbr).index(date.split()[0]))
    year = int(date.split()[2])

    d = datetime.date(int(year), int(month), int(day))
    julianday = d.toordinal() - datetime.date(d.year, 1, 1).toordinal() + 1

    date = '{0:04d}.{1:02d}.{2:02d}'.format(d.year, d.month, d.day)

    rel_path = 'MODIS/{}/{}'.format(product, date)
    path = os.path.join(os.getcwd(), rel_path)
    # path = os.path.join("/Users/nicholas/mnt/daedalus/sequoia/", rel_path)


    band_string = "{0:}.A{1:04d}{2:03d}_MODIS_Grid_500m_2D_sur_refl_b0{3:}_1.tif"
    state_string = "{0:}.A{1:04d}{2:03d}_MODIS_Grid_1km_2D_state_1km_1.tif"
    dst_string = "{0:}.A{1:04d}{2:03d}_MODIS_Grid_500m_2D_{3:}.tif"

    b_list = []
    for b in range(1, 7):
        b_list.append(os.path.join(path, band_string.format(product, d.year, julianday, b)))

    # b1 = os.path.join(path, band_string.format(d.year, julianday, '1'))
    # b2 = os.path.join(path, band_string.format(d.year, julianday, '2'))
    # b3 = os.path.join(path, band_string.format(d.year, julianday, '3'))
    state = os.path.join(path, state_string.format(product, d.year, julianday))
    dst_filename = os.path.join(path, dst_string.format(product, d.year, julianday, dst_name))

    return path, b_list, state, dst_filename, julianday, date



def process_whole_im(date, dst_name, mask=None):

    product = 'MOD09GA'
    path, b_list, state, dst_filename, julianday, date = construct_filestrings(date, dst_name, product=product)

    # Check if the needed image is present.
    # Download and reproject if it is not.
    if not os.path.isdir(path) or not os.path.isfile(b_list[5]):
        mdr.modis_dl_reproject(product, '006', date, os.getcwd(), needy=False)

    # calculate areal fractions from modis image
    if mask:
        print("Creating cloud and land mask...")
        cl_mask = create_cloudland_mask(state)
        im, geotransform, proj = read_modis(b_list, cl_mask)
    else:
        im, geotransform, proj = read_modis(b_list, None)

    # Create an RFC Model
    model_filename = os.path.join(os.getcwd(), 'rfc_model.p')
    with open(model_filename, 'rb') as mf:
        model = pickle.load(mf)

    fract, raw_image, new_lut, gt_new = convert_to_areal(im, {}, geotransform, proj, model=model)

    save_geotiff(dst_filename, fract)
    ai.add_spatial_info(dst_filename, gt_new)

    # save_geotiff(dst_filename, cl_mask)
    # ai.add_spatial_info(dst_filename, geotransform)


def calc_mod_mpf(mask=False):
    '''

    :param task_list: List of coordinates and true values to compare with modis
    :param search_size: Size of MODIS window to process
    :param image_size: Size of area to select from the search window for final results
                        if 'auto' selects based on 'area' parameter from task_list
    :return:
    '''
    # USER SUPPLIED PARAMETERS:
    lat = 78.4
    lon = 113.4
    date = "Jul 12 2015"   #(mmm dd yyyy)

    product = 'MYD09GA' # OR 'MOD09GA'

    latlon = [lat, lon]

    # Create an RFC Model (default filename created with create_rfc_model)
    model_filename = os.path.join(os.getcwd(), 'rfc_8cat.p')
    with open(model_filename, 'rb') as mf:
        model = pickle.load(mf)

    dst_name = "output"

    path, b_list, state, dst_filename, julianday, date = construct_filestrings(date, dst_name, product=product)

    # Check if the needed image is present.
    # Download and reproject if it is not. REQUIRES USERNAME AND PASSWORD
    if not os.path.isdir(path) or not os.path.isfile(b_list[5]):
        mdr.modis_dl_reproject(product, '006', date,os.getcwd(), needy=False)

    # Load the cloud mask if requested
    if mask:
        print("Creating cloud and land mask...")
        cl_mask = create_cloudland_mask(state, latlon=latlon)
        im, geotransform, proj = read_modis(b_list, cl_mask)
    else:
        im, geotransform, proj = read_modis(b_list, None)
        cloud_prcnt = 0

    # calculate areal fractions from modis image
    search_size = 500
    latlon.append(search_size)

    # choose one of these to select between the random forest and spectral unmixing
    fract, raw_image, var_image, gt_new = convert_to_areal(im, geotransform, proj,
                                                   method=process_block_rfc, model=model, latlon=latlon)
    # fract, raw_image, var_image, gt_new = convert_to_areal(im, geotransform, proj,
    #                                                method=process_block_unm, latlon=latlon)


    # Save the raw data for the specific coordinates
    save_geotiff(dst_filename, raw_image)
    # Add spatial information to the newly created file.
    ai.add_spatial_info(dst_filename, gt_new)

    # Save the fraction data
    dst_filename = os.path.splitext(dst_filename)[0] + '_area_frac.tif'
    save_geotiff(dst_filename, fract)
    ai.add_spatial_info(dst_filename, gt_new)

    quit()
    # To drift correct:
    ## Drift correct needs a "true" surface fraction to compare
    ## e.g. true_surface_fraction = [snow, mp, ow]

    # Parameters from the WV image:
    snow = 0.5
    mp = 0.3
    ow = 0.2
    w_size = 20     #in kilometers (one side of a WV image)

    true_surface_fraction = [snow, mp, ow]
    rmse, best_window, raw_avg, var_average, ulxy = drift_correct(fract, raw_image, true_surface_fraction, w_size, var_image)

    print("Area: {} | Window: {}".format(np.shape(fract), w_size))

    # Mask the cloud values
    if mask:
        fract[cl_mask == True] = np.nan

    print("HRI: snow = {0:0.3f}, mp = {1:0.3f}, ow = {2:0.3f}".format(snow, mp, ow))
    print("MOD: snow = {0:0.3f}, mp = {1:0.3f}, ow = {2:0.3f}".format(best_window[2], best_window[1], best_window[0]))
    print("RMS: {0:03f}".format(rmse))
    print("MCE: {0:03f}".format(var_average))
    # print("Cloud percent: {0:0.2f}".format(cloud_prcnt))

    output_data = [dst_name, date, snow, best_window[2], mp, best_window[1], ow, best_window[0], rmse, var_average,
                   raw_avg[0], raw_avg[1], raw_avg[2], raw_avg[3],
                   raw_avg[4], raw_avg[5]]
    output_csv = "WV_MODIS.csv"
    write_to_csv(output_csv, output_data)



def main():
    calc_mod_mpf()
    # process_whole_im('Jul 21 2016', "areal_frac_IBT", mask=False)

if __name__ == '__main__':
    main()
