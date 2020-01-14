import os
import subprocess

import gdal
from osgeo import osr

def add_spatial_info(file, geotransform, epsg_proj=3413):

    gt = _check_geotransform(geotransform)

    # Open file with writing privledges
    src_dataset = gdal.Open(file, gdal.GA_Update)

    # Set the projection
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg_proj)
    src_dataset.SetProjection(dst.ExportToWkt())

    # Set the GeoTransform
    src_dataset.SetGeoTransform(gt)

    # Close the dataset (and write changes)
    src_dataset = None


# Apply the land mask to the surface reflectane bands
def apply_mask(src_image, mask):

    dst_image = os.path.splitext(src_image)[0] + '_masked.tif'

    # output_size = '-3500000 3500000 -3500000 3500000' # in meters from center
    # cmd = 'gdal_merge.py -n 0 -ul_lr {} -o {} {} {}'.format(output_size, dst_image, src_image, mask)
    cmd = 'gdal_merge.py -n 0 -o {} {} {}'.format(dst_image, src_image, mask)

    print(cmd)

    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.read()
    print(output)
    p.wait()


## Converts the given latitude and longitude pair into the corresponding x-y
#   indicies in a modis image.
def coord_to_index(lat, lon, geotransform, dst_proj=False, dst_epsg=3413):
    gt = _check_geotransform(geotransform)

    # Set the lat/lon source projection
    src = osr.SpatialReference()
    src.SetWellKnownGeogCS("WGS84")

    # Set the projection
    dst = osr.SpatialReference()
    if dst_proj:
        dst.ImportFromWkt(dst_proj)
        # dst.SetProjection(dst_proj)
    else:
        dst.ImportFromEPSG(dst_epsg)

    # Create the transformation between projections
    ct = osr.CoordinateTransformation(src,dst)

    # gt = padfTransform
    #Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2];
    #Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5];

    xy = ct.TransformPoint(lon, lat)

    x = (xy[0] - gt[0]) / gt[1]
    y = (xy[1] - gt[3]) / gt[5]

    return x, y


## Converts the given latitude and longitude pair into the corresponding x-y
#   indicies in a modis image.
def index_to_coord(x, y, geotransform, src_proj=False, src_epsg=3413):
    gt = _check_geotransform(geotransform)

    # Set the lat/lon source projection
    dst = osr.SpatialReference()
    dst.SetWellKnownGeogCS("WGS84")

    # Set the projection
    src = osr.SpatialReference()
    if src_proj:
        src.ImportFromWkt(src_proj)
        # dst.SetProjection(dst_proj)
    else:
        src.ImportFromEPSG(src_epsg)

    # Create the transformation between projections
    ct = osr.CoordinateTransformation(src, dst)

    # gt = padfTransform
    #Xgeo = GT(0) + Xpixel * GT(1) + Yline * GT(2)
    #Ygeo = GT(3) + Xpixel * GT(5) + Yline * GT(4)

    a = gt[0] + x*gt[1] + y*gt[2]
    b = gt[3] + y*gt[5] + x*gt[4]

    lon, lat, h = ct.TransformPoint(a, b)

    return lat, lon


# Verify geotransform input
def _check_geotransform(geotransform):
    
    # Not really making sure the input is the correct format, but oh well. 
    if isinstance(geotransform,list):
        gt = geotransform
    elif geotransform == 'nsidcmask':
        # GeoTransform for NSIDC land mask dataset
        gt = [-3850000.0, 6250.0, 0.0, 5850000, 0.0, -6250.0]
    elif geotransform == 'modiscustom':
        # GeoTransform for MODIS Datasets (output of stitching)
        gt = [-3323160.27126498, 500.0, 0.0, 2699429.7924969867, 0.0, -500.0]
    else:
        gt = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    return gt


def main():

    print(coord_to_index(90.,180.,None))
    
    # mask = '/Users/nicholas/Documents/Dartmouth/Projects/MWCProject/landmask_3d.tif'

    # src_file = '/Users/nicholas/Documents/Dartmouth/Projects/MWCProject/results/2008.06.25/MOD09GA.A2008177_MODIS_Grid_500m_2D_areal_frac.tif'

    # add_spatial_info(src_file, 'modiscustom')

    # apply_mask(src_file,mask)

if __name__ == '__main__':
    main()