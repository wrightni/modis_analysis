import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import h5py
import os
from tqdm import tqdm


def main():

    root_dir = os.getcwd()

    albedo_table = load_albedo(root_dir, 'albedo.csv')

    # Uncomment the function that you want to run
    eight_categories(root_dir, albedo_table)
    # three_categories_random(root_dir, albedo_table)
    # three_categories_determine(root_dir, albedo_table)


def eight_categories(write_dir, albedo_table):
    num_obs = 64000
    surf_frac = sample_surf_dist(num_obs, 3)

    # Remove all generated fractions that have an absolute or relative melt pond fraction
    # higher than 60%
    s = 0
    while s < len(surf_frac):
        if surf_frac[s][1] > .65:
            surf_frac = np.delete(surf_frac, s, 0)
        elif surf_frac[s][1] / (surf_frac[s][1] + surf_frac[s][0]) > 0.65:
            surf_frac = np.delete(surf_frac, s, 0)
        else:
            s += 1

    num_obs = len(surf_frac)

    obs_refl = np.zeros((num_obs, 6))       # 6 MODIS BANDS USED

    for i in tqdm(range(num_obs)):

        fi, fp, fo = surf_frac[i]
        # Find the subcategory distributions
        sub_frac_ice, sub_frac_pond = calc_subfrac_8cat()

        for b in range(6):          # 7 MODIS 8 WV
            # Find the three category reflectances using the subcategory distributions
            ice_refl, pond_refl, ocean_refl = parse_8cat(b, albedo_table, sub_frac_ice, sub_frac_pond)
            obs_refl[i, b] = ((ice_refl * fi) + (pond_refl * fp) + (ocean_refl * fo))

    tds_name = os.path.join(write_dir, 'modis_artificial_tds_8cat.hdf')
    write_hdf(tds_name, obs_refl, surf_frac)


def parse_8cat(b, albedo_table, sub_frac_ice, sub_frac_pond):
    # Sample possible subsurfaces to find ice reflectance
    ice_refl = 0
    ice_refl += (albedo_table[b, 0] * sub_frac_ice[0])  # Cold Snow
    ice_refl += (albedo_table[b, 1] * sub_frac_ice[1])  # Melting Snow
    ice_refl += (albedo_table[b, 2] * sub_frac_ice[2])  # Deter. Melting Ice
    ice_refl += (albedo_table[b, 3] * sub_frac_ice[3])  # Dirty Ice

    # Sample possible subsurfaces to find pond reflectance
    pond_refl = 0
    pond_refl += (albedo_table[b, 4] * sub_frac_pond[0])  # Bright ponds
    pond_refl += (albedo_table[b, 5] * sub_frac_pond[1])  # Early Ponds
    pond_refl += (albedo_table[b, 6] * sub_frac_pond[2])  # Late Ponds

    # Ocean has no subsurfaces
    ocean_refl = albedo_table[b, 7]

    return ice_refl, pond_refl, ocean_refl


def calc_subfrac_8cat():
    # Subsurface distribution
    sub_frac_ice = sample_surf_dist(1, 4)[0]
    # Limit the amount of dirty or deteriorating ice.
    while sub_frac_ice[3] > 0.1 or sub_frac_ice[2] > 0.2:
        sub_frac_ice = sample_surf_dist(1, 4)[0]
    sub_frac_pond = sample_surf_dist(1, 3)[0]

    return sub_frac_ice, sub_frac_pond


def three_categories_random(write_dir, albedo_table):
    num_obs = 64000
    surf_frac = sample_surf_dist(num_obs, 3)

    # Remove all generated fractions that have an absolute or relative melt pond fraction
    # higher than 60%
    s = 0
    while s < len(surf_frac):
        if surf_frac[s][1] > .65:
            surf_frac = np.delete(surf_frac, s, 0)
        elif surf_frac[s][1] / (surf_frac[s][1] + surf_frac[s][0]) > 0.65:
            surf_frac = np.delete(surf_frac, s, 0)
        else:
            s += 1

    num_obs = len(surf_frac)

    obs_refl = np.zeros((num_obs, 8)) ##6 for MODIS

    for i in tqdm(range(num_obs)):

        fi, fp, fo = surf_frac[i]

        for b in range(6):
            # Find the three category reflectances using the subcategory distributions
            ice_refl, pond_refl, ocean_refl = parse_3cat(b, albedo_table)
            obs_refl[i, b] = ((ice_refl * fi) + (pond_refl * fp) + (ocean_refl * fo))

    tds_name = os.path.join(write_dir, 'modis_artificial_tds_3cat_random.hdf')
    write_hdf(tds_name, obs_refl, surf_frac)


def three_categories_determine(write_dir, albedo_table):
    '''
    Creates the training dataset with three surface categories
    (using all possible fraction combinations)
    :param write_dir:
    :return:
    '''
    surf_frac = exhaustive_3cat_dist()
    num_obs = len(surf_frac)

    obs_refl = np.zeros((num_obs, 6))

    for i in tqdm(range(num_obs)):
        fi, flp, fo = surf_frac[i]
        for b in range(6):
            ir, lpr, ocr = parse_3cat(b, albedo_table)
            obs_refl[i, b] = ((ir * fi) + (lpr * flp) + (ocr * fo))

    tds_name = os.path.join(write_dir, 'modis_artificial_tds_3cat_determine.hdf')
    write_hdf(tds_name, obs_refl, surf_frac)


# All possibilities with 3 surfaces
def exhaustive_3cat_dist():

    surf_frac = []
    for i in range(101):
        for m in range(101-i):
            if (m) / float(m+i+0.1) > 0.65:
                continue
            o = 100 - (i + m)
            surf_frac.append([i, m, o])

    surf_frac = np.divide(surf_frac, 100)
    print(np.shape(surf_frac))

    return surf_frac


def parse_3cat(b, albedo_table):
    ice_refl = albedo_table[b, 0]  # Cold Snow
    pond_refl = albedo_table[b, 5]  # Early Ponds
    ocean_refl = albedo_table[b, 7]

    return ice_refl, pond_refl, ocean_refl



def calc_subfrac_v1():
    # Subsurface distribution
    sub_frac_ice = sample_surf_dist(1, 5)[0]
    while sub_frac_ice[4] > 0.3 or sub_frac_ice[3] > 0.2:
        sub_frac_ice = sample_surf_dist(1, 5)[0]
    sub_frac_pond = sample_surf_dist(1, 4)[0]

    return sub_frac_ice, sub_frac_pond

def parse_v1(b, albedo_table, sub_frac_ice, sub_frac_pond):
    # Sample possible subsurfaces to find ice reflectance
    ice_refl = 0
    ice_refl += (albedo_table[b, 0] * sub_frac_ice[0])  # Cold Snow
    ice_refl += (albedo_table[b, 1] * sub_frac_ice[1])  # Melting Snow
    ice_refl += (albedo_table[b, 2] * sub_frac_ice[2])  # Deter. Melting Ice
    ice_refl += (albedo_table[b, 3] * sub_frac_ice[3])  # Undetr Melting Ice
    ice_refl += (albedo_table[b, 7] * sub_frac_ice[4])  # Dirty Ice

    # Sample possible subsurfaces to find pond reflectance
    pond_refl = 0
    pond_refl += (albedo_table[b, 4] * sub_frac_pond[0])  # B-G ice
    pond_refl += (albedo_table[b, 5] * sub_frac_pond[1])  # EMP
    pond_refl += (albedo_table[b, 6] * sub_frac_pond[2])  # LMP
    # Add some more darkness to the pond options
    pond_refl += (albedo_table[b, 8] * sub_frac_pond[3]) * 2.  # Ocean

    # Ocean has no subsurfaces
    ocean_refl = albedo_table[b, 8]

    return ice_refl, pond_refl, ocean_refl

# All possibilities with 4 surfaces
def sample_surf_dist_v4():

    surf_frac = []
    for i in range(101):
        for lm in range(101-i):
            for dm in range(101-lm-i):
                if (lm+dm) / float(lm+dm+i+0.1) > 0.65:
                    continue
                o = 100 - (i + lm + dm)
                surf_frac.append([i, lm, dm, o])

    surf_frac = np.divide(surf_frac, 100)
    print(np.shape(surf_frac))
    for i in range(2400,2410):
        print(surf_frac[i], sum(surf_frac[i]))

    return surf_frac



def parse_v4(b, albedo_table):
    # Sample possible subsurfaces to find ice reflectance
    ice_refl = (albedo_table[b, 1])  # Melting Snow

    # Sample possible subsurfaces to find pond reflectance
    lpond_refl = (albedo_table[b, 4])  # Light Ponds

    dpond_refl = (albedo_table[b, 7])

    # Ocean has no subsurfaces
    ocean_refl = albedo_table[b, 8]

    return ice_refl, lpond_refl, dpond_refl, ocean_refl


def sample_surf_dist(num_obs, num_surf):
    '''
    Establishes the sample surface distribution
    :param num_obs: Number of sample observations
    :return: List of randomly generated surface distributions
    '''
    surf_frac = np.zeros((num_obs, num_surf))
    for i in range(num_obs):
        rlist = [0 for s in range(num_surf + 1)]
        rlist[0] = 0
        rlist[-1] = 1000

        # Establish the sampled surface distribution
        for s in range(num_surf - 1):
            rlist[s+1] = random.randint(0, 1000)

        rlist.sort()

        for s in range(num_surf - 1):
            surf_frac[i, s] = (rlist[s + 2] - rlist[s + 1]) / 1000

        surf_frac[i, num_surf-1] = rlist[1] / 1000.

    return surf_frac


def load_albedo(root_dir, filename):

    # Read albedo data       7 MODIS 8 WV
    albedo_table = np.zeros((8, 9))
    csvfname = os.path.join(root_dir, filename)
    with open(csvfname) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            b = int(row[0])-1
            for c in range(1, len(row)):
                albedo_table[b, c - 1] = float(row[c])

    print(albedo_table)
    return albedo_table


def write_hdf(fname, features, labels):
    tds_file = h5py.File(fname, 'w')
    tds_file.create_dataset("features", data=features)
    tds_file.create_dataset("labels", data=labels)
    tds_file.close()


if __name__ == '__main__':
    main()