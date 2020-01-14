import numpy as np
from lib import rosel_alg

def monte_carlo_simulation(refl, srm, func=rosel_alg.spec_unmix):
    '''
    Runs a simulation of 200 samples first holding srm constant and varying
      the observed reflectance, then the opposite.
    :param refl: observed reflectance
    :param srm: chosen spectral reflectance matrix
    :param func: function to use for unmixing. Must accept arguments (refl, srm)
    :return: MonteCarlo errors in each surface category, and each test
            [mce_refl_i, mce_refl_m, mce_refl_o,
             mce_srm_i, mce_refl_m, mcd_refl_o]
    '''
    ## Apply MonteCarlo error propagation
    # Create a multivariate normal distribution from which to draw samples.
    # Mean at the observed reflectance, covariance defined by modis accuracy
    stdev = 0.005 + np.multiply(0.05, refl)
    # Convert to covariance by squaring stdev
    cov = np.diag(stdev)
    cov = np.square(cov)
    samples = np.random.multivariate_normal(refl, cov, 200)
    repeats_refl = []
    for sample in samples:
        repeats_refl.append(np.divide(func(sample, srm=srm), 1000.))

    repeats_refl = np.std(repeats_refl, axis=0)

    ## MonteCarlo error propegation in SRM
    stdev = 0.05
    repeats_srm = []
    for _ in range(200):
        # Create a new SRM with added error
        sample = np.copy(srm)
        for b in range(3):
            for s in range(3):
                sample_i = np.random.normal(srm[b][s], srm[b][s] * stdev)
                if sample_i < 0:
                    sample_i = 0
                elif sample_i > 1:
                    sample_i = 1
                sample[b][s] = sample_i
        repeats_srm.append(np.divide(func(refl, srm=sample), 1000.))

    repeats_srm = np.std(repeats_srm, axis=0)

    mc_errors = np.append(repeats_refl, repeats_srm, axis=0)

    return mc_errors

def monte_carlo_ml(refl, model):
    stdev = 0.005 + np.multiply(0.05, refl)
    # Convert to covariance by squaring stdev
    cov = np.diag(stdev)
    cov = np.square(cov)
    samples = np.random.multivariate_normal(refl, cov, 200)
    repeats_refl = []
    for sample in samples:
        repeats_refl.append(np.divide(rosel_alg.ml_estimation(sample, model), 1000.))

    repeats_refl = np.std(repeats_refl, axis=0)

    return repeats_refl