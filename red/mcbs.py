'''
Tools for Monte Carlo bootstrap error analysis
'''

import math
import numpy as np


def get_bssize(alpha):
    '''Return a bootstrap data set size appropriate for the given confidence level'''
    return int(10 ** (math.ceil(-math.log10(alpha)) + 1))


def mcbs_ci(dataset, estimator, alpha, dlen, n_sets=None, args=None, kwargs=None, sort=np.msort, bayesian=False, return_data=False):
    '''Perform a Monte Carlo bootstrap estimate for the (1-``alpha``) confidence interval
    on the given ``dataset`` with the given ``estimator``.  This routine is not appropriate
    for time-correlated data.

    Returns ``(estimate, ci_lb, ci_ub)`` where ``estimate`` is the application of the
    given ``estimator`` to the input ``dataset``, and ``ci_lb`` and ``ci_ub`` are the
    lower and upper limits, respectively, of the (1-``alpha``) confidence interval on
    ``estimate``.

    ``estimator`` is called as ``estimator(dataset, *args, **kwargs)``. Common estimators include:
      * numpy.mean -- calculate the confidence interval on the mean of ``dataset``
      * numpy.median -- calculate a confidence interval on the median of ``dataset``
      * numpy.std -- calculate a confidence interval on the standard deviation of ``datset``.

    ``n_sets`` is the number of synthetic data sets to generate using the given ``estimator``,
    which will be chosen using `get_bssize()`_ if ``n_sets`` is not given.

    ``sort`` can be used
    to override the sorting routine used to calculate the confidence interval, which should
    only be necessary for estimators returning vectors rather than scalars.
    '''

    if alpha > 0.5:
        raise ValueError('alpha ({}) > 0.5'.format(alpha))

    args = args or ()
    kwargs = kwargs or {}

    # dataset SHOULD be a dictionary.
    d_input = dataset.copy()
    # Here, we're dumping in any extra kwarg arguments to pass in to the estimator.
    try:
        d_input.update(kwargs)
    except Exception:
        pass

    def_indices = np.arange(dlen)
    def_weights = np.ones(dlen, dtype=float)

    d_input['indices'] = def_indices
    d_input['weights'] = def_weights
    fhat = estimator(**d_input)

    try:
        estimator_shape = fhat.shape
    except AttributeError:
        estimator_shape = ()

    try:
        estimator_dtype = fhat.dtype
    except AttributeError:
        estimator_dtype = type(fhat)

    n_sets = n_sets or get_bssize(alpha)

    f_synth = np.empty((n_sets,) + estimator_shape, dtype=estimator_dtype)
    k = np.ones(dlen, dtype=float)

    for i in range(n_sets):
        # print("\rdatasets %d/%d (%3.0f%%)"%(i+1, n_sets, (i+1)*100.0/n_sets), end="")

        if bayesian:
            indices = def_indices
            weights = np.random.dirichlet(k, size=1).flatten()
            d_synth = dataset
        else:
            indices = np.random.randint(dlen, size=(dlen,))
            weights = def_weights
            d_synth = {}
            for key, dset in dataset.items():
                d_synth[key] = np.take(dset, indices, axis=0)
        d_input = d_synth.copy()
        d_input['indices'] = indices
        d_input['weights'] = weights

        try:
            d_input.update(kwargs)
        except Exception:
            pass
        f_synth[i] = estimator(**d_input)

    # print("\n")

    f_synth_sorted = sort(f_synth)
    lbi = int(math.floor(n_sets*alpha/2.0))
    ubi = int(math.ceil(n_sets*(1-alpha/2.0)))-1
    lb = f_synth_sorted[lbi]
    ub = f_synth_sorted[ubi]
    sterr = np.std(f_synth_sorted)

    if return_data:
        return fhat, lb, ub, sterr, f_synth_sorted
    return (fhat, lb, ub, sterr)


def sequence_macro_flux_to_rate(dataset, pops, istate, jstate, **kwargs):
    '''Convert a sequence of macrostate fluxes and corresponding list of trajectory ensemble populations
    to a sequence of rate matrices.

    If the optional ``pairwise`` is true (the default), then rates are normalized according to the
    relative probability of the initial state among the pair of states (initial, final); this is
    probably what you want, as these rates will then depend only on the definitions of the states
    involved (and never the remaining states). Otherwise (``pairwise'' is false), the rates are
    normalized according the probability of the initial state among *all* other states.'''

    weights = kwargs.get('weights', None)
    pairwise = kwargs.get('pairwise', True)
    dc = kwargs.get('corrector', None)

    # durations= kwargs.pop('durations', None)
    # weights= kwargs.pop('weights', None)
    # dtau= kwargs.pop('dtau', None)

    dlen = dataset.shape[0]
    _fluxsum = 0.0
    _psum = 0.0

    if weights is None:
        weights = np.ones(dlen, dtype=float)

    # We want to modify this to be the SUM of fluxes up till this point, divided by the SUM of the population till then.
    for iiter in range(dlen):
        if pairwise and (pops[iiter, istate] + pops[iiter, jstate]) != 0.0:
            _psum += (pops[iiter, istate] / (pops[iiter, istate] + pops[iiter, jstate])) * weights[iiter]
        else:
            _psum += pops[iiter, istate] * weights[iiter]
        _fluxsum = dataset[iiter] * weights[iiter] + _fluxsum

    if _psum > 0 and _fluxsum > 0:
        rate = _fluxsum / _psum
    else:
        rate = 0.0

    # evaluate the red rate if data is provided
    if dc is not None:
        indices = kwargs.get('indices', None)
        weights = kwargs.get('weights', None)
        c = dc.correction(indices, weights)
        rate *= c

    return rate
