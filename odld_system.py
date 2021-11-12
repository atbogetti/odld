from __future__ import print_function, division
import logging

import numpy as np
from numpy.random import normal as random_normal

from westpa.core.binning import RectilinearBinMapper
from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem

from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import RecursiveBinMapper

PI = np.pi
log = logging.getLogger("westpa.rc")

pcoord_len = 21
pcoord_dtype = np.float32
# THESE ARE THE FOUR THINGS YOU SHOULD CHANGE
bintargetcount = 10  # number of walkers per bin
nbins = [10,10]  # You will have prod(binsperdim)+numberofdim*(2*splitIsolated) bins total
pcoordlength = 2  # length of the pcoord


class ODLDPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super().__init__(rc)

        self.coord_len = pcoord_len
        self.coord_dtype = pcoord_dtype
        self.coord_ndim = 2

        self.initial_pcoord = np.array([9.5,9.5], dtype=self.coord_dtype)

        self.sigma = 0.001 ** (0.5)  # friction coefficient

        self.A = 2
        self.B = 10
        self.C = 0.5
        self.x0 = 1
        self.y0 = 1

        # Implement a reflecting boundary at this x value
        # (or None, for no reflection)
        self.reflect_at = 10.0

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""
        state.pcoord = self.initial_pcoord.copy()

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self.initial_pcoord.copy()
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED
        return initial_state

    def propagate(self, segments):

        A, B, C, x0, y0 = self.A, self.B, self.C, self.x0, self.y0

        n_segs = len(segments)

        coords = np.empty(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        for iseg, segment in enumerate(segments):
            coords[iseg, 0] = segment.pcoord[0]

        twopi_by_A = 2 * PI / A
        half_B = B / 2
        sigma = self.sigma
        gradfactor = self.sigma * self.sigma / 2
        coord_len = self.coord_len
        reflect_at = self.reflect_at
        all_displacements = np.zeros(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )
        for istep in range(1, coord_len):
            x = coords[:, istep - 1, 0]
            y = coords[:, istep - 1, 1]

            xarg = twopi_by_A * (x - x0)
            yarg = twopi_by_A * (y - y0)

            eCx = np.exp(C * x)
            eCx_less_one = eCx - 1.0
            eCy = np.exp(C * y)
            eCy_less_one = eCy - 1.0

            all_displacements[:, istep, 0] = xdisplacements = random_normal(
                scale=sigma, size=(n_segs,)
            )
            all_displacements[:, istep, 1] = ydisplacements = random_normal(
                scale=sigma, size=(n_segs,)
            )

            gradx = (
                half_B
                / (eCx_less_one * eCx_less_one)
                * (twopi_by_A * eCx_less_one * np.sin(xarg) + C * eCx * np.cos(xarg))
            )

            grady = (
                half_B
                / (eCy_less_one * eCy_less_one)
                * (twopi_by_A * eCy_less_one * np.sin(yarg) + C * eCy * np.cos(yarg))
            )

            newx = x - gradfactor * gradx + xdisplacements
            newy = y - gradfactor * grady + ydisplacements

            if reflect_at is not None:
                # Anything that has moved beyond reflect_at must move back that much

                # boolean array of what to reflect
                to_reflectx = newx > reflect_at
                to_reflecty = newy > reflect_at

                # how far the things to reflect are beyond our boundary
                reflect_byx = newx[to_reflectx] - reflect_at
                reflect_byy = newy[to_reflecty] - reflect_at

                # subtract twice how far they exceed the boundary by
                # puts them the same distance from the boundary, on the other side
                newx[to_reflectx] -= 2 * reflect_byx
                newy[to_reflecty] -= 2 * reflect_byy

            coords[:, istep, 0] = newx
            coords[:, istep, 1] = newy

        for iseg, segment in enumerate(segments):
            segment.pcoord[...] = coords[iseg, :]
            segment.data["displacement"] = all_displacements[iseg]
            segment.status = segment.SEG_STATUS_COMPLETE

        return segments
