#!/bin/bash -l

rm -f west.h5 binbounds.txt
BSTATES="--bstate initial,9.5"
TSTATES="--tstate final,1.9"
w_init $BSTATES $TSTATES "$@"
