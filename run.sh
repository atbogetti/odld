#!/bin/bash

rm -f west.log
w_run "$@" &> west.log
