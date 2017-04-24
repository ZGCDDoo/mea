#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def compare_arrays(arr1, arr2, rprecision=10**-7, n_diff_max=0, zero_equivalent=10**-10):
    """ """
    try:
        assert(arr1.shape == arr2.shape)
    except AssertionError as err:
        print("SHAPES = : ", arr1.shape, " ", arr2.shape)
        print("Ayaya, arrays are not the same shape : {0}".format(err)) ; raise
        
    count = 0
    for (a ,b) in zip(np.abs(arr1.flatten()), np.abs(arr2.flatten()) ):
        if a < zero_equivalent and b < zero_equivalent:
            continue
        tmp = abs(a - b)/abs(a)
        if tmp > rprecision:
            count += 1
            print("TMP = ", tmp)
        if count > n_diff_max:
            raise AssertionError 