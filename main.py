#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys
from argparse import ArgumentParser
from ui import *


def inputArg():
    ap = ArgumentParser()
    ap.add_argument("-i", "--image", help="source image name", required=True)
    args = ap.parse_args()

    return args.image


if __name__ == "__main__":
    inputPath = "data/TCAR_DATA"
    w = genWindow(inputPath)



