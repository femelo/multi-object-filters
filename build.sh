#!/bin/bash

nuitka3 --standalone --onefile --static-libpython=yes --enable-plugin=multiprocessing --enable-plugin=numpy --enable-plugin=tk-inter demo.py
