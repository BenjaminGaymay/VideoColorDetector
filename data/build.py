#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open('./GX03140', 'r') as fd:
	data = fd.readlines()

with open('./data.csv', 'w') as fd:
	fd.writelines([f'{frame},{value}' for frame, value in enumerate(data)])
