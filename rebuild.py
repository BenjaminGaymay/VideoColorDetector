#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint

data = []
known = {3603: 10769, 23080: 398, 23081: 96603, 23082: 57797, 45691: 1423, 45692: 81174, 45693: 469}

for i in range(48000):
	if i in known:
		data.append(known[i])
	else:
		data.append(randint(340, 370))

with open('GX03140', 'w') as fd:
	for i in data:
		fd.write(f'{i}\n')