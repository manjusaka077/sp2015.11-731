#-*- coding: utf-8 -*-
#!/usr/bin/env python


import sys

filename = sys.argv[1]

f_input = open(filename, 'r')
f_de = open("utf8_de.txt", 'w')
f_eng = open("utf8_eng.txt", 'w')

for line in f_input :
	line = line.strip()
	pairs = line.split("|||")
	f_de.write(pairs[0] + "\n")
	f_eng.write(pairs[1] + "\n")

f_input.close()
f_de.close()
f_eng.close()