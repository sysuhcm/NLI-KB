#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import csv

# download ATOMIC: 2020:https://allenai.org/data/atomic-2020

# convert KB triple to natural language sentence
# divide ATOMIC 2020 into five categories

f = open("./atomic/train.tsv",'r', encoding='UTF-8') # test.tsv && dev.tsv
fw = open("../datasets/atomic.csv", 'a', encoding='UTF-8', newline ='')
writer = csv.writer(fw)

line = f.readline()
count = 0
while line:
    line = line[:-1]
    l = line.split('\t')
    line = f.readline()

    if(l[2] == 'none'):
        continue
    sentence = ""
    first_cat = ''
    second_cat = ''
    if(l[0].find('___') != -1 and l[1] != 'isFilledBy'):
        index = l[0].find('___')
        l[0] =  l[0][:index] + 'something' + l[0][index + 3:]
    if(l[2].find('___') != -1):
        index = l[2].find('___')
        l[2] =  l[2][:index] + 'something' + l[2][index + 3:]
    if(l[1] == 'ObjectUse'):
        sentence = l[0] + ' can be used to ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Affordances'
    elif(l[1] == 'CapableOf'):
        sentence = l[0] + ' is capable of ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Capabilities'
    elif(l[1] == 'MadeUpOf'):
        sentence = l[0] + ' is made up of ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Properties'
    elif(l[1] == 'HasProperty'):
        sentence = 'One property of ' + l[0] + ' is ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Properties'
    elif(l[1] == 'Desires'):
        sentence = l[0] + ' desire to ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Desires'
    elif(l[1] == 'NotDesires'):
        sentence = l[0] + ' do not desire to ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Desires'
    elif(l[1] == 'AtLocation'):
        sentence = l[0] + ' is located at ' + l[2]
        first_cat = 'Physical-Entity'
        second_cat = 'Spatial'
    elif(l[1] == 'Causes'):
        sentence = l[0] + ' causes ' + l[2]
        first_cat = 'Event-Centered'
        second_cat = 'Force-Dynamics'
    elif(l[1] == 'HinderedBy'):
        sentence = 'If ' + l[2] + ', PersonX cannot ' + l[0][7:]
        first_cat = 'Event-Centered'
        second_cat = 'Force-Dynamics'
    elif(l[1] == 'xReason'):
        sentence = l[0] + ' beacause ' + l[2]
        first_cat = 'Event-Centered'
        second_cat = 'Force-Dynamics'
    elif(l[1] == 'isAfter'):
        sentence = 'After ' + l[2] + ', ' + l[0]
        first_cat = 'Event-Centered'
        second_cat = 'Scripts'
    elif(l[1] == 'isBefore'):
        sentence = 'Before ' + l[2] + ', ' + l[0]
        first_cat = 'Event-Centered'
        second_cat = 'Scripts'
    elif(l[1] == 'HasSubEvent'):
        sentence = 'If ' + l[0] + ',  then ' + l[2] 
        first_cat = 'Event-Centered'
        second_cat = 'Scripts'
    elif(l[1] == 'isFilledBy'):
        index = l[0].find('___')
        sentence = l[0][:index] + l[2] + l[0][index + 3:]
        first_cat = 'Event-Centered'
        second_cat = 'isFilledBy'
    elif(l[1] == 'xIntent' or l[1] == 'xWant'):
        sentence = 'If ' + l[0] + ', then PersonX wants ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'MentalState'
    elif(l[1] == 'xReact'):
        sentence = 'If ' + l[0] + ', then PersonX feels ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'MentalState'
    elif(l[1] == 'oReact'):
        other = 'others'
        if(l[0].find("PersonY") != -1):
            other = 'PersonY'
        sentence = 'If ' + l[0] + ', then ' + other + ' feel ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'MentalState'
    elif(l[1] == 'xAttr'):
        sentence = 'If ' + l[0] + ', then PersonX is ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'Persona'
    elif(l[1] == 'xEffect'):
        sentence = 'If ' + l[0] + ', then PersonX ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'Behavior'
    elif(l[1] == 'xNeed'):
        sentence = 'If ' + l[0] + ', then PersonX needs ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'Behavior'
    elif(l[1] == 'xWant'):
        sentence = 'If ' + l[0] + ', then PersonX wants ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'Behavior'
    elif(l[1] == 'oEffect'):
        other = 'others'
        if(l[0].find("PersonY") != -1):
            other = 'PersonY'
        if(l[2].find("PersonX") != -1):
            other = ''
        sentence = 'If ' + l[0] + ', then ' + other + ' ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'Behavior'
    elif(l[1] == 'oWant'):
        other = 'others'
        if(l[0].find("PersonY") != -1):
            other = 'PersonY'
        sentence = 'If ' + l[0] + ', then ' + other + ' want ' + l[2]
        first_cat = 'Social-Interaction'
        second_cat = 'Behavior'
    else:
        sentence = l[1]

    writer.writerow([count, sentence, first_cat, second_cat, l[1]])
    count = count + 1

f.close()
fw.close()




