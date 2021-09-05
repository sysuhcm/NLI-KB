#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import csv

# run extract_cpnet_relation to get conceptnet-assertions-5.7.0.csv.en

# convert KB triple to natural language sentence
# divide ConceptNet into four categories

f = open("./conceptnet-assertions-5.7.0.csv.en",'r', encoding='UTF-8')
fw = open("../../datasets/conceptnet.csv", 'w', encoding='UTF-8', newline ='')
writer = csv.writer(fw)

line = f.readline()
count = 0
while line:
    line = line.replace('_', ' ')
    l = line.split('\t')
    line = f.readline()
    first_cat = ''
    if(l[0] == 'motivatedbygoal'):
        sentence = 'Someone does ' + l[1] + ' because they want result ' + l[2]
        first_cat = 'Social-Interaction'
    elif(l[0] == 'desires'):
        sentence = l[1] + ' desire to ' + l[2]
        first_cat = 'Social-Interaction'
    elif(l[0] == 'causesdesire'):
        sentence = l[1] + ' makes someone want ' + l[2]
        first_cat = 'Social-Interaction'

    elif(l[0] == 'hasprerequisite'):
        sentence = 'In order for ' + l[1] + ' to happen, ' + l[2] + ' needs to happen'
        first_cat = 'Event-Centered'
    elif(l[0] == 'obstructedby'):
        sentence = l[1] + ' can be prevented by ' + l[2]
        first_cat = 'Event-Centered'
    elif(l[0] == 'hassubevent'):
        sentence = l[2] + ' happens as a subevent of ' + l[1]
        first_cat = 'Event-Centered'
    elif(l[0] == 'causes'):
        sentence = l[1] + ' causes ' + l[2]
        first_cat = 'Event-Centered'
    elif(l[0] == 'createdby'):
        sentence = l[2] + ' can create  ' + l[1]
        first_cat = 'Event-Centered'
        second_cat = 'Force-Dynamics'
    elif(l[0] == 'haslastsubevent'):
        sentence = 'After ' + l[1] + ', ' + l[2]
        first_cat = 'Event-Centered'
    elif(l[0] == 'hasfirstsubevent'):
        sentence = 'Before ' + l[1] + ', ' + l[2]
        first_cat = 'Event-Centered'
    elif(l[0] == 'receivesaction'):
        sentence = l[2] + ' can be done to ' + l[1] 
        first_cat = 'Event-Centered'
    elif(l[0] == 'mannerOf'):
        sentence = l[1] + ' is a way to do ' + l[2]
        first_cat = 'Event-Centered'

    elif(l[0] == 'hasa'):
        sentence = l[2] + ' belongs to ' + l[1]
        first_cat = 'Physical-Entity'
    elif(l[0] == 'usedfor'):
        sentence = l[1] + ' is used for ' + l[2]
        first_cat = 'Physical-Entity'
    elif(l[0] == 'capableof'):
        sentence = l[1] + ' can be used to ' + l[2]
        first_cat = 'Physical-Entity'
    elif(l[0] == 'atlocation'):
        sentence = l[1] + ' is located at ' + l[2]
        first_cat = 'Physical-Entity'
    elif(l[0] == 'hasproperty'):
        sentence = 'One property of ' + l[1] + ' is ' + l[2]
        first_cat = 'Physical-Entity'
    elif(l[0] == 'locatednear'):
        sentence = l[1] + ' is near ' + l[2]
        first_cat = 'Physical-Entity'
    elif(l[0] == 'madeof'):
        sentence = l[1] + ' is made of ' + l[2]
        first_cat = 'Physical-Entity'

    elif(l[0] == 'partof'):
        sentence = l[1] + ' is part of ' + l[2]
        first_cat = 'Taxonomic-Lexical'
    elif(l[0] == 'isa'):
        sentence = l[1] + ' is a ' + l[2]
        first_cat = 'Taxonomic-Lexical'
    elif(l[0] == 'antonym'):
        sentence = l[1] + ' and ' + l[2] + ' are opposites'
        first_cat = 'Taxonomic-Lexical'
    elif(l[0] == 'distinctfrom'):
        sentence = 'something that is ' + l[1] + ' is not ' + l[2]
        first_cat = 'Taxonomic-Lexical'
    elif(l[0] == 'symbolof'):
        sentence = l[1] + ' symbolically represents ' + l[2]
        first_cat = 'Taxonomic-Lexical'
    elif(l[0] == 'definedas'):
        sentence = l[1] + ' can be defined as ' + l[2]
        first_cat = 'Taxonomic-Lexical'
    elif(l[0] == 'similarto'):
        sentence = l[1] + ' is similar to ' + l[2]
        first_cat = 'Taxonomic-Lexical'
    else:
        continue

    writer.writerow([count, sentence, first_cat, l[0]])
    count = count + 1

f.close()
fw.close()



