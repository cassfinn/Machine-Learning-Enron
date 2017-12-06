#!/usr/bin/python

import sys
import pickle
import csv
import matplotlib.pyplot as plot
import math


def make_csv(data_dict):
    """ generates a csv file from the data so we can see the data in a spreadsheet """
    fieldnames = ['name'] + data_dict.itervalues().next().keys()
    with open('data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data_dict:
            person = data_dict[record]
            person['name'] = record
            assert set(person.keys()) == set(fieldnames)
            writer.writerow(person)


def visualize(data_dict, feature_x, feature_y):
    """ generates a plot of feature y vs feature x, colors poi """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])

    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        color = 'red' if poi else 'blue'
        plot.scatter(x, y, color=color)
    plot.xlabel(feature_x)
    plot.ylabel(feature_y)
    plot.show()


def count_invalid_values(data_dict):
    """ counts the number of NaN values for each feature """
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] == 'NaN':
                counts[field] += 1
    return counts


def print_separator_line():
    print " "
    stars = 40
    print(stars * "*")
    print " "