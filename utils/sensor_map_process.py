import pandas as pd
import random
import csv
import numpy as np
import geopy.distance
import pickle


def calculate_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of earth in km
    distance = geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km
    return distance


def drop_edges(filename_out: str, stations: list):
    adj = []
    for i, station in enumerate(stations):
        adj.append(station.distances)

    adj = np.array(adj)
    avg_dis = np.average(adj)
    print(avg_dis)

    with open(filename_out, 'wb') as file:
        pickle.dump(adj, file)


class Station:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.distances = None
        self.mean_dis = 0


def load_data_file(file: str):
    df = pd.read_csv(file)
    return df


if __name__ == '__main__':
    df = load_data_file('../data/PEMSD7/PeMSD7_M_Station_Info.csv')

    stations = []
    for index, row in df.iterrows():
        stations.append(Station(row['ID'], row['Longitude'], row['Latitude']))

    for st in stations:
        distances = []
        for _st in stations:
            distance = calculate_distance(lat1=st.lat, lon1=st.lon, lat2=_st.lat, lon2=_st.lon)
            distances.append(distance)

        st.distances = distances
        st.mean_dis = np.sum(np.array(distances)) / (len(stations) - 1)

    all_mean = 0
    for st in stations:
        all_mean += st.mean_dis
        print(st.mean_dis)

    all_mean = all_mean / (len(stations) * 1.0)

    drop_edges(filename_out='../data/PEMSD7/adj_dis.pkl',
               stations=stations)