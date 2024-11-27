"""
Names: Owen and Daniel
Date: Due Date
Description: Final Project Description

https://ride.citibikenyc.com/about#:~:text=Citi%20Bike%20is%20New%20York,part%20of%20our%20transportation%20network.
"""
# import the kaggle api

import pandas as pd # .read_csv(), DataFrame, .to_datetime(), etc.
import shapely.wkt
from sklearn import decomposition # .PCA(), ..fit(), ..transform()
import matplotlib.pyplot as plt # .scatter(), .legend(), .savefig(), etc.
import numpy as np # unique()
import shapely
import geopandas as gpd
import requests
import requests
import zipfile
import io


def main():

    # Step 1: Download the zip file
    url = 'https://s3.amazonaws.com/tripdata/2013-citibike-tripdata.zip'
    response = requests.get(url)

    # Step 2: Check if the request was successful
    if response.status_code == 200:
        # Step 3: Unzip the content in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all files to the current directory
            z.extractall('data/2013_data')  # Replace 'output_directory' with your preferred path
            print(f"Extracted files: {z.namelist()}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")




    print(CBD(-74.04384499999999,40.717732500000004))
    print(CBD(-74.00595504591429, 40.7128641240805))
    # initialize and autheticate the api
   
    # convert csv to Pandas data frame
    bike_share = pd.read_csv('data/NYC-BikeShare-2015-2017-combined.csv')
    # clean data to include only numeric attributes for PCA
    bike_share = clean_data(bike_share)
    
    # Filter rows for males and females
    males = bike_share[(bike_share["Gender"] == 1) & 
                       (bike_share["hour"] == 12)].head(500)
    females = bike_share[(bike_share["Gender"] == 2) & 
                         (bike_share["hour"] == 12)].head(500)

    # Concatenate the two subsets
    X = pd.concat([males, females], ignore_index=True).drop('Gender', axis=1)
    y = pd.concat([males, females], ignore_index=True)['Gender']

    print(X[X["start_CBD"] == 1])
    print(X[X["end_CBD"] == 1])
    X.to_csv('data/output_file.csv', index=False)

    # seperate data set into predictors X and labels y
    # X = bike_share.iloc[200:300].drop('Gender', axis=1)
    X = transform(X)
    # y = bike_share.iloc[200:300]['Gender']
    # do the PCA visualization
    color_dict = {2: 'pink', 1: 'cyan', 0: 'olive', 3: 'yellow'} # define color dictionary
    bike_color_list = colors(y, color_dict); gender = ['male', 'female']

    # plot the results
    plot(X, bike_color_list, 
         "Dimensionality Reduction on NYC CitiBike dataset",
         gender, color_dict, "figures/bike2.pdf")
    


def clean_data(data):
    """Takes in a Pandas dataframe; purpose is to make all data numeric"""
    print(data.shape)
    # drop useless columns
    data = drop_off(data, ['Stop Time', 'Start Station ID', # start station name
                           'Trip_Duration_in_min', 'End Station ID', 
                           'End Station Name'])
    # convert useful categorical columns to numeric columns
    data['hour'] = pd.to_datetime(data['Start Time']).dt.hour
    data['user_type_num'] = (data['User Type'] != 'Subscriber').astype(int)
    data["Bike Type"] = data["Bike ID"].apply(classify_bike_type)
    data["start_CBD"] = CBD_vectorized(data["Start Station Longitude"], data["Start Station Latitude"])
    data["end_CBD"] = CBD_vectorized(data["End Station Longitude"], data["End Station Latitude"])
    #data["CBD_inter"] = data["start_CBD"] * data["end_CBD"]
    # after conversion, drop the non-numeric columns 
    data = drop_off(data, ['Start Time', 'User Type', 'Start Station Longitude',
                           'Start Station Latitude', 'End Station Longitude', 
                           'End Station Latitude', 'Bike ID'])
    # removes unknown gender label
    data = data[data['Gender'] != 0]
    # drops the  index column
    data = data.iloc[:, 1:]
    print(data.shape)
    # prints the first item to give us an idea of what we're working with
    first_row = data.iloc[0]
    for item in first_row:
        print(item)
    print(first_row.axes)
    
    return data

def drop_off(data, drop_list):
    """Drops each column name in the drop_list from the data frame"""
    for column_name in drop_list:
        data.drop(column_name, axis=1, inplace=True)
    return data

def transform(X):
    """Use sklearn decomposition to run PCA on multivariate dataset"""
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    return pca.transform(X)

def colors(y, color_dict):
    """Given the color dictionary and the list of labels, return a 
    corresponding color list"""
    color_list = []
    for label in y:
        for key in color_dict:
            if label == key:
                color_list.append(color_dict[key])
                continue
    return color_list

def plot(X, color_list, title, names, color_dict, filename):
    """Create a scatterplot with lower-dimension data"""
    plt.scatter(X[:,0], X[:,1], c=color_list, edgecolors='black')
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(title)
    # create legend
    leg_objects = []; n = len(np.unique(color_list))
    for i in range(1,3): # this is data-specific
        circle, = plt.plot([], 'o', c=color_dict[i], 
                           markeredgecolor='black')
        leg_objects.append(circle)
    plt.legend(leg_objects,names)
    # save the plot with the legend
    plt.savefig(filename)
    plt.clf()

def CBD(longitude, latitude):
    # https://autogis-site.readthedocs.io/en/latest/lessons/lesson-3/point-in-polygon-queries.html#
    """Given a (latitude, longitude) pair, returns 1 if the point is within the CBD, and 0 otherwise"""
    point = shapely.geometry.Point(longitude, latitude)
    data = pd.read_csv('data/MTA_Central_Business_District_Geofence__Beginning_June_2024_20241118.csv')
    # Convert the WKT strings into a list of geometries
    geometries = [shapely.wkt.loads(wkt) for wkt in data["polygon"]]
    for polygon in geometries:
        if polygon.contains(point):
            return 1
    return 0

def CBD_vectorized(longitudes, latitudes):
    points = gpd.GeoSeries([shapely.geometry.Point(lon, lat) for lon, lat in zip(longitudes, latitudes)])
    cbd_geometries = gpd.read_file('data/MTA Central Business District Geofence_ Beginning June 2024_20241119.geojson')  # Replace with actual path
    cbd_mask = points.apply(lambda p: any(cbd_geometries.contains(p)))
    return cbd_mask.astype(int)

def classify_bike_type(bike_id): 
    bike_id = int(bike_id)
    if 1 <= bike_id <= 9999:
        return 1  # 1st generation 3-speeds
    elif 10000 <= bike_id <= 13999:
        return 2  # 2nd generation 3-speeds
    elif 14000 <= bike_id <= 19999:
        return 3  # Continuous gear
    elif 20000 <= bike_id <= 29999:
        return 2  # 2nd generation 3-speeds
    elif 30000 <= bike_id <= 35999:
        return 3  # Continuous gear
    elif 36000 <= bike_id <= 38999:
        return 4  # 1st generation electric
    elif 40000 <= bike_id <= 44999:
        return 3  # Continuous gear
    elif 50000 <= bike_id <= 77999:
        return 3  # Continuous gear
    elif 78000 <= bike_id <= 83999:
        return 4  # 1st generation electric
    elif 84000 <= bike_id <= 87999:
        return 3  # Continuous gear
    else:
        return 2  # Default: 2nd generation 3-speeds



if __name__ == '__main__':
    main()
