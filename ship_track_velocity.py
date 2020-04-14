"""
    This file contains functionalities to read navigation data from the ACE cruise
    It is used in the RENKU repo cruise-track-legs0-4 to calcualte the the best estiamte of the ship's track and velocity at one and five minute resoulution
"""

import pandas as pd
import numpy as np
import datetime

import glob # for concating the hydrins files
import os


import pyproj
# The Proj class can convert from geographic (longitude,latitude) to native map projection (x,y) coordinates and vice versa
# pyproj.Geod(ellps='WGS84') # https://jswhit.github.io/pyproj/pyproj.Geod-class.html
sphere_model_for_pyproj = 'WGS84' # some approximation of the Geoid
# we will use inv
# inv(self, lons1, lats1, lons2, lats2, radians=False)
# inverse transformation - Returns forward and back azimuths, plus distances between initial points (specified by lons1, lats1) and terminus points (specified by lons2, lats2).
# Works with numpy and regular python array objects, python sequences and scalars.
# if radians=True, lons/lats and azimuths are radians instead of degrees. Distances are in meters.
    
merge_at_nseconds = 60 # output at 60 seconds
USE_HYDRINGS = True

def ang180(x):
    """
        Function to mapp angular data over the interval [-180 +180)

        :param x: data series
        :returns: the data series mapped into [-180 +180)
    """
    return ((x+180)%360)-180 # map any number into -180, 180 keeping the order the same


def minute_track_from_hydrins_data():
    """
        Function to read and filter the uncorrected interatial navigation data

        :returns: data frame containing the ships track and velocity at 1 minute resolution for the second half of leg0
    """
    hydrins_path = r'./data/uncorrected_inertial_nav_10/' # use your path
    all_files = glob.glob(os.path.join(hydrins_path, "*2016*.csv")) # we are only interested in the 2016 data where the 1sec track is incompleete

    hydrins = pd.concat((pd.read_csv(f, usecols =['pc_date_utc', 'hydrins_time_hhmmsssss', 'north_speed_ms-1', 'east_speed_ms-1', 
                                                             'latitude','longitude'] , low_memory=False) for f in all_files), axis=0)
    #  'heading', 'platform_roll', 'platform_pitch', 'vertical_speed_ms-1', 'gps_latitude', 'gps_longitude'
    hydrins.reset_index(drop=True, inplace=True)

    #timest_= pd.to_datetime(pd.to_numeric(pd.to_datetime(hydrins['hydrins_date_utc'] + ' ' + hydrins['hydrins_time_hhmmsssss'], errors = 'coerce') ).interpolate())
    # do this setp by step to note the bad values and make sure that NaT goes to NaN!
    timest_= pd.to_datetime(hydrins['pc_date_utc'] + ' ' + hydrins['hydrins_time_hhmmsssss'], errors = 'coerce') # set occurence of HH:MM:60.FFF to NAT.
    NaT_index = np.where(np.isnat(timest_)) # store where we had bad time stamps
    #print(np.where(np.isnat(timest_)))
    timest_ = pd.to_numeric( timest_ )
    timest_[timest_<0]=np.NaN
    timest_= pd.to_datetime(timest_.interpolate()).astype('datetime64[ms]') # compression to ms accuracy

    hydrins.set_index(pd.DatetimeIndex(timest_, name='date_time'), inplace=True)
    hydrins = hydrins.sort_index() # completely chaotic order of the data lines in the files ^^
    hydrins.index = pd.to_datetime(hydrins.index)

    hydrins['longitude'] = ang180(hydrins['longitude'])

    hydrins = hydrins.rename(index=str, columns={"north_speed_ms-1": "velNorth", "east_speed_ms-1": "velEast"})

    hydrins.drop(columns=['pc_date_utc', 'hydrins_time_hhmmsssss'], inplace=True)

    hydrins = hydrins.assign( SOG = np.sqrt(np.square(hydrins.velNorth)+np.square(hydrins.velEast)))
    
    # remove data with unrealistically high speed
    for var_str in ['velEast','velNorth', 'SOG']:
        hydrins.at[(hydrins.SOG>9), var_str] = np.nan
    
    hydrins.index = pd.to_datetime(hydrins.index)
    hydrins=hydrins.tz_localize(tz='UTC') # cause time stamp not with +00:00:00

    # simple filtering of spikes based on 60 second rolling window
    SPIKES_SOG = np.abs(hydrins.SOG-hydrins.SOG.interpolate().rolling(window=60, center=True).mean())>1
    SPIKES_LATLON = ((np.abs(hydrins.latitude-hydrins.latitude.interpolate().rolling(window=60, center=True).mean())>1) | (np.abs(hydrins.longitude-hydrins.longitude.interpolate().rolling(window=60, center=True).mean())>1) )

    if 1:
        for var_str in ['SOG','velEast','velNorth']:
            hydrins.at[SPIKES_SOG, var_str] = np.nan
    if 1:
        for var_str in ['latitude','longitude']:
            hydrins.at[SPIKES_LATLON, var_str] = np.nan
            
    lon_median=hydrins['longitude'].resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5)) ).median() # calculate 6 second mean to match wind data

    SOG = hydrins.SOG.copy();
    SOG_MAX=SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5))).max()
    SOG_MIN=SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5))).min()

    hydrins=hydrins.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5)) ).mean() # calculate 1min mean to match wind data
    hydrins = hydrins.assign( SOG_DIFF = (SOG_MAX-SOG_MIN))
    # here we fix dateline issues of averaging longitudes around +/-180 degree
    # this is a bit coarse, but it works.
    lon_flip_tollerance = 0.0005 # a value well aboved the normal difference of mean and median longitudes
    hydrins['longitude'][np.abs(hydrins.longitude-lon_median)>lon_flip_tollerance]=lon_median[np.abs(hydrins.longitude-lon_median)>lon_flip_tollerance]

    hydrins['longitude'] = ang180(hydrins['longitude'])

    hydrins = hydrins.assign( COG = ((90-np.rad2deg(np.arctan2(hydrins.velNorth,hydrins.velEast))) % 360) ) # recompute COG from averaged North/Easte velocities
    hydrins.SOG = np.sqrt(np.square(hydrins.velNorth)+np.square(hydrins.velEast) ) # vector average velocity

    return hydrins
    

def minute_track_from_1sec_track():
    """
        Function to read qualitychecked one second GPS track data and derive the ships velocity. Some basic filtering is applied to remove faulty data.

        :returns: data frame containing the ships track and velocity at 1 minute resolution for legs 1 to 4
    """
    # velocities are calculated based on 1sec time series and then vector averaged to 1min
    
    #gps_csv_file_name = 'ace_cruise_track_1sec_2017-02.csv'
    #df_gps = pd.read_csv(gps_csv_file_folder+gps_csv_file_name)
    gps_csv_file_folder = './data/qualitychecked_onesec_10/'
    
    gps_csv_file_name0 = 'ace_cruise_track_1sec_2016-12.csv'
    gps_csv_file_name1 = 'ace_cruise_track_1sec_2017-01.csv'
    gps_csv_file_name2 = 'ace_cruise_track_1sec_2017-02.csv'
    gps_csv_file_name3 = 'ace_cruise_track_1sec_2017-03.csv'
    gps_csv_file_name4 = 'ace_cruise_track_1sec_2017-04.csv'
    print("reading GPS files ...")

    if 1:
        df_gps0 = pd.read_csv(gps_csv_file_folder+gps_csv_file_name0, usecols=['date_time', 'latitude', 'longitude', 'fix_quality', 'device_id'])
        df_gps1 = pd.read_csv(gps_csv_file_folder+gps_csv_file_name1, usecols=['date_time', 'latitude', 'longitude', 'fix_quality', 'device_id'])
        df_gps2 = pd.read_csv(gps_csv_file_folder+gps_csv_file_name2, usecols=['date_time', 'latitude', 'longitude', 'fix_quality', 'device_id'])
        df_gps3 = pd.read_csv(gps_csv_file_folder+gps_csv_file_name3, usecols=['date_time', 'latitude', 'longitude', 'fix_quality', 'device_id'])
        df_gps4 = pd.read_csv(gps_csv_file_folder+gps_csv_file_name4, usecols=['date_time', 'latitude', 'longitude', 'fix_quality', 'device_id'])

        df_gps = [df_gps0, df_gps1, df_gps2, df_gps3, df_gps4] # concatenate the 4 wind data files
        df_gps = pd.concat(df_gps)
    else:
        df_gps = pd.read_csv(gps_csv_file_folder+gps_csv_file_name0)
    print("... done")
    #df_gps = pd.read_csv(gps_csv_file_folder+gps_csv_file_name1)
    #df_gps = df_gps.rename(index=str, columns={"date_time": "timest_"})
    df_gps = df_gps.set_index(pd.to_datetime(df_gps.date_time, format="%Y-%m-%d %H:%M:%S")) # assing the time stamp
    df_gps.drop(columns=['date_time'], inplace=True)

    SOG = np.ones_like(df_gps.latitude)*np.nan
    COG = np.ones_like(df_gps.latitude)*np.nan
    velEast = np.ones_like(df_gps.latitude)*np.nan
    velNorth = np.ones_like(df_gps.latitude)*np.nan
    
    lon1 = np.array(df_gps.longitude[0:-1])
    lat1 = np.array(df_gps.latitude[0:-1])
    lon2 = np.array(df_gps.longitude[1:])
    lat2 = np.array(df_gps.latitude[1:])
    
    dt = (df_gps.index[1:]-df_gps.index[0:-1]); dt=np.array(dt.total_seconds())
    
    MissingValues = np.where(np.isnan(lon1*lon2*lat1*lat2))
    
    lon1[MissingValues]=0;
    lon2[MissingValues]=0;
    lat1[MissingValues]=0;
    lat2[MissingValues]=0;

    (az12, az21, dist) = pyproj.Geod(ellps=sphere_model_for_pyproj).inv(lon1, lat1, lon2, lat2) 
    vel = np.true_divide(dist,dt)
    vel[MissingValues] = np.nan
    print(str(np.sum(vel>12))+'samples with velocity larger than 12 m/s')
    vel[vel>12]=np.nan # cut unrealistic velocities (vel > 12 m/s)
    vel[dt>(1*5)]=np.nan # set time diffs with dt>5sec to NaN to  (don't trust edge of data gap velocities)
    vel[np.abs(dt-np.round(dt))>0.015] = np.nan # these seam to be very noisy in position
    vel[(df_gps.fix_quality[1:].values)==6] = np.nan # 6 = estimated (dead reckoning) (2.3 feature)
    vel[(df_gps.fix_quality[:-1].values)==6] = np.nan
    vel[np.abs(np.diff(df_gps.device_id))>0] = np.nan # switching of the devices can cause wrong velocities
    
    evel = np.cos((-az12+90)*np.pi/180)*vel;
    nvel = np.sin((-az12+90)*np.pi/180)*vel;
    
    timest_vel = df_gps.index[1:]+pd.to_timedelta(dt*0.5, unit='s') # just to have a time stamp to plot vel against
    
    # note SOG, COG are recomputed from velEast velNorth below!
    SOG[1:-1] = np.nanmean([vel[0:-1], vel[1:]],axis=0); # calculate sog as avg of vel(jj-1,jj) and vel(jj,jj+1)
    SOG[0] = vel[0]; SOG[-1] = vel[-1]; # fill first and last reading

    COG[1:-1] = np.rad2deg((np.arctan2( np.nanmean([vel[0:-1]*np.sin(np.deg2rad(az12[0:-1])) , vel[1:]*np.sin(np.deg2rad(az12[1:])) ], axis=0), np.nanmean([vel[0:-1]*np.cos(np.deg2rad(az12[0:-1])) , vel[1:]*np.cos(np.deg2rad(az12[1:])) ], axis=0) )  + 2 * np.pi) % (2 * np.pi) )
    COG[0] = az12[0]; COG[-1] = az12[-1]; # fill first and last reading

    velEast[1:-1] = np.nanmean([evel[0:-1], evel[1:]],axis=0);
    velEast[0] = evel[0]; velEast[-1] = evel[-1]; 
    velNorth[1:-1] = np.nanmean([nvel[0:-1], nvel[1:]],axis=0);
    velNorth[0] = nvel[0]; velNorth[-1] = nvel[-1]; 
    
    df_gps = df_gps.assign( SOG = SOG)
    df_gps = df_gps.assign( COG = COG)
    df_gps = df_gps.assign( velEast = velEast)
    df_gps = df_gps.assign( velNorth = velNorth)
    
    lon_median=df_gps['longitude'].resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5)) ).median() # calculate 6 second mean to match wind data
 
    nGPS=df_gps.SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).count() # counts per 1min average, put the interval center at the center of the 1min interval

    SOG = df_gps.SOG.copy();
    SOG_MAX=SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5))).max()
    SOG_MIN=SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5))).min()

    #df_gps=df_gps.resample('6S', loffset = datetime.timedelta(seconds=3) ).mean() # calculate 6 second mean to match wind data
    df_gps=df_gps.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5)) ).mean() # calculate 1min mean to match wind data, put the interval center at the center of the 1min interval

    df_gps = df_gps.assign( SOG_DIFF = (SOG_MAX-SOG_MIN))

    
    # here we fix dateline issues of averaging longitudes around +/-180 degree
    # this is a bit coarse, but it works.
    lon_flip_tollerance = 0.0005 # a value well aboved the normal difference of mean and median longitudes
    df_gps['longitude'][np.abs(df_gps.longitude-lon_median)>lon_flip_tollerance]=lon_median[np.abs(df_gps.longitude-lon_median)>lon_flip_tollerance]

    
    df_gps.COG = (90-np.rad2deg(np.arctan2(df_gps.velNorth,df_gps.velEast))) % 360 # recompute COG from averaged North/Easte velocities
    df_gps.SOG = np.sqrt(np.square(df_gps.velNorth)+np.square(df_gps.velEast) ) # vector average velocity
    

    if 1:
        print( 'nGPS removing ' + str( np.sum((nGPS>0) & (nGPS<10)) ) + ' of '+str(len(df_gps)) )
        # REMOVE 1min averages with less than 20 readings (out of60) # ~0.24%
        # REMOVE 1min averages with less than 10 readings (out of60) # ~0.026%

        for val in df_gps.columns:
            df_gps.at[nGPS.values<10, val] = np.nan
            #df_gps = df_gps.assign( nGPS = nGPS)
    
    df_gps.drop(columns=['fix_quality', 'device_id'], inplace=True)
    
    return df_gps

def read_and_filter_wind_data():
    """
        Function to read 1/3 second wind data record and remove spurious data

        :returns: data frame containing the filtered wind speed data at 1/3 second resolution for legs 0 to 4
    """
    wind_csv_file_folder = './data/summary_raw_wind_data_fr_10/'
    wind_csv_file_name0 = 'metdata_wind_20161119_20161216.csv'
    wind_csv_file_name1 = 'metdata_wind_20161220_20170118.csv'
    wind_csv_file_name2 = 'metdata_wind_20170122_20170223.csv'
    wind_csv_file_name3 = 'metdata_wind_20170226_20170319.csv'
    wind_csv_file_name4 = 'metdata_wind_20170322_20170411.csv'

    df_wind0 = pd.read_csv(wind_csv_file_folder+wind_csv_file_name0)
    df_wind1 = pd.read_csv(wind_csv_file_folder+wind_csv_file_name1)
    df_wind2 = pd.read_csv(wind_csv_file_folder+wind_csv_file_name2)
    df_wind3 = pd.read_csv(wind_csv_file_folder+wind_csv_file_name3)
    df_wind4 = pd.read_csv(wind_csv_file_folder+wind_csv_file_name4)
    
    df_wind0 = df_wind0.set_index( (pd.to_datetime(df_wind0.date_time, format="%Y-%m-%d %H:%M:%S")+pd.to_timedelta(df_wind0.TIMEDIFF, unit='s'))  ) # assing the time stamp
    
    df_wind1 = df_wind1.set_index( (pd.to_datetime(df_wind1.date_time, format="%Y-%m-%d %H:%M:%S")+pd.to_timedelta(df_wind1.TIMEDIFF, unit='s'))  ) # assing the time stamp
    
    df_wind2 = df_wind2.set_index( (pd.to_datetime(df_wind2.date_time, format="%Y-%m-%d %H:%M:%S")+pd.to_timedelta(df_wind2.TIMEDIFF, unit='s'))  ) # assing the time stamp
    
    df_wind3 = df_wind3.set_index( (pd.to_datetime(df_wind3.date_time, format="%Y-%m-%d %H:%M:%S")+pd.to_timedelta(df_wind3.TIMEDIFF, unit='s'))  ) # assing the time stamp
    
    df_wind4 = df_wind4.set_index( (pd.to_datetime(df_wind4.date_time, format="%Y-%m-%d %H:%M:%S")+pd.to_timedelta(df_wind4.TIMEDIFF, unit='s'))  ) # assing the time stamp
    
    frames = [df_wind0, df_wind1, df_wind2, df_wind3, df_wind4] # concatenate the 4 wind data files
    df_wind = pd.concat(frames)
    
    #df_wind0=df_wind0.tz_localize(tz='UTC') # cause time stamp not with +00:00:00
    
    max_wind_WS = 40    # True wind speed maximum value (arbitrary)
    max_wind_WSR = 100 # bad data flaged with 999 # one spike in WSR2 at 700 
    # removes all bad directions (out of 0-360) for 1 but not for 2!

    print('WSR1: ' +str(np.sum(df_wind.WSR1>-1))+' samples for WSR1')
    print('WSR2: ' +str(np.sum(df_wind.WSR2>-1))+' samples for WSR2')

    
    print('Removing ' +str(np.sum(df_wind.WSR1>max_wind_WSR))+' samples for WSR1 > 100 m/s')

    print('Removing ' +str(np.sum(df_wind.WSR2>max_wind_WSR))+' samples for WSR2 > 100 m/s')

    df_wind.at[df_wind.WSR1>max_wind_WSR, 'WD1'] = np.nan
    df_wind.at[df_wind.WSR1>max_wind_WSR, 'WS1'] = np.nan
    df_wind.at[df_wind.WSR1>max_wind_WSR, 'WDR1'] = np.nan
    df_wind.at[df_wind.WSR1>max_wind_WSR, 'WSR1'] = np.nan

    df_wind.at[df_wind.WS1>max_wind_WS, 'WD1'] = np.nan
    #df_wind.at[df_wind.WS1>max_wind_WS, 'WDR1'] = np.nan
    #df_wind.at[df_wind.WS1>max_wind_WS, 'WSR1'] = np.nan
    df_wind.at[df_wind.WS1>max_wind_WS, 'WS1'] = np.nan

    df_wind.at[df_wind.WSR2>max_wind_WSR, 'WD2'] = np.nan
    df_wind.at[df_wind.WSR2>max_wind_WSR, 'WS2'] = np.nan
    df_wind.at[df_wind.WSR2>max_wind_WSR, 'WDR2'] = np.nan
    df_wind.at[df_wind.WSR2>max_wind_WSR, 'WSR2'] = np.nan

    df_wind.at[df_wind.WS2>max_wind_WS, 'WD2'] = np.nan
    #df_wind.at[df_wind.WS2>max_wind_WS, 'WDR2'] = np.nan
    #df_wind.at[df_wind.WS2>max_wind_WS, 'WSR2'] = np.nan
    df_wind.at[df_wind.WS2>max_wind_WS, 'WS2'] = np.nan
    
    # remove all W2 readings for WDR2>360
    print('Removing ' +str(np.sum(df_wind.WDR1>360))+' samples for WDR1 > 360deg')
    print('Removing ' +str(np.sum(df_wind.WDR2>360))+' samples for WDR2 > 360deg')

    df_wind.at[df_wind.WDR2>360, 'WD2'] = np.nan
    df_wind.at[df_wind.WDR2>360, 'WS2'] = np.nan
    df_wind.at[df_wind.WDR2>360, 'WSR2'] = np.nan
    df_wind.at[df_wind.WDR2>360, 'WD2'] = np.nan
    df_wind.at[df_wind.WDR2>360, 'WDR2'] = np.nan
    df_wind.at[df_wind.WDR2>360, 'WSR2'] = np.nan
    df_wind.at[df_wind.WDR2>360, 'WS2'] = np.nan    
    df_wind.at[df_wind.WDR2>360, 'WDR2'] = np.nan

    dWSR = np.abs(df_wind.WSR1-df_wind.WSR2) # difference between the relative wind speeds
    print('Removing ' +str(np.sum(dWSR>20))+' samples for dWSR > 20 m/s')
    
    df_wind.at[dWSR>20, 'WD1'] = np.nan
    df_wind.at[dWSR>20, 'WS1'] = np.nan
    df_wind.at[dWSR>20, 'WDR1'] = np.nan
    df_wind.at[dWSR>20, 'WSR1'] = np.nan
    df_wind.at[dWSR>20, 'WD2'] = np.nan
    df_wind.at[dWSR>20, 'WS2'] = np.nan
    df_wind.at[dWSR>20, 'WDR2'] = np.nan
    df_wind.at[dWSR>20, 'WSR2'] = np.nan

    # S2 hase some biased low wind speed for 6 hours -> remove WSR2 and WDR2 for these
    print('removing samples for s2 2017-04-05 06:00:00 till 2017-04-05 12:00:00')
    df_wind.at[ ((df_wind.index>pd.to_datetime("2017-04-05 06:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC'))&(df_wind.index<pd.to_datetime("2017-04-05 12:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC')))==1, 'WSR2'] = np.NaN
    df_wind.at[ ((df_wind.index>pd.to_datetime("2017-04-05 06:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC'))&(df_wind.index<pd.to_datetime("2017-04-05 12:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC')))==1, 'WDR2'] = np.NaN
    
    # remove extraneous columns
    df_wind = df_wind.drop(columns=['COG', 'CLOUDTEXT', 'date_time'])
    
    df_wind.index.name = 'date_time'
    
    return df_wind

def minute_track_from_30sec_data_leg0():
    """
        Function to read the 1/30 second meteorological data and drive the ships velocity form the low resolution ship track. Some basic filtering is applied to remove faulty data.

        :returns: data frame containing the ships track and velocity at 1 minute resolution for leg 0
    """
    # before '2016-11-27 10:01:30' we have no 1/sec latitude/longitude record
    # here we load the 1/30sec metdata file and use these latitude/longitude to estimate ship velocity
    # some despiking is needed here!
    csv_file_folder = './data/summary_raw_meteorologic_10/'
    
    met_all_leg0 = pd.read_csv(csv_file_folder+'metdata_all_20161119_20161216.csv', usecols=['date_time','latitude','longitude','TIMEDIFF'])


    met_all_leg0_systime = met_all_leg0.copy
    met_all_leg0_systime = met_all_leg0.set_index( pd.to_datetime(met_all_leg0.date_time, format="%Y-%m-%d %H:%M:%S") )
    met_all_leg0 = met_all_leg0.set_index( (pd.to_datetime(met_all_leg0.date_time, format="%Y-%m-%d %H:%M:%S")+pd.to_timedelta(met_all_leg0.TIMEDIFF, unit='s'))  ) # assing the time stamp
    
    # filtering some unrealistic coordinates
    bad_lad_long = ((met_all_leg0.latitude>22)&(met_all_leg0.latitude<26)&(met_all_leg0.longitude>-14)&(met_all_leg0.longitude<2))
    met_all_leg0.at[bad_lad_long, 'latitude'] = np.nan
    met_all_leg0.at[bad_lad_long, 'longitude'] = np.nan

    # this bounds work for leg0
    met_all_leg0.at[(met_all_leg0.latitude<-36), 'latitude'] = np.nan
    met_all_leg0.at[(met_all_leg0.latitude>57), 'latitude'] = np.nan
        
    SOG = np.ones_like(met_all_leg0.latitude)*np.nan
    COG = np.ones_like(met_all_leg0.latitude)*np.nan
    velEast = np.ones_like(met_all_leg0.latitude)*np.nan
    velNorth = np.ones_like(met_all_leg0.latitude)*np.nan
    
    lon1 = np.array(met_all_leg0.longitude[0:-1])
    lat1 = np.array(met_all_leg0.latitude[0:-1])
    lon2 = np.array(met_all_leg0.longitude[1:])
    lat2 = np.array(met_all_leg0.latitude[1:])
    
    # calculate rolling mean of latitudes to filter outliers later
    met_all_leg0_r = met_all_leg0.rolling(window=5, center=True).mean()
    met_all_leg0_r.at[(met_all_leg0_r.latitude<-36), 'latitude'] = np.nan
    met_all_leg0_r.at[(met_all_leg0_r.latitude>57), 'latitude'] = np.nan
    
    lon1_r =  np.array(met_all_leg0_r['longitude'][0:-1])
    lat1_r =  np.array(met_all_leg0_r['latitude'][0:-1])
    lon2_r =  np.array(met_all_leg0_r['longitude'][1:])
    lat2_r =  np.array(met_all_leg0_r['latitude'][1:])
    
    dt = (met_all_leg0.index[1:]-met_all_leg0.index[0:-1]); 
    #dt = (met_all_leg0_systime.index[1:]-met_all_leg0_systime.index[0:-1]); 
    dt=np.array(dt.total_seconds())
    
    MissingValues = np.where(np.isnan(lon1*lon2*lat1*lat2))
    MissingValues_r = np.where(np.isnan(lon1*lon2*lat1*lat2*lon1_r*lon2_r*lat1_r*lat2_r))

    lon1[MissingValues]=0;
    lon2[MissingValues]=0;
    lat1[MissingValues]=0;
    lat2[MissingValues]=0;

    (az12, az21, dist) = pyproj.Geod(ellps=sphere_model_for_pyproj).inv(lon1, lat1, lon2, lat2) 
    
    lon1[MissingValues_r]=0;
    lon2[MissingValues_r]=0;
    lat1[MissingValues_r]=0;
    lat2[MissingValues_r]=0;
    
    # calculate how fare a point deviates from the rolling mean coordinates and use this to flag outliers
    lon1_r[MissingValues_r]=0;
    lon2_r[MissingValues_r]=0;
    lat1_r[MissingValues_r]=0;
    lat2_r[MissingValues_r]=0;
    (_, _, dist_r1) = pyproj.Geod(ellps=sphere_model_for_pyproj).inv(lon1, lat1, lon1_r, lat1_r) 
    (_, _, dist_r2) = pyproj.Geod(ellps=sphere_model_for_pyproj).inv(lon2, lat2, lon2_r, lat2_r) 

    DIST_R1 = np.ones_like(met_all_leg0.latitude)*np.nan

    DIST_R1[1:-1] = np.nanmax([dist_r1[0:-1], dist_r1[1:]],axis=0).transpose(); # calculate sog as avg of vel(jj-1,jj) and vel(jj,jj+1)
    DIST_R1[0] = dist_r1[0]; DIST_R1[-1] = dist_r1[-1]; # fill first and last reading
    
    DIST_R2 = np.ones_like(met_all_leg0.latitude)*np.nan

    DIST_R2[1:-1] = np.nanmax([dist_r2[0:-1], dist_r2[1:]],axis=0).transpose(); # calculate sog as avg of vel(jj-1,jj) and vel(jj,jj+1)
    DIST_R2[0] = dist_r2[0]; DIST_R2[-1] = dist_r2[-1]; # fill first and last reading
    
    
    vel = np.true_divide(dist,dt)
    vel[MissingValues] = np.nan
    print(str(np.sum(vel>(12)))+'samples with velocity larger than 12 m/s') # give this more tolerance
    vel[(vel>(12))]=np.nan # cut unrealistic velocities (vel > 12 m/s)
    vel[(dt>(1*50))]=np.nan # set time diffs with dt>50sec to NaN to  (don't trust edge of data gap velocities)
    vel[(dist>300)]=np.nan # was 200
    vel[(dist_r1>600)]=np.nan
    vel[(dist_r2>600)]=np.nan

    vel[np.abs(dt-np.round(dt))>0.015] = np.nan # these seam to be very noisy in position
    
    print(np.nanmean(vel))
    
    print(np.nanmean(vel))
    evel = np.cos((-az12+90)*np.pi/180)*vel;
    nvel = np.sin((-az12+90)*np.pi/180)*vel;
    
    timest_vel = met_all_leg0.index[1:]+pd.to_timedelta(dt*0.5, unit='s') # just to have a time stamp to plot vel against
    
    # note SOG, COG are recomputed from velEast velNorth below!
    SOG[1:-1] = np.nanmean([vel[0:-1], vel[1:]],axis=0); # calculate sog as avg of vel(jj-1,jj) and vel(jj,jj+1)
    SOG[0] = vel[0]; SOG[-1] = vel[-1]; # fill first and last reading

    COG[1:-1] = np.rad2deg((np.arctan2( np.nanmean([vel[0:-1]*np.sin(np.deg2rad(az12[0:-1])) , vel[1:]*np.sin(np.deg2rad(az12[1:])) ], axis=0), np.nanmean([vel[0:-1]*np.cos(np.deg2rad(az12[0:-1])) , vel[1:]*np.cos(np.deg2rad(az12[1:])) ], axis=0) )  + 2 * np.pi) % (2 * np.pi) )
    COG[0] = az12[0]; COG[-1] = az12[-1]; # fill first and last reading

    velEast[1:-1] = np.nanmean([evel[0:-1], evel[1:]],axis=0);
    velEast[0] = evel[0]; velEast[-1] = evel[-1]; 
    velNorth[1:-1] = np.nanmean([nvel[0:-1], nvel[1:]],axis=0);
    velNorth[0] = nvel[0]; velNorth[-1] = nvel[-1]; 
    
    #met_all_leg0 = met_all_leg0.assign( DIST_R1 = DIST_R1)
    #met_all_leg0 = met_all_leg0.assign( DIST_R2 = DIST_R2)

    
    met_all_leg0 = met_all_leg0.assign( SOG = SOG)
    met_all_leg0 = met_all_leg0.assign( COG = COG)
    met_all_leg0 = met_all_leg0.assign( velEast = velEast)
    met_all_leg0 = met_all_leg0.assign( velNorth = velNorth)
    
    lon_median=met_all_leg0['longitude'].resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5)) ).median() # calculate 6 second mean to match wind data

    SOG = met_all_leg0.SOG.copy();
    SOG_MAX=SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5))).max()
    SOG_MIN=SOG.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5))).min()


    met_all_leg0=met_all_leg0.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*.5)) ).mean() # calculate 1min mean to match wind data
    met_all_leg0 = met_all_leg0.assign( SOG_DIFF = (SOG_MAX-SOG_MIN))
    # here we fix dateline issues of averaging longitudes around +/-180 degree
    # this is a bit coarse, but it works.
    lon_flip_tollerance = 0.0005 # a value well aboved the normal difference of mean and median longitudes
    #met_all_leg0['longitude'][np.abs(met_all_leg0.longitude-lon_median)>lon_flip_tollerance]=lon_median[np.abs(met_all_leg0.longitude-lon_median)>lon_flip_tollerance]

    met_all_leg0 = met_all_leg0.assign( COG = ((90-np.rad2deg(np.arctan2(met_all_leg0.velNorth,met_all_leg0.velEast))) % 360) ) # recompute COG from averaged North/Easte velocities

    met_all_leg0.SOG = np.sqrt(np.square(met_all_leg0.velNorth)+np.square(met_all_leg0.velEast) ) # vector average velocity

    bad_lad_long = ((met_all_leg0.latitude>22)&(met_all_leg0.latitude<26)&(met_all_leg0.longitude>-14)&(met_all_leg0.longitude<2))
    met_all_leg0.at[bad_lad_long, 'latitude'] = np.nan
    met_all_leg0.at[bad_lad_long, 'longitude'] = np.nan
    
    bad_lad_long = ((met_all_leg0.latitude>3)&(met_all_leg0.latitude<5)&(met_all_leg0.longitude>-18)&(met_all_leg0.longitude<-17))
    met_all_leg0.at[bad_lad_long, 'latitude'] = np.nan
    met_all_leg0.at[bad_lad_long, 'longitude'] = np.nan
    bad_lad_long = ((met_all_leg0.latitude>-8)&(met_all_leg0.latitude<5)&(met_all_leg0.longitude>-6.5)&(met_all_leg0.longitude<-4.4))
    met_all_leg0.at[bad_lad_long, 'latitude'] = np.nan
    met_all_leg0.at[bad_lad_long, 'longitude'] = np.nan
    
    met_all_leg0.at[np.isnan(met_all_leg0.SOG), 'latitude'] = np.nan # latitude/longitude are also scattering for SOG outliers
    met_all_leg0.at[np.isnan(met_all_leg0.SOG), 'longitude'] = np.nan # latitude/longitude are also scattering for SOG outliers
    met_all_leg0.at[np.isnan(met_all_leg0.SOG), 'COG'] = np.nan
    met_all_leg0.at[np.isnan(met_all_leg0.SOG), 'velEast'] = np.nan
    met_all_leg0.at[np.isnan(met_all_leg0.SOG), 'velNorth'] = np.nan
    met_all_leg0.at[np.isnan(met_all_leg0.SOG), 'SOG_DIFF'] = np.nan

    # additional velocity filtering based on deviation from rolling window of 10minutes:
    # here we need to interpolated accross nans
    SPIKES = np.abs(met_all_leg0.SOG-met_all_leg0.SOG.interpolate().rolling(window=10, center=True).mean())>.5
    for var_str in ['SOG','COG','velEast','velNorth','SOG_DIFF']:
        met_all_leg0.at[SPIKES, var_str] = np.nan


    met_all_leg0.drop(columns=['TIMEDIFF'], inplace=True)
    met_all_leg0.index.name='date_time'
    
    return met_all_leg0

def combine_ship_tracks():
    """
        Function to combine the ship track data from the three sources
        - quality checked one-second data
        - uncorreceted one-second data from inertial navigation data stream
        - thirty-second resolution GPS record from the raw meteorolocical data

        :returns: data frame containing the ships track and velocity at 1 minute resolution for legs 0 to 4
    """
    df_gps = minute_track_from_1sec_track(); print("minute_track_from_1sec_track")
    met_all_leg0 = minute_track_from_30sec_data_leg0(); print("minute_track_from_30sec_data_leg0")
    hydrins = minute_track_from_hydrins_data();  print("minute_track_from_hydrins_data")
    
    if USE_HYDRINGS:
        df_gps = pd.concat([hydrins[(hydrins.index<pd.to_datetime("2016-12-21 00:00:00", format="%Y-%m-%d %H:%M:%S").tz_localize(tz='UTC')) ],  df_gps], sort=True)

        df_gps = pd.concat([met_all_leg0[(met_all_leg0.index<hydrins.index[0]) ],  df_gps], sort=True)
    else:
        df_gps = pd.concat([met_all_leg0[(met_all_leg0.index<df_gps.index[0]) ],  df_gps], sort=True)
    
    # use 1/30sec coordinates if 1/1sec coordinates are not available
    df_gps.at[(np.isnan(df_gps.longitude) & ~np.isnan(met_all_leg0.longitude)), 'longitude'] = met_all_leg0.longitude[(np.isnan(df_gps.longitude) & ~np.isnan(met_all_leg0.longitude))]
    df_gps.at[(np.isnan(df_gps.latitude) & ~np.isnan(met_all_leg0.latitude)), 'latitude'] = met_all_leg0.latitude[(np.isnan(df_gps.latitude) & ~np.isnan(met_all_leg0.latitude))]
    
    return df_gps


def merge_wind_and_gps():
    """
        Function to combine the ship track data and the wind speed record at 1 minute resolution
        
        :returns: data frame containing the ships track and velocity as well as the wind observations averaged at 1 minute resolution for legs 0 to 4
    """
    df_gps = combine_ship_tracks();  print("combine_ship_tracks")
    df_wind = read_and_filter_wind_data();  print("read_and_filter_wind_data")
    if USE_HYDRINGS:
        hydrins = minute_track_from_hydrins_data();  print("minute_track_from_hydrins_data")
        
    ###########################
    # here insert Airflow Distortion Correction
    ###########################
    
    ### produce 1min merged wind&gps time series
    # calculate uR,vR and u,v from SR, DR and WS,WR
    # uR are positive for wind along ships main axis
    # vR are positive for wind in port direction
    # u are positive for wind blowing East
    # v are positive for wind blowing North
    df_wind = df_wind.assign(uR1=df_wind.WSR1*np.cos(np.deg2rad(180-df_wind.WDR1)))
    df_wind = df_wind.assign(vR1=df_wind.WSR1*np.sin(np.deg2rad(180-df_wind.WDR1)))
    df_wind = df_wind.assign(u1=df_wind.WS1*np.cos(np.deg2rad(270-df_wind.WD1)))
    df_wind = df_wind.assign(v1=df_wind.WS1*np.sin(np.deg2rad(270-df_wind.WD1)))
    #
    df_wind = df_wind.assign(uR2=df_wind.WSR2*np.cos(np.deg2rad(180-df_wind.WDR2)))
    df_wind = df_wind.assign(vR2=df_wind.WSR2*np.sin(np.deg2rad(180-df_wind.WDR2)))
    df_wind = df_wind.assign(u2=df_wind.WS2*np.cos(np.deg2rad(270-df_wind.WD2)))
    df_wind = df_wind.assign(v2=df_wind.WS2*np.sin(np.deg2rad(270-df_wind.WD2)))

    # add heading sin cos for proper averaging
    df_wind = df_wind.assign(hdg_cos=np.cos(np.deg2rad(df_wind.HEADING)))
    df_wind = df_wind.assign(hdg_sin=np.sin(np.deg2rad(df_wind.HEADING)))

    # assing apparent wind in eart,north coordinates
    df_wind = df_wind.assign(uA1=df_wind.WSR1*np.cos(np.deg2rad(270-df_wind.HEADING-df_wind.WDR1)))
    df_wind = df_wind.assign(vA1=df_wind.WSR1*np.sin(np.deg2rad(270-df_wind.HEADING-df_wind.WDR1)))
    df_wind = df_wind.assign(uA2=df_wind.WSR2*np.cos(np.deg2rad(270-df_wind.HEADING-df_wind.WDR2)))
    df_wind = df_wind.assign(vA2=df_wind.WSR2*np.sin(np.deg2rad(270-df_wind.HEADING-df_wind.WDR2)))


    # Derive Velocity estimates from relative and true wind speeds
    # this assumes COG==HEADING
    WS=df_wind.WS1; WD=df_wind.WD1; WSR=df_wind.WSR1; WDR=df_wind.WDR1; HDG=df_wind.HEADING
    df_wind = df_wind.assign(velEast1 = WS*np.cos(np.deg2rad(270-WD))-WSR*np.cos(np.deg2rad(270-HDG-WDR)) )
    df_wind = df_wind.assign(velNorth1 = WS*np.sin(np.deg2rad(270-WD))-WSR*np.sin(np.deg2rad(270-HDG-WDR)) )
    WS=df_wind.WS2; WD=df_wind.WD2; WSR=df_wind.WSR2; WDR=df_wind.WDR2;
    df_wind = df_wind.assign(velEast2 = WS*np.cos(np.deg2rad(270-WD))-WSR*np.cos(np.deg2rad(270-HDG-WDR)) )
    df_wind = df_wind.assign(velNorth2 = WS*np.sin(np.deg2rad(270-WD))-WSR*np.sin(np.deg2rad(270-HDG-WDR)) )


    nWSR1=df_wind.WSR1.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).count() # calculate 6 second average
    nWSR2=df_wind.WSR2.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).count() # calculate 6 second average
    #plt.plot(nWSR1,'.')
    # strangely some bins contain more than nSec/3 wind samples!!!


    if 1: # compute HEADING DIFF accross 1min interval
        HEADING = df_wind.HEADING.copy();
        HEADING_MAX=HEADING.resample(str(merge_at_nseconds)+'S').max()
        HEADING_MIN=HEADING.resample(str(merge_at_nseconds)+'S').min()
        HEADING = (HEADING-180)%360
        HEADING_MAX_=HEADING.resample(str(merge_at_nseconds)+'S').max()
        HEADING_MIN_=HEADING.resample(str(merge_at_nseconds)+'S').min()
        HEADING_DIFF = np.min([(HEADING_MAX-HEADING_MIN), (HEADING_MAX_-HEADING_MIN_)], axis=0)


    # resample to 1minute resolution    
    df_wind=df_wind.resample(str(merge_at_nseconds)+'S', loffset = datetime.timedelta(seconds=(merge_at_nseconds*0.5))).mean() # calculate 6 second average

    if 1:
        df_wind = df_wind.assign(HEADING_DIFF=HEADING_DIFF)

        df_wind = df_wind.assign(nWSR1=nWSR1)
        df_wind = df_wind.assign(nWSR2=nWSR2)

    if 1: 
        # set data with too few readings to NaN
        nSample_min = 6 # out of 20 possible (usually 12+/2)

        print( 'WSR1 removing ' + str( np.sum((nWSR1>0) & (nWSR1<nSample_min)) ) + ' of '+str(np.sum(nWSR1>0)) )
        print( 'WSR2 removing ' + str( np.sum((nWSR2>0) & (nWSR2<nSample_min)) ) + ' of '+str(np.sum(nWSR1>0)) )

        df_wind.at[nWSR1<nSample_min, 'uR1'] = np.nan
        df_wind.at[nWSR1<nSample_min, 'vR1'] = np.nan
        df_wind.at[nWSR1<nSample_min, 'uA1'] = np.nan
        df_wind.at[nWSR1<nSample_min, 'vA1'] = np.nan
        df_wind.at[nWSR2<nSample_min, 'uR2'] = np.nan
        df_wind.at[nWSR2<nSample_min, 'vR2'] = np.nan
        df_wind.at[nWSR2<nSample_min, 'uA2'] = np.nan
        df_wind.at[nWSR2<nSample_min, 'vA2'] = np.nan

    # rebuild the angles:
    df_wind.HEADING = np.rad2deg(np.arctan2(df_wind.hdg_sin, df_wind.hdg_cos)) % 360
    df_wind.WD1 = (270 - np.rad2deg(np.arctan2(df_wind.v1, df_wind.u1)) )% 360
    df_wind.WDR1 = (180 - np.rad2deg(np.arctan2(df_wind.vR1, df_wind.uR1)) )% 360
    df_wind.WD2 = (270 - np.rad2deg(np.arctan2(df_wind.v2, df_wind.u2)) )% 360
    df_wind.WDR2 = (180 - np.rad2deg(np.arctan2(df_wind.vR2, df_wind.uR2)) )% 360
    # recalcualte the speeds as vector average
    df_wind.WS1 = np.sqrt( np.square(df_wind.v1) + np.square(df_wind.u1) )
    df_wind.WS2 = np.sqrt( np.square(df_wind.v2) + np.square(df_wind.u2) )
    df_wind.WSR1 = np.sqrt( np.square(df_wind.vR1) + np.square(df_wind.uR1) )
    df_wind.WSR2 = np.sqrt( np.square(df_wind.vR2) + np.square(df_wind.uR2) )

    ########################
    # merge wind with GPS track/velocity
    df_wind = df_wind.merge(df_gps, left_on='date_time', right_on='date_time', how='left') # was inner
    ########################

    df_wind.COG = (90-np.rad2deg(np.arctan2(df_wind.velNorth,df_wind.velEast))) % 360 # recompute COG from averaged North/Easte velocities
    df_wind.SOG = np.sqrt(np.square(df_wind.velNorth)+np.square(df_wind.velEast) ) # vector average velocity

    
    # re-calculate SOG COG from True and Relative wind speed/direction
    # This assumes HDG==COG and is only approximately correct for SOG > 2m/s
    df_wind = df_wind.assign(COG1 = (90 - np.rad2deg( np.arctan2( df_wind.velNorth1,df_wind.velEast1 ) ) ) % 360 )
    df_wind = df_wind.assign(COG2 = (90 - np.rad2deg( np.arctan2( df_wind.velNorth2,df_wind.velEast2 ) ) ) % 360 )
    df_wind = df_wind.assign(SOG1 = np.sqrt( np.square(df_wind.velNorth1) + np.square(df_wind.velEast1) ) )
    df_wind = df_wind.assign(SOG2 = np.sqrt( np.square(df_wind.velNorth2) + np.square(df_wind.velEast2) ) )

    # filter the wind vector derived velocities for outliers
    # additional velocity filtering based on deviation from rolling window of 10minutes:
    # here we need to interpolated accross nans
    SPIKES1 = np.abs(df_wind.SOG1-df_wind.SOG1.interpolate().rolling(window=10, center=True).mean())>.5
    for var_str in ['SOG1','COG1','velEast1','velNorth1']:
        df_wind.at[SPIKES1, var_str] = np.nan
    SPIKES2 = np.abs(df_wind.SOG2-df_wind.SOG2.interpolate().rolling(window=10, center=True).mean())>.5
    for var_str in ['SOG2','COG2','velEast2','velNorth2']:
        df_wind.at[SPIKES2, var_str] = np.nan

    SOG_W = np.nanmean([df_wind.SOG1,df_wind.SOG2],axis=0)

    ### 
    ### set a label for the velocity source 
    # flag VELOCITY_SOURCE signifies where the velocity data is from 0=NaN, 1=1/sec-GPS, 3=1/3sec-wind data, 30=1/30sec GPS data
    df_wind = df_wind.assign(VELOCITY_SOURCE = 1)
    
    df_wind.at[~np.isnan(df_wind.SOG), 'VELOCITY_SOURCE'] = 1 # first guess this is 1/sec data
    df_wind.at[np.isnan(df_wind.SOG), 'VELOCITY_SOURCE'] = 0 # no data
    df_wind.at[( np.isnan(df_wind.SOG) & ~np.isnan(SOG_W) & (SOG_W>2) ), 'VELOCITY_SOURCE'] = 3 # use 1/3 sec data if no 1/sec data and SOG>1

    if USE_HYDRINGS:
        # special case before 
        df_wind.at[((df_wind.index<hydrins.index[0]) & ~np.isnan(df_wind.SOG)) , 'VELOCITY_SOURCE'] = 30 # if no 1/sec data available use 1/30 sec data
        df_wind.at[( (df_wind.index<hydrins.index[0]) & ~np.isnan(SOG_W) & (SOG_W>2) ) , 'VELOCITY_SOURCE'] = 3 # 1/3 sec data is better than 1/30 sec data for SOG>2

    # apply the use of the 1/3 sec wind derived velocities
    for var_str in ['velEast','velNorth','SOG']:
        df_wind.at[df_wind['VELOCITY_SOURCE']==3, var_str] = np.nanmean([df_wind[var_str+'1'],df_wind[var_str+'2']],axis=0)[df_wind['VELOCITY_SOURCE']==3]
    df_wind.COG = (90-np.rad2deg(np.arctan2(df_wind.velNorth,df_wind.velEast))) % 360 # recompute COG from averaged North/Easte velocities

    # clean up (remove the extraneous columns)
    df_wind.drop(columns=['velEast1', 'velEast2', 'velNorth1', 'velNorth2', 'SOG1', 'COG1', 'SOG2', 'COG2'], inplace=True)

    ### True wind correction using GPS data

    # use average apparent wind for motion correction
    df_wind.u1 = df_wind.uA1 + df_wind.velEast
    df_wind.v1 = df_wind.vA1 + df_wind.velNorth
    df_wind.u2 = df_wind.uA2 + df_wind.velEast
    df_wind.v2 = df_wind.vA2 + df_wind.velNorth

    # rebuild the angles:
    df_wind.WD1 = (270 - np.rad2deg(np.arctan2(df_wind.v1, df_wind.u1)) )% 360
    df_wind.WD2 = (270 - np.rad2deg(np.arctan2(df_wind.v2, df_wind.u2)) )% 360
    # recalcualte the speeds as vector average
    df_wind.WS1 = np.sqrt( np.square(df_wind.v1) + np.square(df_wind.u1) )
    df_wind.WS2 = np.sqrt( np.square(df_wind.v2) + np.square(df_wind.u2) )

    
    return df_wind


def dirdiff(HEADING,Nmin,loffset):
    """
        Function to calculate the maximum difference of a [0, 360) direction during specified time averging intervals
        
        :param HEADING: time series of a direction in degrees [0, 360)
        :param Nmin: integer specifying the number of minutes to average
        :param loffset: float specifying the number of minutes to apply as ofset to the out put of pandas.resample
        
        returns: HEADING_DIFF: time series of the maximal difference between the direction estimates during the specified averaging interval
               
    """
    HEADING_MAX=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING = (HEADING-180)%360
    HEADING_MAX_=HEADING.resample(str(Nmin)+'T', loffset = loffset).max()
    HEADING_MIN_=HEADING.resample(str(Nmin)+'T', loffset = loffset).min()
    HEADING_DIFF = np.min([(HEADING_MAX-HEADING_MIN), (HEADING_MAX_-HEADING_MIN_)], axis=0)
    return HEADING_DIFF



if __name__ == "__main__":
    
    track = merge_wind_and_gps();  print("merge_wind_and_gps")
    
    track['longitude'] = ang180(track['longitude'])
    
    #track.to_csv('./data/ship_track_velocity_onemin/cruise-track-1min-legs0-4.csv', columns=['latitude','longitude','velEast','velNorth','HEADING', 'COG', 'SOG','VELOCITY_SOURCE']); print("safe 1 minute data")
    # rename the columns to conform with CF standard
    track_CF = track[['latitude','longitude','velEast','velNorth','HEADING', 'COG', 'SOG','VELOCITY_SOURCE']].copy()
    track_CF = track_CF.rename(columns={'velEast':'platform_speed_wrt_sea_water_east','velNorth':'platform_speed_wrt_sea_water_north','HEADING':'platform_orientation', 'COG':'platform_course', 'SOG':'platform_speed_wrt_ground'})
    track_CF.to_csv('./data/ship_track_velocity_onemin/cruise-track-1min-legs0-4.csv',date_format="%Y-%m-%dT%H:%M:%S+00:00",na_rep="NaN")
    
    # we also create a 5-minute version
    Nmin = 5
    loffset = datetime.timedelta(minutes=0.5*Nmin)
    lon_flip_tollerance=0.0005 # to detect where the longitude averaging goes accross the phase shift
    
    HEADING_DIFF = dirdiff(track.HEADING.copy(),Nmin,loffset)
    SOG = track.SOG.copy();
    SOG_MAX=SOG.resample(str(Nmin)+'T', loffset = loffset).max()
    SOG_MIN=SOG.resample(str(Nmin)+'T', loffset = loffset).min()

    #lon_median = wind_5min.longitude.copy();
    lon_median=ang180( track['longitude'].resample(str(Nmin)+'T', loffset = loffset).median() ) # median of longitudes

    track = track.resample(str(Nmin)+'T', loffset = loffset).mean()
    
    track = track.assign(HEADING_DIFF = HEADING_DIFF)
    track = track.assign(SOG_DIFF = ( SOG_MAX-SOG_MIN) )
    
    track.COG = (90-np.rad2deg(np.arctan2(track.velNorth,track.velEast))) % 360 # recompute COG from averaged North/Easte velocities
    track.HEADING = np.rad2deg(np.arctan2(track.hdg_sin, track.hdg_cos)) % 360
    track.SOG = np.sqrt(np.square(track.velNorth)+np.square(track.velEast) ) # vector average velocity

    
    track['VELOCITY_SOURCE'] = np.round(track['VELOCITY_SOURCE'])
    track.at[(track['VELOCITY_SOURCE']>15), 'VELOCITY_SOURCE'] = 30
    track.at[(track['VELOCITY_SOURCE']<=15), 'VELOCITY_SOURCE'] = 3
    
    track['longitude'] = ang180(track['longitude']) # enusure longitude within [-180 +180]
    
    track.at[(np.abs(track.longitude-lon_median)>lon_flip_tollerance), 'longitude']=lon_median[np.abs(track.longitude-lon_median)>lon_flip_tollerance]
    
    # recorde which coordinates where not measured
    track = track.assign(COORDINATES_INTERPOLATED = np.isnan(track.longitude*track.latitude).astype(int) )

    
    track.latitude = track.latitude.interpolate()
    track.longitude = ang180(track.longitude.interpolate())

    #track.to_csv('./data/ship_track_velocity_fivemin/cruise-track-5min-legs0-4.csv', columns=['latitude','longitude','velEast','velNorth','HEADING', 'COG', 'SOG','VELOCITY_SOURCE','COORDINATES_INTERPOLATED'])
    # rename the columns to conform with CF standard
    track_CF = track[['latitude','longitude','velEast','velNorth','HEADING', 'COG', 'SOG','VELOCITY_SOURCE','COORDINATES_INTERPOLATED']].copy()
    track_CF = track_CF.rename(columns={'velEast':'platform_speed_wrt_sea_water_east','velNorth':'platform_speed_wrt_sea_water_north','HEADING':'platform_orientation', 'COG':'platform_course', 'SOG':'platform_speed_wrt_ground'})
    track_CF.to_csv('./data/ship_track_velocity_fivemin/cruise-track-5min-legs0-4.csv',date_format="%Y-%m-%dT%H:%M:%S+00:00",na_rep="NaN")


    
    
    
    
    