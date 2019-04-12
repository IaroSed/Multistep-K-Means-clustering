def SingleCountry_clusters(min_in_cl, max_in_cl, country):

    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    import statistics
    
    
    
    MAX_IN_CLUSTER = max_in_cl
    MIN_IN_CLUSTER = min_in_cl
    
    
    
    #Creating dictionnaries

    counter = 0
    
    industry_coding = {'Automotive': 1, 'Manufacturing & Resources': 10, 'Financial Services': 25, 'Government' : 40, 'Health' : 55, 'Media & Communications' : 70,
                       'Other Commercial Industries' : 90, 'Retail' : 99}
    
    
    
    def getnewqueries(server,db,query):
        ''' Extracting new source and destination information from SQL Server
        @params:
            server   - Required  :  SQL Server name (Str)
            db:      - Required  :  Data Base name (Str)
            query    - Required  : SQL query (Str)
        '''
            
        import sqlalchemy
    #    import pandas as pd
            
        encoding='utf-8'
        driver = 'SQL+Server'      
        engine = sqlalchemy.create_engine('mssql+pyodbc://{}/{}?driver={}?encoding={}'.format(server, db, driver, encoding))
    
        NewQueries = pd.read_sql(query,con=engine)
        

        NewQueries['Strategic_Coded'] = NewQueries['Strategic_flag']
        NewQueries['Industry_Coded'] = NewQueries['Custom Industry'].map(industry_coding)
            
        #Creating additional columns: Key by concatenating Source and Destination, TravelDuration and TravelDistance
        new = pd.DataFrame({'TPID': NewQueries['TPID'],
                            'Latitude': NewQueries['Latitude'],
                            'Longitude': NewQueries['Longitude'],
                            'Strategic_Coded': NewQueries['Strategic_Coded'],
                            'Industry_Coded': NewQueries['Industry_Coded']}).fillna(0)
        #.dropna()
        
        
        return new
    
    
    
    def storeprep(data,counter,level):
        
        if level == 0:
        
            new_tostore = pd.DataFrame({'TPID': data['TPID'],
                                        'Latitude': data['Latitude'],
                                        'Longitude': data['Longitude'],
                                        'Strategic_Coded': data['Strategic_Coded'],
                                        'Industry_Coded': data['Industry_Coded'],
                                        'K-Means Territory': data['K-Means Territory'],
                                        'Country': data['Country'],
                                        'Global territory': data['Country'] + str(counter)
                                        })
            counter += 1
        
        elif level == 1:
            new_tostore = pd.DataFrame({'TPID': data['TPID'],
                                        'Latitude': data['Latitude'],
                                        'Longitude': data['Longitude'],
                                        'Strategic_Coded': data['Strategic_Coded'],
                                        'Industry_Coded': data['Industry_Coded'],
                                        'K-Means Territory': data['K-Means Territory'],
                                        'Country': data['Country'],
                                        'Global territory': data['Country'].str.cat(others=(str(counter)+ "big" + data['K-Means Territory'].astype('str')),sep=' ')
                                        })
            counter += 1
        
        return new_tostore, counter
        
    def storequeries(server, db, data):
        '''Storing the results and errors in SQL
        @params: 
        server   - Required  :  SQL Server name (Str)
        db:      - Required  :  Data Base name (Str)
        data:    - Required  :  Data to be stored (Pandas dataframe)
        '''
            
        import sqlalchemy
            
        encoding='utf-8'
        driver = 'SQL+Server'
            
        engine = sqlalchemy.create_engine('mssql+pyodbc://{}/{}?driver={}?encoding={}'.format(server, db, driver,encoding))
            
        data.to_sql('WW_sp', con=engine, if_exists='append', index=False)
    
    def calculate_wcss(data_in, columns):
        #Calculating WCSS
        MAX_CLUSTERS = MIN_IN_CLUSTER
            
        wcss = []
        for i in range(1, MAX_CLUSTERS+1):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
            kmeans.fit(data_in.loc[:, columns].values)
            wcss.append(kmeans.inertia_)
                
        return wcss
        
    def optimal_n_clusters(wcss):
        
        #Normalizing wcss before proceding
        mu = sum(wcss)/len(wcss)
        std = statistics.stdev(wcss)  
        wcss = (wcss - mu)/std
        
        #Choose the optimal number clusters:
        opt_clust = 0
        d_max = 0
        bc = np.array([len(wcss) - 1, wcss[len(wcss)-1] - wcss[0]])
        bc_n = np.linalg.norm(bc)
        for i in range(2, len(wcss) + 1):
            ba = np.array([i - 1, wcss[i-1] - wcss[0]])
            d = np.linalg.norm(ba)**2 - (sum(ba*bc)/bc_n)**2
            if (d > d_max):
                d_max = d
                opt_clust = i  
        return opt_clust       
    
    def fitting_dataset(data_in, opt_clust, columns):
        
        # Fitting K-Means to the dataset
        kmeans = KMeans(n_clusters = opt_clust, init = 'k-means++', random_state = 0)
            
        assignments = pd.DataFrame({'K-Means Territory': kmeans.fit_predict(data_in.loc[:, columns].values)}).reset_index()
      
            
        data_out = pd.DataFrame({'TPID': data_in['TPID'],
                                'Latitude': data_in['Latitude'],
                                'Longitude': data_in['Longitude'],
                                'Strategic_Coded': data_in['Strategic_Coded'],
                                'Industry_Coded': data_in['Industry_Coded'],
                                'K-Means Territory': assignments['K-Means Territory'],
                                'Country': str(country)
                                })
    
        
        return data_out, list(data_out.groupby('K-Means Territory').count()['TPID'])
    
    
    def fitting_dataset_balanced(data_in_b, columns, min_size):
    
        #Calculating WCSS
        wcss = calculate_wcss(data_in_b, columns)
    
        #Choose the optimal number clusters
        opt_clust = optimal_n_clusters(wcss)
    
        # Fitting K-Means to the dataset
        data_out_b, freq = fitting_dataset(data_in_b.reset_index(), opt_clust, columns)
        
        strategic = list(data_out_b.groupby('K-Means Territory').sum()['Strategic_Coded'])
        
        clust_size =[]
        for i in range(0,len(freq)):
            clust_size.append(freq[i] + 4*strategic[i])
        
        while min(clust_size) < min_size:
            opt_clust -= 1
    
            data_out_b, freq = fitting_dataset(data_in_b.reset_index(), opt_clust, columns)
            #print(freq)
            
            strategic = list(data_out_b.groupby('K-Means Territory').sum()['Strategic_Coded'])
            
            clust_size =[]
            for i in range(0,len(freq)):
                clust_size.append(freq[i] + 4*strategic[i])
            
    
        return data_out_b, freq
    
    
    def fitting_dataset_feature(columns,data,big_list,counter):
       
        # Fitting K-Means to the dataset in a balanced way
        new_out, freq = fitting_dataset_balanced(data, columns, MIN_IN_CLUSTER)
        
        strategic = list(new_out.groupby('K-Means Territory').sum()['Strategic_Coded'])
        
        
        # 1 stratgic counts as 5 penetrated
        ok_clusters = [index for index,value in enumerate(freq) if value - strategic[index] + 5*strategic[index] < MAX_IN_CLUSTER]
        big_clusters = [index for index,value in enumerate(freq) if value - strategic[index] + 5*strategic[index]  > MAX_IN_CLUSTER]
    
    
        for cc in ok_clusters:
            
            is_cluster_cc =  new_out['K-Means Territory']==cc
            new_ok = new_out.loc[is_cluster_cc].reset_index()

            #Preparing to store
            new_tostore,counter = storeprep(new_ok,counter, 0)
            #Storing in SQL
            storequeries(server, db, new_tostore)
    
        for cnc in big_clusters:

            is_cluster_cnc =  new_out['K-Means Territory']==cnc
            new_big = new_out.loc[is_cluster_cnc].reset_index()
            
            big_list.append(new_big)
            
        return big_list, counter
    
    
    ## End of functions
    
    
    
    ## Main loop start
        
    server = 'IAROLAPTOP\IAROSQLSERVER'
    db = 'IARODB'
    query ="SELECT [TPID],[Latitude],[Longitude],[Strategic_flag], [Custom Industry]   FROM [dbo].[WW_data] WHERE [Country] = '"+str(country)+"'"
    
    new = getnewqueries(server, db, query)
    
    ColumnsList = [['Latitude', 'Longitude'],\
                   ['Latitude', 'Longitude'],\
                   ['Latitude', 'Longitude'], \
                   ['Latitude', 'Longitude','Industry_Coded'], \
                   ['Latitude', 'Longitude','Industry_Coded'], \
                   ['Latitude', 'Longitude','Industry_Coded']]
    
    
    
    # 1st step
    
    new_big_l = []
    
    new_big_l,counter = fitting_dataset_feature(ColumnsList[0],new,new_big_l,counter)
    
    # End of 1st step
    
    #2nd step     
        
    new_big_l2 = []
    
    for i in range(0,len(new_big_l)):
        
        new_big_l2,counter = fitting_dataset_feature(ColumnsList[1],new_big_l[i],new_big_l2,counter)
    
    # End of 2nd step        
    
    #3rd step
    
    new_big_l3 = []
    
    for i in range(0,len(new_big_l2)):
        
        new_big_l3,counter = fitting_dataset_feature(ColumnsList[2],new_big_l2[i],new_big_l3,counter)
    
    #End of 3rd step
        
    #4th step    
        
    new_big_l4 = []
    
    for i in range(0,len(new_big_l3)):
        
        new_big_l4,counter = fitting_dataset_feature(ColumnsList[3],new_big_l3[i],new_big_l4,counter)        
            
    
    #End of 4th step 
        
    #5th step    
        
    new_big_l5 = []
    
    for i in range(0,len(new_big_l4)):
        
        new_big_l5,counter = fitting_dataset_feature(ColumnsList[4],new_big_l4[i],new_big_l5,counter)        
            
    
    #End of 5th step 
    
    big = 0
    
    for i in range(0,len(new_big_l5)):
        
        
        if new_big_l5[i]['TPID'].count() <  MIN_IN_CLUSTER:
            #Preparing to store
            new_tostore,counter = storeprep(new_big_l5[i],counter, 1)
            #Storing in SQL
            storequeries(server, db, new_tostore)
            big +=1
        else:
            
            rand_list = []
            rand_list_i = []
            c = True
            for j in range(0,new_big_l5[i]['TPID'].count()):
                
                rand_list.append(c)
                rand_list_i.append(not(c))
                c = not(c)
                
    
            #Preparing to store
            new_tostore,counter = storeprep(new_big_l5[i].loc[rand_list].reset_index(),counter, 0)
            #Storing in SQL
            storequeries(server, db, new_tostore)
            #Preparing to store
            new_tostore,counter = storeprep(new_big_l5[i].loc[rand_list_i].reset_index(),counter, 0)
            #Storing in SQL
            storequeries(server, db, new_tostore)
            big +=2
    
    
    return counter, 1-big/counter
    

# Code to find the optimal min and max
Exit_RS = open("Results country 3G 3I.txt", 'w',encoding='utf-8')

Countries = ['Country A']

for country in Countries:

    print(country)
    max_eff = 0
    i_eff = 0
    j_eff = 0
        
    for i in range(5,14):
        for j in range(15,35):
            print(i,j)
            try:
                total, eff = SingleCountry_clusters(i, j, country)
                print(i,j,total,eff)
                Exit_RS.write(str(country) + "^" + "i:" + '^' + str(i)  + '^' +  "j" + '^' + str(j)  + '^' +  "Clusters:" + '^' +  str(total) + '^' +  "Efficiency:" + '^' + str(eff)  + "\n")
        
            except:
                eff = 0
                print("error")
            if (eff>max_eff):
                max_eff = eff
                i_eff = i
                j_eff = j
                
                
    print(i_eff, j_eff, max_eff)
        
Exit_RS.close()

# Code to calculate the clusters with the best min and max
total, eff = SingleCountry_clusters(i_eff, j_eff, country)

