#from __future__ import division

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt




def infection_model(network, p, flights, start_node):

    #Extract data of first flight and last flight 
    

    #Create dataframe storing airport and infection time 
    airports = sorted(network.nodes())
    inf_time = np.full((len(airports),), np.inf)
    infection = pd.DataFrame({"Airport":airports, "InfectionTime": inf_time}) 
    
    #Set the infection time of first infected node:
    infection.InfectionTime[start_node] = flights.StartTime.min()
    #Loop over flights and start infection 
    for i in range(len(flights)):
        source = flights.Source[i]
        source_inf_time = infection.InfectionTime[source]
        if (source_inf_time < flights.StartTime[i]):
            random = np.random.rand()
            if random <= p:
                target = flights.Destination[i]
                target_cur_inf_time = infection.InfectionTime[target]
                target_new_inf_time = flights.EndTime[i]
                if target_new_inf_time < target_cur_inf_time:
                    infection.InfectionTime[target] = target_new_inf_time
    return infection


flights = pd.read_csv("./Data/events_US_air_traffic_GMT.txt", sep = " ")

#Read in network data
networkpath = "./Data/aggregated_US_air_traffic_network_undir.edg"
network = nx.read_weighted_edgelist(networkpath, nodetype = int )

start_node_0 = flights.Source[0] #Initiate the first infected node
flights = flights.sort_values("StartTime")
start_time = flights.StartTime.min() #First infected time
end_time = flights.EndTime.max()

def averaged_prevalence_visualization(network, flights, start_node, start, end, p, label):
    stepsize = 50    
    t = np.linspace(start, end, stepsize)  #To increase this stepsize
    p_t = np.zeros((stepsize,10), dtype=float)
    for k in range(0,10):
        infection_p = infection_model(network, p ,flights, start_node)
        for j in range (0,stepsize):
            count = (infection_p.InfectionTime < t[j]).sum()
            prob = float(count/len(infection_p))
            p_t[j,k] = prob
    prevalence = np.average(p_t, axis = 1)
    plt.plot(t, prevalence, label = label)


fig = plt.figure(figsize=(10,7))

for node in (0,4,41,100,200):
    start_node = flights.Source[node]
    averaged_prevalence_visualization(network, flights, start_node, start_time, end_time, 0.1, node)
plt.xlabel("Time")
plt.ylabel("Averaged prevalence for different nodes")
plt.legend()
plt.show()

n_repeat = 50
time_median = np.zeros((len(network),n_repeat), dtype=float)
for i in range(n_repeat):
    start_node = np.random.random_integers(0,len(network)) 
    infection_i = infection_model(network, 0.5 ,flights, start_node)
    time_median[:,i] = infection_i.InfectionTime
print(time_median)

betweenness = np.array(list(nx.betweenness_centrality(network).values()))
closeness = np.array(list(nx.closeness_centrality(network).values()))
kshell = np.array(list(nx.core_number(network).values()))
clustering_coef = np.array(list(nx.clustering(network).values()))    
degrees = np.array(list(nx.degree(network).values()))
strengths = np.array(list(nx.degree(network, weight = "weight").values()))
                   
y_values = [betweenness, closeness, kshell, clustering_coef, degrees, strengths]
y_labels = ["betweenness", "closeness", "kshell", "clustering_coef", "degrees", "strengths"]
time_median = np.average(time_median,axis = 1)




