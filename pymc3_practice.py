#!/usr/bin/env python
# coding: utf-8


import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import COMET.__main__

def read_args():
    '''modified from COMET's main implementation. Put here because COMET's
    version is not callable'''
    start_dt = datetime.datetime.now()
    start_time = time.time()
    print("Started on " + str(start_dt.isoformat()))
    
    args = init_parser(argparse.ArgumentParser(
        description=("Hypergeometric marker detection.")
    )).parse_args()
    
    output_path = args.output_path
    C = args.C
    K = args.K
    Abbrev = args.Abbrev
    Down = args.Down
    X = args.X
    L = args.L
    marker_file = args.marker
    tsne_file = args.vis
    cluster_file = args.cluster
    gene_file = args.g
    Trim = args.Trim
    count_data = args.Count
    tenx = args.tenx
    online = args.online
    skipvis = args.skipvis
    plot_pages = 30  # number of genes to plot (starting with highest ranked)

    csv_path = output_path + 'data/'
    vis_path = output_path + 'vis/'
    pickle_path = output_path + '_pickles/'
    try:
        os.makedirs(csv_path)
    except:
        os.system('rm -r ' + csv_path)
        os.makedirs(csv_path)

    try:
        os.makedirs(vis_path)
    except:
        os.system('rm -r ' + vis_path)
        os.makedirs(vis_path)

    try:
        os.makedirs(pickle_path)
    except:
        os.system('rm -r ' + pickle_path)
        os.makedirs(pickle_path)

    if Trim is not None:
        Trim = int(Trim)
    else:
        Trim = int(2000)
    if C is not None:
        C = abs(int(C))
    else:
        C = 1
    if X is not None:
        try:
            X = float(X)
        except:
            raise Exception('X param must be a number between 0 and 1')
        if X > 1:
            X = int(1)
        elif X <= 0:
            X = int(0)
        else:
            X = float(X)
        print("Set X to " + str(X) + ".")
    if L is not None:
        L = int(L)
        print("Set L to " + str(L) + ".")
    if K is not None:
        K = int(K)
    else:
        K = 2
    if K > 4:
        K = 4
        print('Only supports up to 4-gene combinations currently, setting K to 4')
    if count_data is not None:
        if count_data == str(True):
            count_data = 1
            print('Count Data')
        elif count_data == 'yes':
            count_data = 1
            print('Count Data')
        else:
            count_data = int(0)
    else:
        count_data = int(0)
    if tenx is not None:
        if tenx == str(True):
            tenx = int(1)
        elif tenx == 'yes':
            tenx = int(1)
        else:
            tenx = int(0)
    else:
        tenx = int(0)
    if online is not None:
        if online == str(True):
            online = int(1)
        elif online == 'yes':
            online = int(1)
        else:
            online = int(0)
    else:
        online = int(0)
    if skipvis is not None:
        if skipvis == str(True):
            skipvis = int(1)
        elif skipvis == 'yes':
            skipvis = int(1)
        else:
            skipvis = int(0)
    else:
        skipvis = int(0)
    print("Reading data...")
    if gene_file is None:
        (cls_ser, tsne, no_complement_marker_exp, gene_path) = read_data(
            cls_path=cluster_file,
            tsne_path=tsne_file,
            marker_path=marker_file,
            gene_path=None,
            D=Down,
            tenx=tenx,
            online=online,
            skipvis=skipvis)
    else:
        (cls_ser, tsne, no_complement_marker_exp, gene_path) = read_data(
            cls_path=cluster_file,
            tsne_path=tsne_file,
            marker_path=marker_file,
            gene_path=gene_file,
            D=Down,
            tenx=tenx,
            online=online,
            skipvis=skipvis)

'''got here'''
    print("Generating complement data...")
    marker_exp = hgmd.add_complements(no_complement_marker_exp)
    #throw out vals that show up in expression matrix but not in cluster assignments
    for ind,row in marker_exp.iterrows():
        if ind in cls_ser.index.values.tolist():
            continue
        else:
            marker_exp.drop(ind, inplace=True)
        #print(marker_exp.index.values.tolist().count(str(ind)))
        #print(marker_exp[index])
    #throw out gene rows that are duplicates and print out a message to user
    marker_exp.sort_values(by='cell',inplace=True)
    cls_ser.sort_index(inplace=True)

    # Process clusters sequentially
    clusters = cls_ser.unique()
    clusters.sort()
    cluster_overall=clusters.copy()

    #Only takes a certain number of clusters (cuts out smallest ones)
    if online == 1:
        max_clus_size = 15
        if len(clusters) <= max_clus_size:
            pass
        else:
            cls_helper = list(clusters.copy())
            cls_size_count = {}
            for item in cls_ser:
                if item in cls_size_count:
                    cls_size_count[item] = cls_size_count[item] + 1
                else:
                    cls_size_count[item] = 1
            for counted in cls_size_count:
                cls_size_count[counted] = cls_size_count[counted] / len(cls_ser)
            while len(cls_helper) > max_clus_size:
                lowest = 1
                place = 0
                for key in cls_size_count:
                    if cls_size_count[key] < lowest:
                        place = key
                        lowest = cls_size_count[key]
                cls_helper.remove(place)
                del cls_size_count[place]
            clusters = np.array(cls_helper)

            
    #Below could probably be optimized a little (new_clust not necessary),
    #cores is number of simultaneous threads you want to run, can be set at will
    cores = C
    cluster_number = len(clusters)
    # if core number is bigger than number of clusters, set it equal to number of clusters
    if cores > len(clusters):
        cores = len(clusters)
    #below loops allow for splitting the job based on core choice
    group_num  = math.ceil((len(clusters) / cores ))
    for element in range(group_num):
        new_clusters = clusters[:cores]
        print(new_clusters)
        jobs = []
        #this loop spawns the workers and runs the code for each assigned.
        #workers assigned based on the new_clusters list which is the old clusters
        #split up based on core number e.g.
        #clusters = [1 2 3 4 5 6] & cores = 4 --> new_clusters = [1 2 3 4], new_clusters = [5 6]
        for cls in new_clusters:
            p = multiprocessing.Process(target=process,
                args=(cls,X,L,plot_pages,cls_ser,tsne,marker_exp,gene_file,csv_path,vis_path,pickle_path,cluster_number,K,Abbrev,cluster_overall,Trim,count_data,skipvis))
            jobs.append(p)
            p.start()
        p.join()
        new_clusters = []
        clusters = clusters[cores:len(clusters)]

    end_time = time.time()

    

    # Add text file to keep track of everything
    end_dt = datetime.datetime.now()
    print("Ended on " + end_dt.isoformat())
    metadata = open(output_path + 'metadata.txt', 'w')
    metadata.write("Started: " + start_dt.isoformat())
    metadata.write("\nEnded: " + end_dt.isoformat())
    metadata.write("\nElapsed: " + str(end_dt - start_dt))
    #metadata.write("\nGenerated by COMET version " + conf.version)


    print('Took ' + str(end_time-start_time) + ' seconds')
    print('Which is ' + str( (end_time-start_time)/60 ) + ' minutes')
###############################################################################
###############################################################################


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# Initialize random number generator
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use('arviz-darkgrid')


# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2
X = np.vstack([X1,X2]).reshape(100,2)
# Simulate outcome variable
#Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma
Y = alpha + 5 * X[:,0] +  np.random.randn(size) * sigma
Y = (Y - Y.min()) / (Y.max() - Y.min())
Y = np.rint(Y)

basic_model = pm.Model()
with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=3)
    beta = pm.Beta("beta", alpha = 1/4, beta = 1/2,shape = 2)
    #sigma = pm.HalfNormal("sigma", sigma=1)
    
#     mu = alpha
#     # Expected value of outcome
#     for col in range(X.shape[1]):
#         mu = mu + beta[col] * X[:,col] 

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli("Y_obs", logit_p = alpha + X @ beta, observed=Y)


map_estimate = pm.find_MAP(model=basic_model)
map_estimate


with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)


az.plot_trace(trace);
az.summary(trace, round_to=2)

y1 = np.zeros(55).astype(int)
y1[30:] = 1
y2 = np.zeros(55)
xs = []
count = 0
for i in range(2):
     for j in range(2):
        #10 genes from each cluster group
        #meta cluster 1 - first three genes are upregulated
        if i == 0:
            
            #cluster 1 - gene 3 upregulated
            if j == 0:
                x = np.random.randn(10,8)
                x[:,:3] =  x[:,:3] + np.random.normal(2,1,3)
                x[:,2] = x[:,2] +  np.random.normal(7,1,10)
                xs.append(x)
                count += 10
            #cluster 2 - genes 4 upregulated
            if j == 1:
                x = np.random.randn(20,8)
                x[:,:3] =  x[:,:3] + np.random.normal(2,1,3)
                x[:,3] = x[:,3] +  np.random.normal(7,1,20)
                y2[count:count+20] = 1
                xs.append(x)
                count += 20
        #meta cluster 2 - last three genes are upregulated
        if i == 1:
            
            #cluster 3 gene 5 upregulated
            if j == 0:
                x = np.random.randn(15,8)
                x[:,5:8] =  x[:,5:8] + np.random.normal(2,1,3)
                x[:,4] = x[:,4] +  np.random.normal(7,1,15)
                y2[count:count+12] = 2
                xs.append(x)
                count += 12
            #cluster 4 - gene 6 upregulated
            if j == 1:
                x = np.random.randn(10,8)
                x[:,5:8] =  x[:,5:8] + np.random.normal(2,1,3)
                x[:,5] = x[:,5] +  np.random.normal(7,1,10)
                y2[count:] = 3
                xs.append(x)
X = np.vstack(xs)
c1 = np.array(y2 == 0).astype(float)
c2 = np.array(y2 == 1).astype(int)
c3 = np.array(y2 == 2).astype(int)
c4 = np.array(y2 == 3).astype(int)

basic_model = pm.Model()

with basic_model:
    #priors for random intercept
    mu_a = pm.Normal('mu_a', mu=0., sigma=2)
    sigma_a = pm.HalfNormal('sigma_a', 2)
    # Intercept for each cluster, distributed around group mean mu_a
    # Above we just set mu and sd to a fixed value while here we
    # plug in a common group distribution for all a and b (which are
    # vectors of length n_counties).
    alpha = pm.Normal('alpha', mu=mu_a, sigma=sigma_a,shape = 2)
    
    # Priors for unknown model parameters
    #beta = pm.Beta("beta", alpha=1/2, beta = 1/2, shape=8)
    #beta = pm.Laplace("beta", mu=0, b = 1, shape=8)
    beta = pm.Normal("beta", mu=0, sigma = 0.5, shape=8)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli("Y_obs", logit_p=alpha[y1] + X @ beta, observed=c4)



map_estimate = pm.find_MAP(model=basic_model)
map_estimate


with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)


az.plot_trace(trace);


az.summary(trace, round_to=2)
