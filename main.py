#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10

@author: steffen cornwell
"""

import lime
import desc
import numpy as np
import pandas as pd
import scanpy.api as sc
import numpy as np
import shutil
import math
import statistics

sc.settings.verbosity = 3
sc.logging.print_versions()
shutil.rmtree("./result_tmp")

adata = desc.utilities.read_mtx('/home/steffen/Genetics/Limonada/GSM2560248_2.1.mtx').T
gene_names = pd.read_csv("/home/steffen/Genetics/Limonada/GSE96583_batch2.genes.tsv", sep="\t", names=["ensembl_id", "gene_name"])
gene_names = pd.DataFrame(gene_names, dtype=np.str)
barcodes = pd.read_csv("/home/steffen/Genetics/Limonada/GSM2560248_barcodes.tsv.gz", sep="\t", names=["barcodes"], index_col=0)
cell_information = pd.read_csv("/home/steffen/Genetics/Limonada/GSE96583_batch2.total.tsne.df.tsv", sep="\t", index_col=0)
adata.var = gene_names
adata.obs = barcodes
adata.obs = adata.obs.join(other=cell_information, how="inner")

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

mito_genes = adata.var_names.str.startswith('MT-')
# for each cell compute fraction of counts in mito genes vs. all genes
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
adata.obs['n_counts'] = adata.X.sum(axis=1).A1

adata = adata[adata.obs['n_genes'] < 2500, :]
adata = adata[adata.obs['percent_mito'] < 0.05, :]

desc.normalize_per_cell(adata, counts_per_cell_after=1e4)

desc.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
adata = adata[:, adata.var['highly_variable']]

desc.scale(adata, zero_center=True, max_value=3)
#Let the max value be changed

print("Training")
#Failure to save encoder overwrite
adata = desc.train(adata, dims=[adata.shape[1], 32, 16], tol=0.005, n_neighbors=10,
                   batch_size=256, louvain_resolution=[0.8],
                   do_tsne=False, learning_rate=300,
                   do_umap=False, num_Cores_tsne=4,
                   save_encoder_weights=True)
                   
#%%
from lime.lime_tabular import LimeTabularExplainer
lime.lime_tabular.LimeTabularExplainer.explain_instance

clusters = adata.obs['desc_0.8']
num_genes = adata.X.shape[1]
num_cells = adata.X.shape[0]
num_clusters = max(clusters) + 1
y_clusters = np.zeros(shape=(num_cells, num_clusters))
clusters_ndarray = clusters._ndarray_values.reshape((num_cells, 1))
np.put_along_axis(y_clusters, clusters_ndarray, 1, axis=1)
explainer = LimeTabularExplainer(adata.X, mode="classification", training_labels=clusters,
                             feature_names=adata.var["gene_name"], 
                             discretize_continuous=False)

from keras.layers import Input, Dropout, Dense
from keras.models import Model
import keras

desc_model = keras.engine.saving.load_model("/home/steffen/Genetics/Limonada/result_tmp/encoder_model.h5")
desc_model.compile(optimizer="sgd", loss="kld")

reduced_input = Input(shape=(16,))
dropout = Dropout(rate=1/16)(reduced_input)
output_cluster = Dense(units=num_clusters, activation="sigmoid")(dropout)
clustering_model = Model(inputs=reduced_input, outputs=output_cluster)
clustering_model.compile(optimizer="sgd", loss="categorical_hinge")
reduced_data = adata.obsm
reduced_data_array = np.zeros(shape=(num_cells, 16))
je = 0
for data_point in reduced_data:
    reduced_data_array[je] = data_point[0]
    je += 1
del reduced_data
clustering_model.fit(x=reduced_data_array , y=y_clusters, epochs=300)
clustering_model.predict(reduced_data_array)

from keras.engine.sequential import Sequential
jdesc_model = Sequential()
first_layer = Dense(units=32, input_shape=(num_genes,), activation="relu")
jdesc_model.add(first_layer)
second_layer = Dense(units=16, activation="tanh")
jdesc_model.add(second_layer)
third_layer = Dense(units=num_clusters, activation="sigmoid")
jdesc_model.add(third_layer)
jdesc_model.compile(optimizer="sgd", loss="categorical_hinge")
jdesc_model_weights = desc_model.get_weights() + clustering_model.get_weights()
jdesc_model.set_weights(jdesc_model_weights)    
jdesc_model_predictions = jdesc_model.predict(adata.X)
predicted_clusters_array = jdesc_model_predictions.argmax(axis=1)
clusters_array = clusters_ndarray.T[0]

error_array = np.zeros(shape=(num_clusters, num_clusters))
correct_array = np.zeros(shape=(num_clusters,), dtype=int)
for i in range(num_cells):
    if clusters_array[i] != predicted_clusters_array[i]:
        error_array[clusters_array[i], predicted_clusters_array[i]] += 1
    else:
        correct_array[clusters_array[i]] += 1

correct_percentage = correct_array / np.bincount(clusters_array)

#%%
import random
from tqdm import tqdm
import scipy.stats as st

num_sample_per_label = 50
pvalue = 0.01

def predict_fn(row):
    preds = jdesc_model.predict(row)
    return np.apply_along_axis(lambda row : row / np.sum(row), axis = 1, arr = preds)

predicted_clusters_label_indexes = [[] for i in range(num_clusters)]
for index in range(num_cells):
    predicted_clusters_label_indexes[predicted_clusters_array[index]].append(index)

cluster_gene_weights = []
cluster_genes_compiled = []
cluster_top_genes = []
for label in range(num_clusters): 
    print("\n Calculating Genes for Label "+str(label))
    gene_weights = pd.DataFrame()
    top_gene_weights = pd.DataFrame()
    genes_compiled = pd.DataFrame(columns=["gene", "weight"]).set_index("gene")
    noise_weights = []
    predicted_cluster_indexes = predicted_clusters_label_indexes[label]
    if len(predicted_cluster_indexes) < num_sample_per_label:
        predicted_cluster_indexes_sample = predicted_cluster_indexes
    else:
        predicted_cluster_indexes_sample = random.sample(
                predicted_cluster_indexes, num_sample_per_label)
    for j in tqdm(predicted_cluster_indexes_sample):
        explanation = explainer.explain_instance(data_row = adata.X[j, :],
                                                 predict_fn = predict_fn, 
                                                 num_features = 100,
                                                 labels=(label,))
        list_genes = explanation.as_list(label = label)
        for gene in list_genes:
            gene_weights.loc[adata.obs.index[j], gene[0]] = gene[1]
            if genes_compiled.index.contains(gene[0]):
                recent_value = genes_compiled.loc[gene[0]]
                genes_compiled.loc[gene[0]] = recent_value + gene[1]
            else:
                genes_compiled.loc[gene[0]] = gene[1]
                
    genes_compiled['abs_weight'] = abs(genes_compiled['weight'])
    genes_compiled.sort_values(inplace=True, by="abs_weight", ascending=False)
    genes_compiled = genes_compiled.drop(labels="abs_weight", axis="columns")
    genes_compiled = genes_compiled / num_sample_per_label
    top_gene_weights = genes_compiled.iloc[:100]
    for gene in genes_compiled.iloc[100:].iterrows():
        gene_name = gene[0]
        for weight in gene_weights[gene_name]:
            if not math.isnan(weight):
                noise_weights.append(weight)
     
    noise_mean = statistics.mean(noise_weights)
    noise_sd = statistics.stdev(noise_weights)
    noise_z = st.norm.ppf(1-(pvalue/2.0))
    noise_cutoff = noise_z*noise_sd
    print(noise_mean)
    print(noise_cutoff)
    noise_index = 0
    while abs(top_gene_weights['weight'].iloc[noise_index]) >= noise_cutoff:
        noise_index += 1
    print(noise_index)
    top_gene_weights = top_gene_weights.iloc[:noise_index]
    cluster_top_genes.append(top_gene_weights)
    cluster_gene_weights.append(gene_weights)
    cluster_genes_compiled.append(genes_compiled)
#%%

cells = np.unique(adata.obs['cell'])

num_cell_types = len(cells)
#adata.obs.loc[:, ('cluster', 'cell')].sort_values(by='cluster', ascending=True).loc[:, 'cell']
cell_types_of_cluster = np.zeros(shape=(num_clusters, num_cell_types))
for i in range(num_cells):
    cell_index = 0
    for j in range(num_cell_types):
        if adata.obs['cell'][i] == cells[j]:
            cell_index = j  
    cell_types_of_cluster[predicted_clusters_array[i], cell_index] += 1

cell_type_percentages_of_cluster_pd = pd.DataFrame(cell_types_of_cluster, columns=cells)
cell_type_percentages_of_cluster_pd = cell_type_percentages_of_cluster_pd.divide(
        cell_type_percentages_of_cluster_pd.sum(axis=1), axis='index')


#import matplotlib.pyplot as plt
#import sklearn
#gene_correlation = 1 - gene_weights[cluster_genes_compiled.index.to_list()[0:50]].corr()
#tsne = sklearn.manifold.TSNE(n_components=2, metric='precomputed')
#gene_tsne_coordinate = pd.DataFrame(tsne.fit_transform(gene_correlation))
#plt.plot(x = gene_tsne_coordinate[:, 0], y = gene_tsne_coordinate[:, 1])
#np.save("gene_correlation", gene_correlation)
#np.load()

#%%
output = open('GSE96583_Report.txt', 'w')
output.write("Report on Cell Clusters\n")
for label in range(num_clusters):
    output.write("Cell Cluster:\n")
    output.write("Cell Type Percentages:\n")
    output.write(cell_type_percentages_of_cluster_pd.iloc[label].to_string() + "\n")
    output.write(cluster_top_genes[label].to_string())
    output.write("\n\n\n\n\n")
output.close()

#%%
cell_marker_database = pd.read_csv("all_cell_types.txt")
panglao_database = pd.read_csv("")