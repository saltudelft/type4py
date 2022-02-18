"""
This script uses PCA to reduce the dimension of type clusters, which decreases the size of type clusters (Annoy Index).
NOTE THAT the reduced version of type clusters causes a slight performance loss in type prediction.
"""

from type4py import logger, KNN_TREE_SIZE
from type4py.utils import load_model_params
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
from os.path import join
from tqdm import tqdm
import numpy as np
import pickle

logger.name = __name__

def reduce_tc(args):
    model_params = load_model_params()
    type_cluster_index = AnnoyIndex(model_params['output_size'], 'euclidean')
    type_cluster_index.load(join(args.o, "type4py_complete_type_cluster"))
    logger.info("Loaded type clusters")

    type_cluster_dps = np.zeros((type_cluster_index.get_n_items(), model_params['output_size']))
    for i in tqdm(range(type_cluster_index.get_n_items()), desc="Retrieving data points from type clusters"):
        type_cluster_dps[i] = type_cluster_index.get_item_vector(i)

    logger.info(f"Applying PCA to type clusters to reduce dimension from {model_params['output_size']} to {args.d}")
    pca = PCA(n_components=args.d)
    reduced_type_clusters = pca.fit_transform(type_cluster_dps)

    pickle.dump(pca, open(join(args.o, 'type_clusters_pca.pkl'), 'wb'))

    logger.info("Building the reduced type clusters")
    tc_reduced_index = AnnoyIndex(pca.n_components_, 'euclidean')
    for i, v in tqdm(enumerate(reduced_type_clusters), total=len(reduced_type_clusters)):
        tc_reduced_index.add_item(i, v)

    tc_reduced_index.build(KNN_TREE_SIZE)
    tc_reduced_index.save(join(args.o, 'type4py_complete_type_cluster_reduced'))
    logger.info("Saved the reduced type clusters on the disk")
    