"""
This module contains helper and common functions used in other evaluation
modules

@Author: Charini
@Year: 2024
"""

import gzip
import csv
import json


def load_zipped_sim_file(sim_file_path):
  """
  Load the similarity graph into a dictionary (record pairs and corresponding
  similarity)
  :param sim_file_path: Path to the file containing pairwise similarity values
  :return sim_dict: Dictionary containing the record pairs as keys and pairwise
  similarity as values
  """
  sim_dict = {}

  if sim_file_path.endswith('.csv'):
    sim_file_path = sim_file_path + '.gz'

  try:
    csv_file = gzip.open(sim_file_path, mode='rt')
  except IOError:
    print('Weight vector file "%s" not found.' % sim_file_path)
    raise IOError

  csv_reader = csv.reader(csv_file)

  next(csv_reader)  # Skip over header row

  # Load the weight vectors and convert weights into numerical values
  #
  for row in csv_reader:
    iid1 = row[0].strip()
    iid2 = row[1].strip()

    sorted_id_pair = tuple(sorted((iid1, iid2)))
    assert sorted_id_pair not in sim_dict, row

    pair_sim = float(row[2])
    sim_dict[sorted_id_pair] = pair_sim

  return sim_dict


def load_sim_graph_and_gt(db_name):
  """
  Function to load the similarity graph and ground truth of different
  datasets
  :param db_name:
  :return pairwise_sim_dict, gt_pairs_set: The pairwise similarity dictionary
  and the set containing the ground truth matches.
  """

  root = f'../data/similarity_graphs_and_ground_truth/{db_name}/'
  pairwise_sim_dict = {}
  gt_pairs_set = set()

  if db_name == 'ios':
    sim_file_path = root + 'ios_similarity_file.csv.gz'
    gt_file_path = root + 'ios_gt_links.csv'

    pairwise_sim_dict = load_zipped_sim_file(sim_file_path)  # Load the
    # similarity file to a dictionary

    with open(gt_file_path) as csv_file:
      csv_reader = csv.reader(csv_file)
      for row in csv_reader:
        assert row[0] < row[1], (row[0], row[1])  # Assert pairs are sorted
        gt_pairs_set.add((row[0], row[1]))

  elif db_name == 'DS_C0':
    sim_file_path = root + 'links.csv'
    gt_file_path = root + 'gold_standard.csv'

    with open(sim_file_path) as csv_file:
      csv_reader = csv.reader(csv_file)

      for row in csv_reader:
        id1 = row[0].strip()
        id2 = row[1].strip()
        pair_sim = float(row[2].strip())
        assert 0 <= pair_sim <= 1, pair_sim
        pairwise_sim_dict[tuple(sorted([id1, id2]))] = pair_sim

    with open(gt_file_path) as csv_file:
      csv_reader = csv.reader(csv_file)

      for row in csv_reader:
        id1 = row[0].strip()
        id2 = row[1].strip()
        gt_pairs_set.add(tuple(sorted([id1, id2])))

  elif db_name == 'cora':
    sim_and_gt_file_path = root + 'cora_similarities_and_gt.csv'

    with open(sim_and_gt_file_path) as csv_file:
      csv_reader = csv.reader(csv_file)
      header = next(csv_reader)

      id1_idx = header.index('source_id')
      id2_idx = header.index('target_id')
      truth_idx = header.index('label')
      pair_sim_idx = header.index('cosine_tfidf')

      for row in csv_reader:
        id1 = row[id1_idx].strip()
        id2 = row[id2_idx].strip()
        pair_sim = float(row[pair_sim_idx].strip())
        assert 0 <= pair_sim <= 1, pair_sim

        pairwise_sim_dict[tuple(sorted([id1, id2]))] = pair_sim

        truth_val = row[truth_idx]
        assert truth_val in ['True', 'False'], truth_val
        if truth_val == 'True':
          gt_pairs_set.add(tuple(sorted([id1, id2])))

  elif db_name == 'musicbrainz':
    sim_file_path = root + 'edges.json'
    gt_file_path = root + 'vertices.json'
    gt_cluster_dict = {}

    # Read the nodes file to generate the GT.
    #
    with open(gt_file_path, 'r') as json_file:
      file_lines = json_file.readlines()
      for line in file_lines:
        row_dict = json.loads(line)

        rec_id = row_dict['id'].strip()

        clus_id = row_dict['data']['recId']
        gt_cluster = gt_cluster_dict.get(clus_id, set())
        gt_cluster.add(rec_id)
        gt_cluster_dict[clus_id] = gt_cluster

    # Generate GT pairs
    #
    for clus_id, gt_set in gt_cluster_dict.items():
      sorted_id_list = sorted(gt_set)
      for i, id1 in enumerate(sorted_id_list[:-1]):
        for id2 in sorted_id_list[i + 1:]:
          gt_pairs_set.add((id1, id2))

    # Load the similarity file
    with open(sim_file_path, 'r') as json_file:
      file_lines = json_file.readlines()
      for line in file_lines:
        row_dict = json.loads(line)

        id1 = row_dict['source'].strip()
        id2 = row_dict['target'].strip()
        pair_sim = float(row_dict['data']['value'])
        pairwise_sim_dict[tuple(sorted([id1, id2]))] = pair_sim

  else:
    assert db_name in ['google_scholar_full', 'google_scholar_limited'], \
      db_name
    if db_name == 'google_scholar_full':
      sim_file_path = root + 'google_scholar_sim_file_all_rec.csv.gz'
      gt_file_path = root + 'google_scholar_rec_and_ground_truth_all_rec.csv'
    else:
      sim_file_path = root + 'google_scholar_sim_file_rec_in_orig_gt.csv.gz'
      gt_file_path = root + 'google_scholar_rec_and_ground_truth_rec_in_' \
                            'orig_gt.csv'

    # Read GT pairs
    with open(gt_file_path) as csv_file:
      csv_reader = csv.reader(csv_file)
      rec_count = int(next(csv_reader)[0])  # Get no of records

      actual_rec_count = 0
      for i in range(rec_count):  # Read records
        next(csv_reader)
        actual_rec_count += 1
      assert rec_count == actual_rec_count, (rec_count, actual_rec_count)

      for row in csv_reader:  # Read GT record pairs
        assert len(row) == 2, len(row)
        rec_id1 = row[0]
        rec_id2 = row[1]
        assert int(rec_id1) < int(rec_id2)
        gt_pairs_set.add(tuple(sorted([rec_id1, rec_id2])))

    # Load the similarity file
    pairwise_sim_dict = load_zipped_sim_file(sim_file_path)

  return pairwise_sim_dict, gt_pairs_set


def load_clusters(cluster_result_file_path, min_size=1):
  """
  Load the clustering result obtained with a group record linkage algorithm
  :param cluster_result_file_path: Path to the clustering result file
  :param min_size: Minimum GT cluster size to consider
  :return: cluster_per_rec_dict, classified_pairs_set, records_per_cluster_dict
  """
  cluster_per_rec_dict = {}
  classified_pairs_set = set()
  records_per_cluster_dict = {}

  with open(cluster_result_file_path) as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)

    ind_id_idx = header.index('ds_ind_id')
    cluster_id_idx = header.index('cluster_id')

    for row in csv_reader:
      ind_id = row[ind_id_idx]
      cluster_id = row[cluster_id_idx]

      rec_set = records_per_cluster_dict.get(cluster_id, set())
      assert ind_id not in rec_set
      rec_set.add(ind_id)
      records_per_cluster_dict[cluster_id] = rec_set

      assert ind_id not in cluster_per_rec_dict
      cluster_per_rec_dict[ind_id] = cluster_id

  singleton_count = 0
  for cluster_id in list(records_per_cluster_dict.keys()):
    rec_set = records_per_cluster_dict[cluster_id]
    assert len(rec_set) > 0
    if len(rec_set) == 1:
      singleton_count += 1

    if len(rec_set) < min_size:  # Filter the clusters which are less than
      # min_size
      del records_per_cluster_dict[cluster_id]
      for rec in rec_set:
        del cluster_per_rec_dict[rec]
      continue  # Ignore removed clusters when obtaining pairs
    sorted_id_list = sorted(rec_set)
    for i, ind1 in enumerate(sorted_id_list[:-1]):
      for ind2 in sorted_id_list[i+1:]:
        assert ind1 in cluster_per_rec_dict
        assert ind2 in cluster_per_rec_dict
        rec_pair = (ind1, ind2)
        assert rec_pair not in classified_pairs_set
        classified_pairs_set.add(rec_pair)

  print(f'  Number of records across clusters = {len(cluster_per_rec_dict)}')
  print(f'  Total number of clusters = {len(records_per_cluster_dict)}')
  print(f'      {singleton_count} out of these are singletons')
  print(f'  Total number of classified matches = {len(classified_pairs_set)}')
  print()

  return cluster_per_rec_dict, classified_pairs_set, records_per_cluster_dict


def precision(tp, fp):
  assert tp + fp > 0
  return tp/(tp+fp)


def recall(tp, fn):
  if tp == 0:
    return 0
  assert tp + fn > 0
  return tp/(tp+fn)


def f_star(tp, fp, fn):
  assert tp + fp + fn > 0
  return tp / (tp + fp + fn)


def supervised_eval(classified_pairs_set, gt_pairs_set):
  """

  :param classified_pairs_set: Set of record pairs classified as matches
  :param gt_pairs_set: Set of ground truth pairs
  :return prec, reca, fstar: Precision, recall and F-star measure
  """
  assert len(gt_pairs_set) > 0
  assert len(classified_pairs_set) > 0

  tp = len(classified_pairs_set & gt_pairs_set)
  fp = len(classified_pairs_set - gt_pairs_set)
  fn = len(gt_pairs_set - classified_pairs_set)

  prec = precision(tp, fp)
  reca = recall(tp, fn)
  fstar = f_star(tp, fp, fn)

  return tp, fp, fn, prec, reca, fstar


def get_scores(sim_val_dict):
  """
  Function to  obtain scores that show the confidence of a link being a TP, FP,
  TN or FN. These scores are calculated based on the pairwise similarity.
  Used in the link based unsupervised evaluation method.
  :param sim_val_dict: A dictionary of the format {'m_within': [],
  'u_within': [], 'm_across': [], 'u_across': []} to hold similarity values of
  links.
  :return score_dict: A dictionary containing score values. This has the same
  structure as the sim_val_dict.
  """

  score_dict = {}

  # Get match scores
  for match_non_match_type in ['m_within', 'm_across']:
    sim_list = sim_val_dict[match_non_match_type]
    score_list = []
    for sim_value in sim_list:
      score_list.append(round(sim_value, 6))

    assert len(score_list) == len(sim_list)
    score_dict[match_non_match_type] = score_list

  # Get non match scores
  for match_non_match_type in ['u_within', 'u_across']:
    sim_list = sim_val_dict[match_non_match_type]
    score_list = []
    for sim_value in sim_list:
      score_list.append(round((1-sim_value), 6))

    assert len(score_list) == len(sim_list)
    score_dict[match_non_match_type] = score_list

  return score_dict


def cluster_dist_or_sim(node, cluster, dist_or_sim_graph, inter_or_intra,
                        dist_or_sim='dist', all_or_graph_edges='all_edges'):
  """
  This method calculates the average distance (or similarity) from node to the
  rest of the nodes in the cluster if the cluster_dist_type is given as
  'intra', whereas the average to a different cluster is calculated if the
  cluster_dist_type is given as 'inter'
  :param node: Node for which the cluster distance (or similarity) needs to be
  calculated
  :param cluster: Set of all nodes in the cluster to which distance
  (or similarity) must be calculated
  :param dist_or_sim_graph: Pairwise distance or similarity graph
  :param inter_or_intra: Intra or inter cluster distance
  :param dist_or_sim: Whether a distance or similarity graph is given
  :param all_or_graph_edges: Whether to only consider the edges in the graph
  or all edges when calculating distance (or similarity)
  :return avg_dist_or_sim: Cluster distance (or similarity) of node
  """
  assert inter_or_intra in ['intra', 'inter'], inter_or_intra
  assert dist_or_sim in ['dist', 'sim']
  divisor = None

  if dist_or_sim == 'dist':
    default_val = 1
  else:
    default_val = 0

  if inter_or_intra == 'intra':
    assert node in cluster
    assert len(cluster) > 1  # Singleton clusters are not handled here
    other_nodes = cluster - {node}
    if all_or_graph_edges == 'all_edges':
      divisor = len(cluster)-1

  else:
    assert node not in cluster
    assert len(cluster) >= 1
    other_nodes = cluster
    if all_or_graph_edges == 'all_edges':
      divisor = len(cluster)

  tot_dist = 0
  sim_graph_edges_count = 0
  for other_node in other_nodes:
    rec_pair = tuple(sorted([node, other_node]))
    if rec_pair in dist_or_sim_graph:
      tot_dist += dist_or_sim_graph[rec_pair]
      sim_graph_edges_count += 1
    else:
      tot_dist += default_val
  if divisor is None:
    assert all_or_graph_edges == 'graph_edges'
    divisor = sim_graph_edges_count
    if divisor == 0:
      assert inter_or_intra == 'intra'  # For inter clusters, we only send
      # clusters with a link to the given node in the similarity graph
      return 0.0  # A within cluster calculation where the node has no links
      # to other nodes in assigned cluster. Can occur in a rare case (where we
      # post process clusters to match constraints, and linked nodes are
      # removed as a result, leaving on non-linked nodes in a cluster)

  avg_dist_or_sim = tot_dist/divisor
  assert 0.0 <= avg_dist_or_sim <= 1
  return avg_dist_or_sim
# -----------------------------------------------------------------------------
