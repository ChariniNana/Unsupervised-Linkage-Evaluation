"""
This program conducts unsupervised cluster evaluation based on the basic
silhouette coefficient measures of intra cluster and inter cluster
similarity measures. Instead of the distance as used in the original
silhouette measure, here we use the similarity. The concept is...
1) Calculate the intra cluster similarity a and inter cluster similarity b for
each record as defined in https://en.wikipedia.org/wiki/Silhouette_(clustering)

2) If a < b, count those records as wrong assignments.

3) Calculate FP and FN for each wrong assignment (don't double count FP).
FN is counted based on the closest cluster size corresponding to distance b.

4) Disregard all wrongly assigned nodes and calculate the TP. Overall cluster
quality can be measured by considering the total TP, FP, FN.

In the paper, this corresponds to Algorithm 1

@Author: Charini
@Year: 2024
"""

import time
import csv
import sys
import numpy as np
import utils.eval_utils as utils

assert len(sys.argv) == 2
db_name = sys.argv[1]
assert db_name in ['google_scholar_full', 'google_scholar_limited', 'ios',
                   'DS_C0', 'cora', 'musicbrainz'], db_name

sim_graph_type = 'all'

pairwise_sim_dict, gt_pairs_set = utils.load_sim_graph_and_gt(db_name)

# Construct a dictionary, where for each record, the set of other records with
# which it has a link with a similarity above 0 is contained
sim_rec_dict = {}
for rec_pair, sim in pairwise_sim_dict.items():
  rec_id1 = rec_pair[0]
  rec_id2 = rec_pair[1]
  assert rec_id1 != rec_id2
  if sim > 0:
    sim_rec_set = sim_rec_dict.get(rec_id1, set())
    sim_rec_set.add(rec_id2)
    sim_rec_dict[rec_id1] = sim_rec_set

    sim_rec_set = sim_rec_dict.get(rec_id2, set())
    sim_rec_set.add(rec_id1)
    sim_rec_dict[rec_id2] = sim_rec_set

# Open a csv file to write results
csv_file = open(f'unsupervised_silhouette_clusters_{db_name}_results.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
  ['clustering_algo', 'graph_base_thresh', 'clustering_thresh',
   'no_of_clusters', 'full_similarity_graph_size |E_S|',
   'no_of_predicted_matches |E_T| (edges_in_clusters)',
   'tp', 'fp', 'fn', 'prec', 'reca', 'fstar', 'gt_tp', 'gt_fp',
   'gt_fn', 'gt_prec', 'gt_reca', 'gt_fstar',
   'time_unsupervised_eval', 'time_supervised_eval'])

for algo, thresh in [('clip_orig', 0.5), ('clip_orig', 0.55),
                     ('clip_orig', 0.6), ('clip_orig', 0.65),
                     ('clip_orig', 0.7), ('clip_orig', 0.75),
                     ('clip_orig', 0.8), ('clip_orig', 0.85),
                     ('clip_orig', 0.9), ('clip_orig', 0.95),
                     ('clip_orig', 1.0), ('star', 0.5), ('star', 0.55),
                     ('star', 0.6), ('star', 0.65), ('star', 0.7),
                     ('star', 0.75), ('star', 0.8), ('star', 0.85),
                     ('star', 0.9), ('star', 0.95), ('star', 1.0),
                     ('conn_comp', 0.5), ('conn_comp', 0.55),
                     ('conn_comp', 0.6), ('conn_comp', 0.65),
                     ('conn_comp', 0.7), ('conn_comp', 0.75),
                     ('conn_comp', 0.8), ('conn_comp', 0.85),
                     ('conn_comp', 0.9), ('conn_comp', 0.95),
                     ('conn_comp', 1.0)]:

  clustering_result_file_path = f'../../data/er_clusters/{db_name}/' \
                                f'{db_name}-{algo}-clusters-ids-' \
                                f'{sim_graph_type}-{thresh}.csv'

  start_time = time.time()
  cluster_per_rec_dict, classified_pairs_set, records_per_cluster_dict = \
      utils.load_clusters(clustering_result_file_path)  # Load the
  # classification
  cluster_load_time = time.time() - start_time

  start_time = time.time()
  print('Overall cluster quality as per supervised evaluation based on ground '
        'truth...')
  gt_tp, gt_fp, gt_fn, gt_prec, gt_reca, gt_fstar = utils.supervised_eval(
    classified_pairs_set, gt_pairs_set)
  print(f'  TP = {gt_tp}, FP = {gt_fp}, FN = {gt_fn}')
  print(f'  Precision = {gt_prec}, Recall = {gt_reca}, F-star = {gt_fstar}')
  sup_total_time = (time.time() - start_time) + cluster_load_time
  print(f'  Time for supervised evaluation = {sup_total_time:.2f}')
  print()

  inter_cluster_sim_dict = {}
  inter_cluster_conn_dict = {}
  min_graph_sim = min(pairwise_sim_dict.values())

  print(f'Running unsupervised silhouette evaluation experiments on '
        f'clustering results obtained with algorithm {algo} and '
        f'similarity threshold {thresh}...')
  print()
  start_time = time.time()

  tot_fp = 0
  tot_fn = 0
  tot_tp = 0

  best_clus_size_list = []

  wrong_counts_per_cluster_size_dict = {}  # Number of wrongly assigned
  # nodes per cluster size

  # Iterate over each record assigned to a cluster
  #
  for rec_id, cluster_id_of_rec in cluster_per_rec_dict.items():
    assigned_cluster = records_per_cluster_dict[cluster_id_of_rec]
    if len(assigned_cluster) < 2:
      continue  # If rec_id is alone in a cluster, ignore
    assert rec_id not in inter_cluster_sim_dict
    max_sim = 0
    closest_cluster = None
    processed_linked_cluster_set = set()
    assert rec_id in sim_rec_dict, rec_id
    linked_rec_set = sim_rec_dict[rec_id]
    external_linked_rec_set = linked_rec_set - assigned_cluster

    # Calculate the inter-cluster similarity per record
    #
    for ext_linked_rec in external_linked_rec_set:
      linked_cluster_id = cluster_per_rec_dict[ext_linked_rec]
      assert linked_cluster_id != cluster_id_of_rec
      if linked_cluster_id in processed_linked_cluster_set:
        continue  # Do not recalculate similarity per each link to cluster.

      processed_linked_cluster_set.add(linked_cluster_id)
      sim_val = utils.cluster_dist_or_sim(
        rec_id, records_per_cluster_dict[linked_cluster_id],
        pairwise_sim_dict, 'inter', 'sim')

      if sim_val > max_sim:
        max_sim = sim_val
        closest_cluster = linked_cluster_id

    # Get the closest cluster to record
    inter_cluster_sim_dict[rec_id] = max_sim
    inter_cluster_conn_dict[rec_id] = closest_cluster

  # Calculate the intra cluster similarity per node in cluster and get the
  # wrongly assigned nodes where intra sim < inter sim
  for cluster_id, rec_set in records_per_cluster_dict.items():
    wrongly_assigned_rec_dict = {}
    correct_rec_set = set()
    local_fp = 0
    local_tp = 0
    fn_per_cluster = 0
    if len(rec_set) < 2:  # Defined for clusters with at least 2 records
      continue

    tot_pairs = (len(rec_set) * (len(rec_set) - 1)) / 2

    for rec_id in rec_set:
      intra_sim = utils.cluster_dist_or_sim(rec_id, rec_set, pairwise_sim_dict,
                                            'intra', 'sim')

      assert rec_id in inter_cluster_sim_dict
      inter_sim = inter_cluster_sim_dict[rec_id]

      if intra_sim < inter_sim:
        assert rec_id in inter_cluster_conn_dict
        best_cluster = inter_cluster_conn_dict[rec_id]
        wrongly_assigned_rec_dict[rec_id] = best_cluster
      else:
        correct_rec_set.add(rec_id)

    # Count the FP and TP
    #
    if len(wrongly_assigned_rec_dict) > 0:
      local_fp = (len(wrongly_assigned_rec_dict) *
                  (len(wrongly_assigned_rec_dict) - 1)) / 2 + \
                 (len(wrongly_assigned_rec_dict) * len(correct_rec_set))
    local_tp = (len(correct_rec_set) * (len(correct_rec_set) - 1)) / 2
    assert local_fp + local_tp == tot_pairs, (local_fp, local_tp,
                                              tot_pairs)

    # Count the FN considering the best cluster
    #
    for wrong_rec, best_cluster in wrongly_assigned_rec_dict.items():
      fn_per_rec = len(records_per_cluster_dict[best_cluster])
      fn_per_cluster += fn_per_rec
      best_clus_size_list.append(fn_per_rec)

    cluster_size = len(rec_set)
    wrong_count = len(wrongly_assigned_rec_dict)
    print(f'  Quality of cluster {cluster_id} '
          f'(cluster size = {cluster_size}): TP = {local_tp}, '
          f'FP = {local_fp}, FN = {fn_per_cluster}, '
          f'Wrongly assigned node count = {wrong_count}')

    # Increment the global counters
    tot_tp += local_tp
    tot_fp += local_fp
    tot_fn += fn_per_cluster

    # Populate the wrong count list
    wrong_count_list = wrong_counts_per_cluster_size_dict.get(cluster_size,
                                                              [])
    wrong_count_list.append(wrong_count)
    wrong_counts_per_cluster_size_dict[cluster_size] = wrong_count_list

  print()
  print('  Overall cluster quality as per the unsupervised evaluation '
        'technique considering all clusters...')
  print(f'    TP = {tot_tp}, FP = {tot_fp}, FN = {tot_fn}')

  # Calculate the precision, recall and F* measure
  prec = utils.precision(tot_tp, tot_fp)
  reca = utils.recall(tot_tp, tot_fn)
  fstar = utils.f_star(tot_tp, tot_fp, tot_fn)

  print(f'    Precision = {prec}, Recall = {reca}, F-star = {fstar}')
  unsup_total_time = (time.time() - start_time) + cluster_load_time
  print(f'    Time for silhouette evaluation '
        f'= {unsup_total_time:.2f}')

  print()
  print('  Closest (best) cluster size statistics')
  if len(best_clus_size_list) > 0:
    min_size = min(best_clus_size_list)
    max_size = max(best_clus_size_list)
    avr = sum(best_clus_size_list) / len(best_clus_size_list)
    med = np.median(best_clus_size_list)
    print(f'    min = {min_size}, max = {max_size}, mean = {avr}, '
          f'median = {med}')
  else:
    print(f'    No wrong assignments. Therefore no closest cluster '
          f'statistics')

  csv_writer.writerow([algo, min_graph_sim, thresh,
                       len(records_per_cluster_dict), len(pairwise_sim_dict),
                       len(classified_pairs_set),
                       tot_tp, tot_fp, tot_fn, prec, reca, fstar,
                       gt_tp, gt_fp, gt_fn, gt_prec, gt_reca, gt_fstar,
                       unsup_total_time, sup_total_time])

  print()
  print('  Number of wrong nodes per cluster size distribution')
  for cluster_size, wrong_count_list in sorted(list(
    wrong_counts_per_cluster_size_dict.items())):
    if len(wrong_count_list) == 0:
      min_wrong = max_wrong = avr = med = 0
    else:
      min_wrong = min(wrong_count_list)
      max_wrong = max(wrong_count_list)
      avr = sum(wrong_count_list) / len(wrong_count_list)
      med = np.median(wrong_count_list)
    print(f'    Cluster size = {cluster_size} -> Wrong count '
          f'min = {min_wrong}, max = {max_wrong}, mean = {avr}, '
          f'median = {med}')

  print('-' * 80)
  print()

csv_file.close()
print('--Done--')
