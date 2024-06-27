"""
This program implements the framework for graph structure based evaluation
methods. In the paper this corresponds to Equations 15 to 24.

@Author: Charini
@Year: 2024
"""

import json
import time
import csv
import sys
import utils.eval_utils as utils


def local_link_score(rec_sim_to_cluster_dict, rec1, rec2, clus1, clus2):
  """
  This function calculates the link scores by considering the local
  neighbourhood of record links (only considering the links a record has to
  its assigned cluster, or an external linked cluster, but not both)
  :param rec_sim_to_cluster_dict: A dictionary storing the average similarity
  from a record to a cluster.
  :param rec1: The first record in a record pair to be considered
  :param rec2: The second record in a record pair to be considered
  :param clus1: The cluster to which the first record is assigned
  :param clus2: The cluster to which the second record is assigned
  :return link_score: The score of link (rec1, rec2) calculated by considering
  the local neighbourhood of each record
  """
  cluster_avg_sim_dict_rec1 = rec_sim_to_cluster_dict[rec1]
  cluster_avg_sim_dict_rec2 = rec_sim_to_cluster_dict[rec2]

  if clus1 == clus2:
    rec_id1_score = cluster_avg_sim_dict_rec1[clus1]
    rec_id2_score = cluster_avg_sim_dict_rec2[clus2]
  else:
    assert clus1 != clus2
    assert clus2 in cluster_avg_sim_dict_rec1
    assert clus1 in cluster_avg_sim_dict_rec2
    rec_id1_score = cluster_avg_sim_dict_rec1[clus2]
    rec_id2_score = cluster_avg_sim_dict_rec2[clus1]

  link_score = (rec_id1_score + rec_id2_score) / 2
  assert 0 <= link_score <= 1, link_score
  return link_score
  # --------------------------------------------------------------------------


def global_link_score(meth_str, global_rec_score_per_external_clus_dict, rec1,
                      rec2, clus1, clus2):
  """
  This function calculates the link scores by considering the global
  neighbourhood of record links (considering the links a record has to
  its assigned cluster, or external linked clusters)
  :param meth_str: One of 'global:min', 'global:max', 'global:avr' methods
  :param global_rec_score_per_external_clus_dict: A dictionary storing the
  global score of records corresponding to each external cluster they are
  linked to.
  :param rec1: The first record in a record pair to be considered
  :param rec2: The second record in a record pair to be considered
  :param clus1: The cluster to which the first record is assigned
  :param clus2: The cluster to which the second record is assigned
  :return link_score: The score of link (rec1, rec2) calculated by considering
  the global neighbourhood of each record
  """
  assert meth_str.startswith('global:'), meth_str
  overall_node_score_funct = meth_str.replace('global:', '')
  assert overall_node_score_funct in ['min', 'max', 'avr'], \
      overall_node_score_funct

  global_rec_score_dict_rec1 = global_rec_score_per_external_clus_dict[rec1]
  global_rec_score_dict_rec2 = global_rec_score_per_external_clus_dict[rec2]

  if clus1 == clus2:
    assert clus1 not in global_rec_score_dict_rec1
    assert clus2 not in global_rec_score_dict_rec2
    if len(global_rec_score_dict_rec1) == 0:
      # Rec1 has no external links. Therefore, best fit in assigned cluster
      rec_id1_score = 1
    else:
      if overall_node_score_funct == 'min':
        rec_id1_score = min(global_rec_score_dict_rec1.values())
      elif overall_node_score_funct == 'max':
        rec_id1_score = max(global_rec_score_dict_rec1.values())
      else:
        rec_id1_score = sum(global_rec_score_dict_rec1.values())/len(
          global_rec_score_dict_rec1)

    if len(global_rec_score_dict_rec2) == 0:
      # Rec2 has no external links. Therefore, best fit in assigned cluster
      rec_id2_score = 1
    else:
      if overall_node_score_funct == 'min':
        rec_id2_score = min(global_rec_score_dict_rec2.values())
      elif overall_node_score_funct == 'max':
        rec_id2_score = max(global_rec_score_dict_rec2.values())
      else:
        rec_id2_score = sum(global_rec_score_dict_rec2.values())/len(
          global_rec_score_dict_rec2)

    link_score = (rec_id1_score + rec_id2_score) / 2

  else:
    assert clus1 != clus2
    assert clus2 in global_rec_score_dict_rec1
    assert clus1 in global_rec_score_dict_rec2
    rec_id1_score = global_rec_score_dict_rec1[clus2]
    rec_id2_score = global_rec_score_dict_rec2[clus1]

    link_score = 1 - ((rec_id1_score + rec_id2_score) / 2)

  assert 0 <= link_score <= 1, link_score
  return link_score
  # --------------------------------------------------------------------------


assert len(sys.argv) == 3
db_name = sys.argv[1]
assert db_name in ['google_scholar_full', 'google_scholar_limited', 'ios',
                   'DS_C0', 'cora', 'musicbrainz'], db_name

sub_method = eval(sys.argv[2])
local_or_global, likely_class, all_or_graph_edges = sub_method

assert local_or_global in ('local', 'global:min', 'global:max',
                           'global:avr'), local_or_global
assert likely_class in ('with_likely_class',
                        'without_likely_class'), likely_class
assert all_or_graph_edges in ('all_edges', 'graph_edges'), all_or_graph_edges

sim_graph_type = 'all'

pairwise_sim_dict, gt_pairs_set = utils.load_sim_graph_and_gt(db_name)

# Construct a dictionary, where for each record, the set of other records with
# which it has a link with a similarity above 0 is contained
sim_rec_dict = {}
for rec_pair, sim in pairwise_sim_dict.items():
  rec_id1 = rec_pair[0]
  rec_id2 = rec_pair[1]
  assert rec_id1 != rec_id2
  sim_rec_set = sim_rec_dict.get(rec_id1, set())
  sim_rec_set.add(rec_id2)
  sim_rec_dict[rec_id1] = sim_rec_set

  sim_rec_set = sim_rec_dict.get(rec_id2, set())
  sim_rec_set.add(rec_id1)
  sim_rec_dict[rec_id2] = sim_rec_set

# Open a csv file to write results
sub_method_str = '-'.join(map(str, sub_method))
csv_file = open(f'graph_structure_based_eval_clusters_{db_name}_'
                f'{sub_method_str}_results.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
  ['clustering_algo', 'graph_base_thresh', 'clustering_thresh',
   'no_of_clusters', 'full_similarity_graph_size |E_S|',
   'no_of_predicted_matches |E_T| (edges_in_clusters)',
   'no_of_predicted_non_matches NM = n*(n-1)/2 - |E_T|',
   'no_of_predicted_matches_and_graph_links = |E_S.union(E_T)|',
   'no_of_predicted_non_matches_in_graph = |E_S - E_T|',
   'no_of_predicted_non_matches_not_in_graph = n*(n-1)/2 - |E_S.union(E_T)|',
   'no_of_predicted_matches_in_graph = |E_S & E_T|',
   'no_of_predicted_matches_not_in_graph = |E_T - E_S|', 'sub_method',
   'est_tp', 'est_fp', 'est_fn', 'prec', 'reca', 'fstar',
   'gt_tp', 'gt_fp', 'gt_fn', 'gt_prec', 'gt_reca', 'gt_fstar',
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
  print(f'Loading clustering result for algorithm {algo} and clustering '
        f'similarity threshold {thresh}...')
  cluster_per_rec_dict, classified_pairs_set, records_per_cluster_dict = \
      utils.load_clusters(clustering_result_file_path)  #
  # Load the classification
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

  tot_links = int(
    len(cluster_per_rec_dict) * (len(cluster_per_rec_dict) - 1) / 2)
  print(f'  Total possible links = {tot_links}')
  min_graph_sim = min(pairwise_sim_dict.values())
  print(f'  Minimum graph pair similarity = {min_graph_sim}')
  print()

  approx_true_match_thresh = 0.5
  approx_true_non_match_thresh = 0.5

  m_within_sum = 0
  u_within_sum = 0
  m_across_sum = 0
  u_across_sum = 0

  rec_avg_sim_to_cluster_dict = {}  # Dictionary storing the average
  # similarity each record has to a cluster

  within_cluster_link_tot = len(classified_pairs_set)
  across_cluster_link_tot = tot_links - within_cluster_link_tot

  # Calculate the number of record pairs in both the similarity graph and
  # classification (E_S \cup E_T)
  classified_matches_and_pairs_in_graph_count = len(
    set(pairwise_sim_dict.keys()).union(classified_pairs_set))

  # Calculate the number of classified non matches in the similarity graph
  #
  classified_non_matches_in_graph = set(
    pairwise_sim_dict.keys()) - classified_pairs_set
  classified_non_matches_in_graph_count = len(
    classified_non_matches_in_graph)
  assert 0 <= classified_non_matches_in_graph_count <= \
         len(pairwise_sim_dict), (classified_non_matches_in_graph_count,
                                  len(pairwise_sim_dict))

  # Calculate the number of classified non matches not in the similarity
  # graph
  classified_non_matches_not_in_graph_count = \
      tot_links - classified_matches_and_pairs_in_graph_count

  # Calculate the number of classified matches in the similarity graph
  #
  classified_matches_in_graph = set(pairwise_sim_dict.keys()).intersection(
    classified_pairs_set)
  classified_matches_in_graph_count = len(classified_matches_in_graph)

  # Calculate the number of classified matches not in the similarity graph
  #
  classified_matches_not_in_graph = classified_pairs_set - set(
    pairwise_sim_dict.keys())
  classified_matches_not_in_graph_count = len(
    classified_matches_not_in_graph)

  assert len(classified_matches_in_graph) + \
         len(classified_non_matches_in_graph) == len(pairwise_sim_dict), \
    (len(classified_matches_in_graph), len(classified_non_matches_in_graph),
     len(pairwise_sim_dict))

  print(f'Running unsupervised node and cluster centric evaluation '
        f'experiments with sub method {str(sub_method)}, on clustering '
        f'results obtained with algorithm {algo} and similarity threshold '
        f'{thresh}...')
  print()
  start_time = time.time()

  # Iterate over each record assigned to a cluster, and calculate the
  # average similarity it has within its own cluster, and external connected
  # clusters
  #
  for rec_id, assigned_cluster_id in cluster_per_rec_dict.items():
    if rec_id not in sim_rec_dict:
      rec_avg_sim_to_cluster_dict[rec_id] = {}
      continue
    assert rec_id not in rec_avg_sim_to_cluster_dict
    cluster_avg_sim_dict = {}
    assigned_cluster_rec_set = records_per_cluster_dict[assigned_cluster_id]

    # Calculate the intra (within) cluster similarity per record
    if len(assigned_cluster_rec_set) > 1:
      intra_sim = utils.cluster_dist_or_sim(rec_id, assigned_cluster_rec_set,
                                            pairwise_sim_dict, 'intra', 'sim',
                                            all_or_graph_edges)
    else:
      assert len(assigned_cluster_rec_set) == 1
      intra_sim = 1
    cluster_avg_sim_dict[assigned_cluster_id] = intra_sim

    assert 0 <= intra_sim <= 1, intra_sim

    # Calculate the inter-cluster similarity per record
    assert rec_id in sim_rec_dict, rec_id
    linked_rec_set = sim_rec_dict[rec_id]
    external_linked_rec_set = linked_rec_set - assigned_cluster_rec_set

    for ext_linked_rec in external_linked_rec_set:
      linked_cluster_id = cluster_per_rec_dict[ext_linked_rec]
      sorted_cluster_id_pair = tuple(sorted([linked_cluster_id,
                                             assigned_cluster_id]))
      assert linked_cluster_id != assigned_cluster_id
      if linked_cluster_id in cluster_avg_sim_dict:
        continue  # Do not recalculate similarity per each link to a cluster.

      linked_cluster_rec_set = records_per_cluster_dict[linked_cluster_id]
      inter_sim = utils.cluster_dist_or_sim(rec_id, linked_cluster_rec_set,
                                            pairwise_sim_dict, 'inter', 'sim',
                                            all_or_graph_edges)
      assert 0 <= inter_sim <= 1, inter_sim
      assert linked_cluster_id not in cluster_avg_sim_dict
      cluster_avg_sim_dict[linked_cluster_id] = inter_sim

    rec_avg_sim_to_cluster_dict[rec_id] = cluster_avg_sim_dict

  # For global score calculation method, update rec_avg_sim_to_cluster_dict to
  # store the internal/(internal + external score to given cluster) per each
  # node
  if local_or_global in ['global:min', 'global:max', 'global:avr']:
    for rec_id, cluster_avg_sim_dict in rec_avg_sim_to_cluster_dict.items():
      if rec_id not in sim_rec_dict:
        assert len(cluster_avg_sim_dict) == 0
        continue
      global_node_score_dict = {}
      assigned_cluster_id = cluster_per_rec_dict[rec_id]
      assert assigned_cluster_id in cluster_avg_sim_dict
      avg_sim_to_assigned_cluster = cluster_avg_sim_dict[assigned_cluster_id]
      for cluster_id, avg_sim_to_external_cluster in cluster_avg_sim_dict.items():
        if cluster_id != assigned_cluster_id:
          if avg_sim_to_external_cluster == 0:
            global_node_score_dict[cluster_id] = 1.0
          elif avg_sim_to_assigned_cluster == 0:
            global_node_score_dict[cluster_id] = 0.0
          else:
            global_node_score = avg_sim_to_assigned_cluster/(
              avg_sim_to_assigned_cluster + avg_sim_to_external_cluster)
            assert 0 <= global_node_score <= 1, global_node_score
            global_node_score_dict[cluster_id] = global_node_score
      rec_avg_sim_to_cluster_dict[rec_id] = global_node_score_dict

  # Iterate over the within cluster links (which are the classified
  # matches) and obtain the link scores
  #
  for rec_pair in classified_pairs_set:
    rec_id1, rec_id2 = rec_pair
    cluster_id1 = cluster_per_rec_dict[rec_id1]
    cluster_id2 = cluster_per_rec_dict[rec_id2]
    assert cluster_id1 == cluster_id2

    if local_or_global == 'local':
      within_cluster_link_score = local_link_score(rec_avg_sim_to_cluster_dict,
                                                   rec_id1, rec_id2,
                                                   cluster_id1, cluster_id2)
    else:
      assert local_or_global in ['global:min', 'global:max', 'global:avr']
      within_cluster_link_score = global_link_score(local_or_global,
                                                    rec_avg_sim_to_cluster_dict,
                                                    rec_id1, rec_id2,
                                                    cluster_id1, cluster_id2)

    if likely_class == 'with_likely_class':
      if within_cluster_link_score >= approx_true_match_thresh:
        m_within_sum += within_cluster_link_score
      else:
        assert within_cluster_link_score < approx_true_non_match_thresh
        u_within_sum += (1 - within_cluster_link_score)
    else:
      m_within_sum += within_cluster_link_score
      u_within_sum += (1 - within_cluster_link_score)

  # Only consider across cluster edges that are contained in the
  # similarity graph.
  # Note that we also need to consider the across cluster links which are
  # not contained in the similarity graph for 'u_across' separately
  # (contributes to TN)
  for rec_pair in classified_non_matches_in_graph:
    assert rec_pair not in classified_pairs_set
    assert rec_pair in pairwise_sim_dict
    rec_id1, rec_id2 = rec_pair
    cluster_id1 = cluster_per_rec_dict[rec_id1]
    cluster_id2 = cluster_per_rec_dict[rec_id2]

    if local_or_global == 'local':
      across_cluster_link_score = local_link_score(rec_avg_sim_to_cluster_dict,
                                                   rec_id1, rec_id2,
                                                   cluster_id1, cluster_id2)
    else:
      assert local_or_global in ['global:min', 'global:max', 'global:avr']
      across_cluster_link_score = global_link_score(local_or_global,
                                                    rec_avg_sim_to_cluster_dict,
                                                    rec_id1, rec_id2,
                                                    cluster_id1, cluster_id2)
    if likely_class == 'with_likely_class':
      if across_cluster_link_score >= approx_true_match_thresh:
        m_across_sum += across_cluster_link_score
      else:
        assert across_cluster_link_score < approx_true_non_match_thresh
        u_across_sum += (1 - across_cluster_link_score)
    else:
      m_across_sum += across_cluster_link_score
      u_across_sum += (1 - across_cluster_link_score)

  overall_sim_sum = m_within_sum + u_within_sum + m_across_sum + \
      u_across_sum
  assert overall_sim_sum > 0

  weight = classified_matches_and_pairs_in_graph_count / overall_sim_sum
  est_tp = m_within_sum * weight
  est_fp = u_within_sum * weight
  est_fn = m_across_sum * weight
  est_tn = u_across_sum * weight + classified_non_matches_not_in_graph_count

  assert int(round(est_tp + est_fp + est_fn + est_tn)) == tot_links, (
    est_tp + est_fp + est_fn + est_tn, tot_links)

  print('  Overall cluster quality as per the unsupervised graph structure '
        'based evaluation technique...')
  print(
    f'    TP = {est_tp}, FP = {est_fp}, FN = {est_fn}, TN = {est_tn}')

  # Calculate the precision, recall and F* measure
  prec = utils.precision(est_tp, est_fp)
  reca = utils.recall(est_tp, est_fn)
  fstar = utils.f_star(est_tp, est_fp, est_fn)

  print(f'    Precision = {prec}, Recall = {reca}, F-star = {fstar}')
  unsup_total_time = (time.time() - start_time) + cluster_load_time
  print(f'    Time for graph structure based evaluation '
        f'= {unsup_total_time:.2f}')

  csv_writer.writerow([algo, min_graph_sim, thresh,
                       len(records_per_cluster_dict),
                       len(pairwise_sim_dict),
                       len(classified_pairs_set),
                       across_cluster_link_tot,
                       classified_matches_and_pairs_in_graph_count,
                       classified_non_matches_in_graph_count,
                       classified_non_matches_not_in_graph_count,
                       classified_matches_in_graph_count,
                       classified_matches_not_in_graph_count,
                       sub_method_str, est_tp, est_fp, est_fn,
                       prec, reca, fstar, gt_tp, gt_fp, gt_fn, gt_prec,
                       gt_reca, gt_fstar, unsup_total_time,
                       sup_total_time])

  print('*' * 80)
  print()

csv_file.close()
print('--Done--')