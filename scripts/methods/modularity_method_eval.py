"""
This program conducts unsupervised evaluation of clustering results using
a method which is inspired by the approach proposed by the technique proposed
by Raad et al (2018).
There are two main techniques considered here, which are as follows.

Similar to the link based technique, here we classify record pairs as
likely matches and non matches based on a 0.5 threshold. Then, based on the
link similarity and a weighting, we determine the likelihood for a link to be
a TP, FP, TN or FN, and calculate the total TP, FP, TN and FN counts based on
that.

The weights are calculated as follows:

1) For within cluster links, w_i = sum of all similarities for links in
   cluster i / total number of links in cluster i
2) For across cluster links, w_i,j = sum of all similarities for links across
   clusters i and j / total number of links across clusters i and j

Next, the likelihood for each link to contribute to a tp, fp, tn or fn is
calculated based on both the pairwise similarity and cluster weight (as
shown in Equations 13 in the paper).

@Author: Charini
@Year: 2024
"""

import time
import csv
import sys
import utils.eval_utils as utils

assert len(sys.argv) == 2
db_name = sys.argv[1]
assert db_name in ['google_scholar_full', 'google_scholar_limited', 'ios',
                   'DS_C0', 'cora', 'musicbrainz'], db_name

sim_graph_type = 'all'

pairwise_sim_dict, gt_pairs_set = utils.load_sim_graph_and_gt(db_name)

# Open a csv file to write results
csv_file = open(f'modularity_eval_clusters_{db_name}_results.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
  ['clustering_algo', 'graph_base_thresh', 'clustering_thresh',
   'no_of_clusters', 'full_similarity_graph_size |E_S|',
   'no_of_predicted_matches |E_T| (edges_in_clusters)',
   'no_of_predicted_non_matches = n*(n-1)/2 - |E_T|',
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

  approx_true_match_thresh = 0.5
  approx_true_non_match_thresh = 0.5

  for likely_class_classification in ['with_likely_class',
                                      'without_likely_class']:
    m_within_sum = 0
    u_within_sum = 0
    m_across_sum = 0
    u_across_sum = 0

    cluster_sim_dict = {}  # A dictionary to store the pairwise similarity
    # values within or across clusters. The key is a sorted cluster ID pair.
    # For within cluster similarities, the ID is repeated twice.

    cluster_weight_dict = {}  # A dictionary to store the weights per cluster
    # or cluster pair. The keys are similar to cluster_sim_dict.

    tot_links = int(len(cluster_per_rec_dict) * (len(cluster_per_rec_dict)-1)/2)
    print(f'  Total possible links = {tot_links}')
    min_graph_sim = min(pairwise_sim_dict.values())
    print(f'  Minimum graph pair similarity = {min_graph_sim}')

    within_cluster_link_tot = len(classified_pairs_set)
    across_cluster_link_tot = tot_links - within_cluster_link_tot

    # Calculate the number of record pairs in both the similarity graph and
    # classification (E_S \cup E_T)
    classified_matches_and_pairs_in_graph_count = len(
      set(pairwise_sim_dict.keys()).union(classified_pairs_set))

    # Calculate the number of classified non matches in the similarity graph
    #
    classified_non_matches_in_graph = set(pairwise_sim_dict.keys()) - \
        classified_pairs_set
    classified_non_matches_in_graph_count = len(classified_non_matches_in_graph)
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
    classified_matches_not_in_graph_count = len(classified_matches_not_in_graph)

    assert len(classified_matches_in_graph) + \
           len(classified_non_matches_in_graph) == len(pairwise_sim_dict), \
    (len(classified_matches_in_graph), len(classified_non_matches_in_graph),
     len(pairwise_sim_dict))

    print(f'Running unsupervised link-based evaluation experiments for method '
          f'{likely_class_classification}, on clustering results obtained with '
          f'algorithm {algo} and clustering similarity threshold {thresh}...')
    print()
    start_time = time.time()

    # Iterate through the record pairs in the pairwise similarity graph and
    # get the similarities within and across clusters
    #
    for rec_pair in pairwise_sim_dict:
      assert rec_pair in classified_matches_in_graph or \
             rec_pair in classified_non_matches_in_graph
      if rec_pair in classified_matches_in_graph:
        # The record pair appears in a single cluster
        cluster_id1 = cluster_per_rec_dict[rec_pair[0]]
        cluster_id2 = cluster_per_rec_dict[rec_pair[1]]
        assert cluster_id1 == cluster_id2, (cluster_id1, cluster_id2)
      else:
        # The record pair appears in two different clusters
        cluster_id1 = cluster_per_rec_dict[rec_pair[0]]
        cluster_id2 = cluster_per_rec_dict[rec_pair[1]]
        assert cluster_id1 != cluster_id2, (cluster_id1, cluster_id2, rec_pair)
      rec_pair_sim = pairwise_sim_dict[rec_pair]
      sorted_cluster_id_pair = tuple(sorted([cluster_id1, cluster_id2]))
      sim_list = cluster_sim_dict.get(sorted_cluster_id_pair, [])
      sim_list.append(rec_pair_sim)
      cluster_sim_dict[sorted_cluster_id_pair] = sim_list

    # Calculate the weights within and across clusters
    #
    for cluster_pair, sim_list in cluster_sim_dict.items():
      assert len(sim_list) > 0
      if cluster_pair[0] == cluster_pair[1]:  # Within cluster weight
        cluster_rec_count = len(records_per_cluster_dict[cluster_pair[0]])
        tot_cluster_link_count = cluster_rec_count * (cluster_rec_count-1)/2
      else:  # Across cluster weight
        tot_cluster_link_count = \
          len(records_per_cluster_dict[cluster_pair[0]]) * \
          len(records_per_cluster_dict[cluster_pair[1]])
      cluster_weight = sum(sim_list)/tot_cluster_link_count
      assert type(cluster_weight) == float
      assert cluster_pair not in cluster_weight_dict
      cluster_weight_dict[cluster_pair] = cluster_weight

    # Iterate over the within cluster links (which are the classified
    # matches) and obtain the pairwise similarities
    #
    for rec_pair in classified_pairs_set:
      within_cluster_link_sim = pairwise_sim_dict.get(rec_pair, 0)
      cluster_id1 = cluster_per_rec_dict[rec_pair[0]]
      cluster_id2 = cluster_per_rec_dict[rec_pair[1]]
      assert cluster_id1 == cluster_id2
      sorted_cluster_id_pair = (cluster_id1, cluster_id2)
      cluster_weight = cluster_weight_dict[sorted_cluster_id_pair]

      if likely_class_classification == 'with_likely_class':
        if within_cluster_link_sim >= approx_true_match_thresh:
          m_within_sum += within_cluster_link_sim * cluster_weight
        else:
          assert within_cluster_link_sim < approx_true_non_match_thresh
          u_within_sum += (1-within_cluster_link_sim) * (1-cluster_weight)
      else:  # Each link within a cluster contributes to both TP and FP
        m_within_sum += within_cluster_link_sim * cluster_weight
        u_within_sum += (1 - within_cluster_link_sim) * (1 - cluster_weight)

    # Consider all across cluster link similarities. Note that we also need
    # to consider the across cluster links which are not contained in the
    # similarity graph for 'u_across' (contributes to TN)
    for rec_pair in classified_non_matches_in_graph:
      assert rec_pair not in classified_pairs_set
      assert rec_pair in pairwise_sim_dict
      across_cluster_link_sim = pairwise_sim_dict[rec_pair]
      cluster_id1 = cluster_per_rec_dict[rec_pair[0]]
      cluster_id2 = cluster_per_rec_dict[rec_pair[1]]
      assert cluster_id1 != cluster_id2
      sorted_cluster_id_pair = tuple(sorted([cluster_id1, cluster_id2]))
      cluster_weight = cluster_weight_dict[sorted_cluster_id_pair]

      if likely_class_classification == 'with_likely_class':
        if across_cluster_link_sim >= approx_true_match_thresh:
          m_across_sum += across_cluster_link_sim * cluster_weight
        else:
          assert across_cluster_link_sim < approx_true_non_match_thresh
          u_across_sum += (1-across_cluster_link_sim) * (1-cluster_weight)
      else:  # Each link across a cluster pair contributes to both FN and TN
        m_across_sum += across_cluster_link_sim * cluster_weight
        u_across_sum += (1-across_cluster_link_sim) * (1-cluster_weight)

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

    print('  Overall cluster quality as per the unsupervised link-based '
          'evaluation technique...')
    print(f'    TP = {est_tp}, FP = {est_fp}, FN = {est_fn}, TN = {est_tn}')

    # Calculate the precision, recall and F* measure
    prec = utils.precision(est_tp, est_fp)
    reca = utils.recall(est_tp, est_fn)
    fstar = utils.f_star(est_tp, est_fp, est_fn)

    print(f'    Precision = {prec}, Recall = {reca}, F-star = {fstar}')
    unsup_total_time = (time.time() - start_time) + cluster_load_time
    print(f'    Time for link-based evaluation '
          f'= {unsup_total_time:.2f}')

    csv_writer.writerow([algo, min_graph_sim, thresh,
                         len(records_per_cluster_dict), len(pairwise_sim_dict),
                         len(classified_pairs_set), across_cluster_link_tot,
                         classified_matches_and_pairs_in_graph_count,
                         classified_non_matches_in_graph_count,
                         classified_non_matches_not_in_graph_count,
                         classified_matches_in_graph_count,
                         classified_matches_not_in_graph_count,
                         likely_class_classification, est_tp, est_fp, est_fn,
                         prec, reca, fstar, gt_tp, gt_fp, gt_fn, gt_prec,
                         gt_reca, gt_fstar, unsup_total_time, sup_total_time])

    print('-' * 80)
    print()
  print('*' * 80)
  print()

csv_file.close()
print('--Done--')
