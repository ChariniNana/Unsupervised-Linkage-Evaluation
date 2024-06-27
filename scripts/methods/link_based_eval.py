"""
This program conducts link based unsupervised evaluation of clustering results.
There are two main techniques considered here, which are as follows.

1) Count the within cluster links above a clustering threshold 0.5 as tp, below
   0.5 as fp, above 0.5 across clusters as fn, and below 0.5 across clusters
   as tn (Equations 8 in the paper).

2) The second method is also based on the above logic. However, rather than
   simply taking a count, the similarity of edges is taken into account to
   reflect the confidence of links being tp, fp, tn, anf fn (Equations 9 in
   the paper)


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
csv_file = open(f'link_based_eval_clusters_{db_name}_results.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
  ['clustering_algo', 'graph_base_thresh', 'clustering_thresh',
   'no_of_clusters', 'full_similarity_graph_size |E_S|',
   'no_of_predicted_matches |E_T| (edges_in_clusters)',
   'no_of_predicted_non_matches = n*(n-1)/2 - |E_T|',
   'no_of_predicted_non_matches_in_graph = |E_S - E_T|',
   'no_of_predicted_non_matches_not_in_graph = n*(n-1)/2 - |E_S.union(E_T)|',
   'no_of_predicted_matches_in_graph = |E_S & E_T|',
   'no_of_predicted_matches_not_in_graph = |E_T - E_S|',
   'link_based_method', 'est_tp', 'est_fp', 'est_fn',
   'prec', 'reca', 'fstar', 'gt_tp', 'gt_fp', 'gt_fn',
   'gt_prec', 'gt_reca', 'gt_fstar',
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
  for link_based_meth in ['count', 'score_with_likely_class',
                          'score_without_likely_class']:

    sim_val_dict = {'m_within': [], 'u_within': [],
                    'm_across': [], 'u_across': []}  # Dictionary to hold
    # similarity values of links, where link within cluster and
    # sim >= approx_true_match_thresh, link within cluster and
    # sim < approx_true_non_match_thresh, link across cluster and
    # sim >= approx_true_match_thresh, and link across cluster and
    # sim < approx_true_non_match_thresh, are stored as m_within,
    # u_within, m_across, and u_across, respectively.

    tot_links = int(len(cluster_per_rec_dict) * (len(cluster_per_rec_dict)-1)/2)
    print(f'  Total possible links = {tot_links}')
    min_graph_sim = min(pairwise_sim_dict.values())
    print(f'  Minimum graph pair similarity = {min_graph_sim}')

    within_cluster_link_tot = len(classified_pairs_set)
    across_cluster_link_tot = tot_links - within_cluster_link_tot

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
    classified_non_matches_not_in_graph_count = tot_links - len(
      set(pairwise_sim_dict.keys()).union(classified_pairs_set))

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

    print(f'Running unsupervised link-based evaluation experiments for '
          f'method {link_based_meth}, on clustering results obtained with '
          f'algorithm {algo} and clustering similarity threshold {thresh}...')
    print()
    start_time = time.time()

    # Iterate over the within cluster links (which are the classified matches)
    # and obtain the pairwise similarities
    #
    for rec_pair in classified_matches_in_graph:
      assert rec_pair in pairwise_sim_dict
      within_cluster_link_sim = pairwise_sim_dict[rec_pair]

      if link_based_meth in ['count', 'score_with_likely_class']:
        if within_cluster_link_sim >= approx_true_match_thresh:
          sim_val_dict['m_within'].append(within_cluster_link_sim)
        else:
          assert within_cluster_link_sim < approx_true_non_match_thresh
          sim_val_dict['u_within'].append(within_cluster_link_sim)
      else:  # Each link within a cluster contributes to both TP and FP
        sim_val_dict['m_within'].append(within_cluster_link_sim)
        sim_val_dict['u_within'].append(within_cluster_link_sim)
    if link_based_meth in ['count', 'score_with_likely_class']:
      assert len(sim_val_dict['m_within']) + len(sim_val_dict['u_within']) + \
             classified_matches_not_in_graph_count == \
             within_cluster_link_tot, (len(sim_val_dict['m_within']),
                                       len(sim_val_dict['u_within']),
                                       classified_matches_not_in_graph_count,
                                       within_cluster_link_tot)

    # Consider all across cluster link similarities. Note that we also need to
    # consider the across cluster links which are not contained in the
    # similarity graph for 'u_across' (contributes to TN)
    for rec_pair in classified_non_matches_in_graph:
      assert rec_pair not in classified_pairs_set
      assert rec_pair in pairwise_sim_dict
      across_cluster_link_sim = pairwise_sim_dict[rec_pair]
      if link_based_meth in ['count', 'score_with_likely_class']:
        if across_cluster_link_sim >= approx_true_match_thresh:
          sim_val_dict['m_across'].append(across_cluster_link_sim)
        else:
          assert across_cluster_link_sim < approx_true_non_match_thresh
          sim_val_dict['u_across'].append(across_cluster_link_sim)
      else:  # Each link across a cluster pair contributes to both FN and TN
        sim_val_dict['m_across'].append(across_cluster_link_sim)
        sim_val_dict['u_across'].append(across_cluster_link_sim)

    if link_based_meth in ['count', 'score_with_likely_class']:
      assert len(sim_val_dict['m_across']) + len(sim_val_dict['u_across']) + \
             classified_non_matches_not_in_graph_count == \
             across_cluster_link_tot, (len(sim_val_dict['m_across']),
                                       len(sim_val_dict['u_across']),
                                       classified_non_matches_not_in_graph_count,
                                       across_cluster_link_tot)

    if link_based_meth == 'count':
      est_tp = len(sim_val_dict['m_within'])
      est_fp = len(sim_val_dict['u_within']) + \
          classified_matches_not_in_graph_count
      est_fn = len(sim_val_dict['m_across'])
      est_tn = len(sim_val_dict['u_across']) + \
          classified_non_matches_not_in_graph_count

    else:
      assert link_based_meth in ['score_with_likely_class',
                                 'score_without_likely_class'], link_based_meth

      score_dict = utils.get_scores(sim_val_dict)

      m_within_sum = sum(score_dict['m_within'])
      u_within_sum = sum(score_dict['u_within'])
      m_across_sum = sum(score_dict['m_across'])
      u_across_sum = sum(score_dict['u_across'])

      overall_sim_sum = m_within_sum + u_within_sum + m_across_sum + \
          u_across_sum

      weight = len(pairwise_sim_dict) / overall_sim_sum
      est_tp = m_within_sum * weight
      est_fp = u_within_sum * weight + classified_matches_not_in_graph_count
      est_fn = m_across_sum * weight
      est_tn = u_across_sum * weight + \
          classified_non_matches_not_in_graph_count

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
                         classified_non_matches_in_graph_count,
                         classified_non_matches_not_in_graph_count,
                         classified_matches_in_graph_count,
                         classified_matches_not_in_graph_count,
                         link_based_meth, est_tp, est_fp, est_fn, prec, reca,
                         fstar, gt_tp, gt_fp, gt_fn, gt_prec, gt_reca,
                         gt_fstar, unsup_total_time, sup_total_time])

    print('-' * 80)
    print()
  print('*' * 80)
  print()

csv_file.close()
print('--Done--')
