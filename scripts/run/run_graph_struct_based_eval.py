import subprocess
from multiprocessing import Pool

prog_path = '../methods/graph_structure_based_eval.py'


def run_graph_struct_based_eval(db_name, sub_meth_tuple):
  sub_method_str = '-'.join(map(str, sub_meth_tuple))
  f = open(f'{db_name}_{sub_method_str}_output_graph_struct_eval.txt',
           'w')
  ret_val = subprocess.call(['python3', prog_path, db_name,
                             str(sub_meth_tuple)], stdout=f, stderr=f)
  f.close()
  return ret_val, db_name, sub_method_str


if __name__ == '__main__':
  # Configuration list
  #
  conf_list = []
  db_list = ['ios', 'DS_C0', 'musicbrainz', 'cora', 'google_scholar_full',
             'google_scholar_limited']

  # Construct the method configs
  for local_or_global in ['local', 'global:min', 'global:max', 'global:avr']:
    for likely_class in ['with_likely_class', 'without_likely_class']:
      for all_or_graph_edges in ['all_edges', 'graph_edges']:
        for db in db_list:
          conf_list.append((db, (local_or_global, likely_class,
                                 all_or_graph_edges)))

  num_processes = 4  # Update as per server processing power
  with Pool(processes=num_processes) as pool:
    for result in pool.starmap(run_graph_struct_based_eval, conf_list):
      if result[0] == 0:
        print(f'Completed running graph structure based evaluation '
              f'experiments for DB {result[1]}-{result[2]}')
      else:
        print(f'Error!!! Running graph structure based evaluation experiments '
              f'for DB {result[1]}-{result[2]} failed!')
print('--Done--')
