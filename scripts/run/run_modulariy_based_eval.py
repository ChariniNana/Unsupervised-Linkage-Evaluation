import subprocess
from multiprocessing import Pool

prog_path = '../methods/modularity_method_eval.py'


def run_mod_based_eval(db_name):
  f = open(f'{db_name}_output_modularity_eval.txt', 'w')
  ret_val = subprocess.call(['python3', prog_path, db_name], stdout=f, stderr=f)
  f.close()
  return ret_val, db_name


if __name__ == '__main__':
  # Configuration list
  #
  conf_list = ['ios', 'DS_C0', 'musicbrainz', 'cora', 'google_scholar_full',
               'google_scholar_limited']
  num_processes = len(conf_list)  # Update as per server processing power
  with Pool(processes=num_processes) as pool:
    for result in pool.map(run_mod_based_eval, conf_list):
      if result[0] == 0:
        print(f'Completed running modularity-based evaluation experiments for '
              f'DB {result[1]}')
      else:
        print(f'Error!!! Running modularity-based evaluation experiments for '
              f'DB {result[1]} failed!')
print('--Done--')
