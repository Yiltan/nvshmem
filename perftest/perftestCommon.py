#!/usr/bin/env python3
import sys
import os
import time
from subprocess import Popen, PIPE
from threading import Thread

test_process = 0
failed_binary_cmdlines_list = []
NVSHMEM_LAUNCHER = 0
MPI_LAUNCHER = 1
SHMEM_LAUNCHER = 2

def to_bytes(s):
  if type(s) is bytes:
    return s
  elif type(s) is str or (sys.version_info[0] < 3 and type(s) is unicode):
    return codecs.encode(s, 'utf-8')
  else:
    raise TypeError("Expected bytes or string, but got %s." % type(s))

def display_time(func):
  def wrapper(*args):
    t1 = time.time()
    req = func(*args)
    t2 = time.time()
    print('Total time {:.4}s'.format(t2 - t1))
    return req
  return wrapper

def report_failure(cmd_line, test_path, ftesto, fteste):
  global failed_tests_list
  Popen(['echo', ' '.join([str(elem) for elem in cmd_line]) + ' failed\r\n'], stdout=fteste)
  failed_binary_cmdlines_list.append((test_path, str(cmd_line)))
  return

def get_all_tests(ftestlist):
  tests_set = []
  skipped_tests_set = []
  with open(ftestlist, 'r') as f:
    for line in f:
      if line.startswith("#"):
        skipped_tests_set.append(line[1:-1].strip())
        tests_set.append(line[1:-1].strip())
      else:
        tests_set.append(line.strip())
  return (tests_set, skipped_tests_set)

def get_args_combinations_pe_range(full_test_path, npe_start_end_step, max_pes):
  args_combs = []
  npe_range_ = list(npe_start_end_step)
  npe_range = [npe for npe in npe_range_ if npe <= max_pes]
  if 'pt-to-pt' in full_test_path:
    npe_range[0] = 2
    if 1 < len(npe_range):
      elemsDelCnt = len(npe_range) - 1
      for cnt in range(0, elemsDelCnt):
        del npe_range[-1]
#TODO : delete the test of this def
    full_args_path = full_test_path + '.args'
    if not os.path.isfile(full_args_path):
      return (args_combs, npe_range)
    with open(full_args_path) as f:
      lines = f.readlines()
      for i in range(1, len(lines)):
        if lines[i]:
          args_combs.append(lines[i])
    return (args_combs, npe_range)
  full_args_path = full_test_path+'.args'
  if not os.path.isfile(full_args_path):
    return (args_combs, npe_range)
  firstline = None
  secondline = None
  with open(full_args_path) as f:
    firstline, secondline = f.readline(), f.readline()
    vals = map(int, firstline.split())
    idx = 0
    for v in vals:
      if v <= max_pes:
        npe_range[idx] = v #XXX:all perf tests need at least 2 avail GPUs
      else:
        break
      idx += 1
    if idx < len(npe_range):
      elemsDelCnt = len(npe_range) - idx
      for cnt in range(0, elemsDelCnt):
        del npe_range[-1]
    if secondline:
      args_combs.append(secondline)
    for line in f:
      args_combs.append(line)
  return (args_combs, npe_range)

def get_env_combinations(full_test_path):
  envs = []
  env_combs = []
  full_env_path = full_test_path+'.env'
  if not os.path.isfile(full_env_path):
    return env_combs 
  with open(full_env_path) as f:
    for line in f:
      envs.append(line)
  for e in envs:
    env_combs.append(e.split())
  return env_combs

def thread_func(cmd_line, ftesto, fteste):
  global test_process
  print(' '.join([str(elem) for elem in cmd_line]))
  test_process = Popen(['echo', 'Running ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n'], stdout=ftesto)
  test_process = Popen(cmd_line, stdout=ftesto, stderr=fteste)
  test_process.communicate()
  return

@display_time
def run_cmd(cmd_line, test_path, timeout, ftesto, fteste):
  th = Thread(target=thread_func, args=(cmd_line, ftesto, fteste))
  th.start()
  th.join(timeout)
  if th.is_alive():
    Popen(['echo', 'Timed out ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n'], stdout=fteste)
    test_process.terminate()
    th.join()
    report_failure(cmd_line, test_path, ftesto, fteste)
  if test_process.returncode:
    p = Popen(['echo', 'EXPECTING PASSED, GOT FAILURE'], stdout=PIPE)
    print(to_bytes(p.communicate()[0]).decode('utf-8'))
    report_failure(cmd_line, test_path, ftesto, fteste)
  else:
    p = Popen(['echo', 'PASSED'], stdout=PIPE)
    print(to_bytes(p.communicate()[0]).decode('utf-8'))
    
  cmd = 'rm'
  args = "%s*" % '/dev/shm/nvshmem-shm'
  Popen("%s %s" % (cmd, args), shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
  return

def run_cmd_given_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_all, ppn, timeout, launcher_choice, ftesto, fteste):
  cmd_line = cmd_line_prefix[:]
  cmd_line.append(str(npe_all))
  if launcher_choice == NVSHMEM_LAUNCHER:
    cmd_line.append('-ppn')
  elif launcher_choice == MPI_LAUNCHER or launcher_choice == SHMEM_LAUNCHER:
    cmd_line.append('-npernode')
  cmd_line.append(str(ppn))

  bind_scr = os.environ.get("GPUBIND_SCRIPT")
  if bind_scr != "" and bind_scr is not None:
    cmd_line.append("%s" % bind_scr)

  cmd_line.append(full_test_path)
  if cmd_line_suffix:
    cmd_line.append(cmd_line_suffix)
  run_cmd(cmd_line, full_test_path.replace(test_install_path, ''), timeout, ftesto, fteste)

def run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste):
  cmd_line = cmd_line_prefix[:]
  if 'pt-to-pt' in full_test_path:
    if nhosts == 1:
      ppn = 2
      npe_all = 2
    else:
      ppn = 1
      npe_all = 2
    run_cmd_given_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_all, ppn, timeout, launcher_choice, ftesto, fteste)
  elif 'coll' in full_test_path or 'init' in full_test_path:
    if nhosts == 1:
      npe_range_ = npe_range[1:]
    else:
      npe_range_ = npe_range
    for npe in npe_range_:
      ppn = npe
      npe_all = nhosts*ppn
      run_cmd_given_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_all, ppn, timeout, launcher_choice, ftesto, fteste)
  return

def enumerate_env_lines(env_combs, cmd_line_suffix, nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste):
  nhosts = hosts.count(",")+1
  if 'CUDA_HOME' in os.environ:
    cuda_install_path=os.environ['CUDA_HOME']
  else:
    print('CUDA_HOME not set')
    sys.exit()
  if 'GDRCOPY_HOME' in os.environ:
    gdrcopy_install_path=os.environ['GDRCOPY_HOME']
  else:
    print('GDRCOPY_HOME not set, will not use gdrcopy')

  if 'NCCL_HOME' in os.environ:
    nccl_install_lib=":%s/lib64:%s/lib" % (os.environ['NCCL_HOME'], os.environ['NCCL_HOME'])
  else:
    nccl_install_lib=""

  if 'PMIX_HOME' in os.environ:
    pmix_install_lib=":%s/lib" % os.environ['PMIX_HOME']
  else:
    pmix_install_lib=""

  if env_combs:
    for combidx in range(0, len(env_combs)):
      if launcher_choice == NVSHMEM_LAUNCHER:
        cmd_line_prefix = [nvshmem_install_path+'/bin/nvshmrun.hydra', '--bind-to', 'none', '--launcher', 'ssh', '--hosts', hosts]
        extra_parameters = extra_parameters_string.split()
        first_e = cmd_line_prefix.index("--launcher")
        for item in extra_parameters[::-1]:
          cmd_line_prefix.insert(first_e, "-genv=%s" % item)

        for envidx in range(0, len(env_combs[0]), 2):
          var = env_combs[combidx][envidx]
          val = env_combs[combidx][envidx + 1]
          cmd_line_prefix.append('-genv')
          cmd_line_prefix.append(var)
          cmd_line_prefix.append(val)
        cmd_line_prefix.append('-n')
        run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
      if launcher_choice == MPI_LAUNCHER:
        cmd_line_prefix = [mpi_install_path+'/bin/mpirun', '--mca', 'btl', '^uct', '--allow-run-as-root', '-oversubscribe', '--bind-to', 'none', '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+'/lib64'+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+':$LD_LIBRARY_PATH', '-x', 'NVSHMEMTEST_USE_MPI_LAUNCHER=1' , '--host', hosts]
        extra_parameters = extra_parameters_string.split()
        first_x = cmd_line_prefix.index("-x")
        for item in extra_parameters[::-1]:
          cmd_line_prefix.insert(first_x, item)
          cmd_line_prefix.insert(first_x, "-x")
          
        for envidx in range(0, len(env_combs[0]), 2):
          var = env_combs[combidx][envidx]
          val = env_combs[combidx][envidx + 1]
          cmd_line_prefix.append('-x')
          cmd_line_prefix.append(var+'='+val)
        cmd_line_prefix.append('-n')
        run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
      if launcher_choice == SHMEM_LAUNCHER:
        cmd_line_prefix = [mpi_install_path+'/bin/oshrun', '--mca', 'btl', '^uct',  '--allow-run-as-root', '-oversubscribe', '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+'/lib64'+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+':$LD_LIBRARY_PATH', '-x', 'NVSHMEMTEST_USE_SHMEM_LAUNCHER=1' , '--host', hosts]
        first_x = cmd_line_prefix.index("-x")
        for item in extra_parameters[::-1]:
          cmd_line_prefix.insert(first_x, item)
          cmd_line_prefix.insert(first_x, "-x")        
        for envidx in range(0, len(env_combs[0]), 2):
          var = env_combs[combidx][envidx]
          val = env_combs[combidx][envidx + 1]
          cmd_line_prefix.append('-x')
          cmd_line_prefix.append(var+'='+val)
        cmd_line_prefix.append('-n')
        run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
  else:
    if launcher_choice == NVSHMEM_LAUNCHER:
      cmd_line_prefix = [nvshmem_install_path+'/bin/nvshmrun.hydra', '--bind-to', 'none', '--launcher', 'ssh', '--hosts', hosts, '-n']
      extra_parameters = extra_parameters_string.split()
      first_e = cmd_line_prefix.index("--launcher")
      for item in extra_parameters[::-1]:
        cmd_line_prefix.insert(first_e, "-genv=%s" % item)
      run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
    if launcher_choice == MPI_LAUNCHER:
      cmd_line_prefix = [mpi_install_path+'/bin/mpirun', '--mca', 'btl', '^uct', '--allow-run-as-root', '-oversubscribe', '--bind-to', 'none', '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+'/lib64'+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+':$LD_LIBRARY_PATH', '-x', 'NVSHMEMTEST_USE_MPI_LAUNCHER=1', '--host', hosts, '-n']
      extra_parameters = extra_parameters_string.split()
      first_x = cmd_line_prefix.index("-x")
      for item in extra_parameters[::-1]:
        cmd_line_prefix.insert(first_x, item)
        cmd_line_prefix.insert(first_x, "-x")
      run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
    if launcher_choice == SHMEM_LAUNCHER:
      cmd_line_prefix = [mpi_install_path+'/bin/oshrun', '--mca', 'btl', '^uct', '--allow-run-as-root', '-oversubscribe', '--bind-to', 'none', '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+'/lib64'+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+':$LD_LIBRARY_PATH', '-x', 'NVSHMEMTEST_USE_SHMEM_LAUNCHER=1', '--host', hosts, '-n']
      extra_parameters = extra_parameters_string.split()
      first_x = cmd_line_prefix.index("-x")
      for item in extra_parameters[::-1]:
        cmd_line_prefix.insert(first_x, item)
        cmd_line_prefix.insert(first_x, "-x")
      run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
  return

def enumerate_args_lines(args_combs, env_combs, nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste):
  if args_combs:
    for args in args_combs:
      cmd_line_suffix = args.rstrip()
      enumerate_env_lines(env_combs, cmd_line_suffix, nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste)
  else:
    enumerate_env_lines(env_combs, '', nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste)
  return

def walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, tests_set, skipped_tests_set, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste):
  for test_path in tests_set:
    full_test_path = os.path.join(test_install_path, test_path.lstrip(os.path.sep))
    if enable_skip and (test_path in skipped_tests_set):
      Popen(['echo', (full_test_path)+' found in list and skipped\r\n'], stdout=ftesto)
      continue
    if not os.access(full_test_path, os.X_OK):
      Popen(['echo', (full_test_path)+' found in list and binary missing\r\n'], stdout=fteste)
      continue
    env_combs = get_env_combinations(full_test_path)
    tup = get_args_combinations_pe_range(full_test_path, npe_start_end_step, max_pes)
    enumerate_args_lines(tup[0], env_combs, nvshmem_install_path, test_install_path, full_test_path, tup[1], hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste)
  return

def walk_dir(nvshmem_install_path, mpi_install_path, test_install_path, launcher_choice, npe_start_end_step, max_pes, hosts, timeout, enable_skip, ftestlist_any_launcher, extra_parameters_string, ftesto, fteste):
  stup = get_all_tests(ftestlist_any_launcher)
  if len(stup[0]) != 0:
    if launcher_choice == 1:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], NVSHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], MPI_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], SHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    elif launcher_choice == 2:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], SHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    elif launcher_choice == 3:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], NVSHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    elif launcher_choice == 0:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], MPI_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    else:
      print("Please select launcher use 0/1/2/3. [1: Three launchers, 0: mpirun, 2: openshmem, 3: nvshmem]")
  return
