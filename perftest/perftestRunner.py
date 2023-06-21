#!/usr/bin/env python3
import os
import time
import sys
from subprocess import Popen
import perftestCommon
import logging

NPE_START_END_STEP = [1,2,4,8,16]

def post_process(nvshmem_install_path, perftest_install_path, ftesto, fteste):
  Popen(['echo', 'nvidia-smi\r\n'], stdout=fteste)
  Popen(['nvidia-smi'], stdout=fteste)
  time.sleep(5)
  Popen(['echo', '\r\n'], stdout=fteste)
  Popen(['echo', 'nvidia-smi topo -m\r\n'], stdout=fteste)
  Popen(['nvidia-smi', 'topo', '-m'], stdout=fteste)
  time.sleep(5)
  Popen(['echo', '\r\n'], stdout=fteste)
  if 'CUDA_HOME' in os.environ:
    Popen(['echo', os.environ['CUDA_HOME']+'\r\n'], stdout=fteste)
  else:
    Popen(['echo', 'CUDA_HOME not set\r\n'], stdout=fteste)
  if 'MPI_HOME' in os.environ:
    Popen(['echo', os.environ['MPI_HOME']+'\r\n'], stdout=fteste)
  else:
    Popen(['echo', 'MPI_HOME not set\r\n'], stdout=fteste)
  Popen(['echo', os.environ['PATH']+'\r\n'], stdout=fteste)
  Popen(['echo', os.environ['LD_LIBRARY_PATH']+'\r\n'], stdout=fteste)
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    Popen(['echo', os.environ['CUDA_VISIBLE_DEVICES']+'\r\n'], stdout=fteste)
  else:
    Popen(['echo', 'CUDA_VISIBLE_DEVICES not set\r\n'], stdout=fteste)
  Popen(['ldd', nvshmem_install_path+'/bin/nvshmrun.hydra'], stdout=fteste)
  time.sleep(5)
  Popen(['echo', 'Failed tests : \r\n'], stdout=fteste)
  for binary_cmdline in perftestCommon.failed_binary_cmdlines_list:
    Popen(['echo', perftest_install_path+binary_cmdline[1]+'\r\n'], stdout=fteste)
    Popen(['echo', 'ldd '+perftest_install_path+binary_cmdline[0]+'\r\n'], stdout=fteste)
    Popen(['ldd', perftest_install_path+binary_cmdline[0]], stdout=fteste)
    time.sleep(5)


if __name__ == '__main__':
  logger = logging.getLogger('[perftestRunner]')

  args_list = sys.argv[1:]
  if "--nvshmem" in args_list or "--perftest_install" in args_list or "--help" in args_list:
    import argparse

    parser = argparse.ArgumentParser(
      description="NVSHMEM Performance Test Runner", epilog=None,
      formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
      "--mpirun", nargs='?', dest="mpi_install_path",
      default="/usr/local/openmpi_nvshmem", help='MPI HOME')

    parser.add_argument(
      "--nvshmem", nargs='?', dest="nvshmem_install_path",
      help='NVSHMEM HOME')

    parser.add_argument(
      "--launcher", nargs='?', dest="launcher_choice",
      choices=['all', 'mpirun', 'shmem', 'nvshmem'],
      required=True,
      help='Launcher choice: all/shmem/nvshmem/mpirun')

    parser.add_argument(
      "--perftest_install", nargs='?', dest="perftest_install_path",
      required=True, help='The localcation of perf_install folder.')

    parser.add_argument(
      "--test_list", nargs='?', dest="test_list_name",
      required=True, help='Test list file name in the folder.')

    parser.add_argument(
      "--pe", nargs='?', dest="max_pes_per_node",
      required=True, help='Max PEs per node. Int.')

    parser.add_argument(
      "--hosts", nargs='?', dest="hosts",
      required=True, help='Hosts, ip/hostname which can ssh directly. Use comma in different hosts.')

    parser.add_argument(
      "--timeout", nargs='?', dest="timeout",
      required=True, help='Timeout value for each case, unit is second.')

    parser.add_argument(
      "-a", nargs='?', dest="extra_parameters_string",
      required=False, help='Extra parameters. For example -a "NVSHMEM_DEBUG=INFO"')

    parser.add_argument(
      "--partial", dest="partial", action="store_true",
      required=False, help='partial')

    parser.add_argument(
      "--gpubind", nargs='?', dest="bind_scr",
      required=False, help='Bind a script to cmdline to do something. For example, bind gpu to processes.')

    args = parser.parse_args()
    enable_skip = 0

    # Args parser
    if args.launcher_choice is not None:
      launcher = args.launcher_choice
      if launcher == "all":
        launcher_choice = 1
      elif launcher == "shmem":
        launcher_choice = 2
      elif launcher == "nvshmem":
        launcher_choice = 3
      elif launcher == "mpirun":
        launcher_choice = 0
      else:
        logger.error("Unsupported launcher...")
        sys.exit()
      launcher = "-%s" % launcher
      logger.info("launcher_choice is %s" % launcher_choice)
    else:
      logger.info("--launcher <name> required.")
      sys.exit(255)

    if args.mpi_install_path is not None:
      mpi_install_path = args.mpi_install_path
    elif launcher_choice != 3:
      logger.error("--mpirun is required for all/mpirun/shmem launcher. ")
    else:
      pass

    if args.nvshmem_install_path is not None:
      nvshmem_install_path = args.nvshmem_install_path
    elif launcher_choice == 1 or launcher_choice == 3:
      logger.error("--nvshmem is required for all/nvshmem launcher.")
      sys.exit(255)
    else:
      nvshmem_install_path = ""

    if args.perftest_install_path is not None:
      perftest_install_path = args.perftest_install_path

    if args.test_list_name is not None:
      test_list_name = args.test_list_name

    if args.max_pes_per_node is not None:
      max_pes_per_node = int(args.max_pes_per_node)

    if args.hosts is not None:
      hosts = args.hosts

    if args.timeout is not None:
      timeout = int(args.timeout)

    if args.extra_parameters_string is not None:
      extra_parameters_string = args.extra_parameters_string
      logger.info("Will add extra parameters: \"%s\" in perftest commands." % extra_parameters_string)
    else:
      extra_parameters_string = ""

    if args.bind_scr is not None:
      bind_scr = args.bind_scr
      if os.access(bind_scr, os.F_OK):
        logger.info("Find script file: %s" % bind_scr)
      else:
        logger.error("Failed to find %s" % bind_scr)
        sys.exit(245)
    else:
      bind_scr = ""
    os.environ["GPUBIND_SCRIPT"] = bind_scr

    if args.partial:
      try:
        from bullet import colors
        from bullet import Check, keyhandler, styles
        from bullet.charDef import NEWLINE_KEY
      except Exception as e:
        print("Please pip3 install bullet and use python3.")
        logger.info(e)
        sys.exit(255)

      class MinMaxCheck(Check):
        def __init__(self, min_selections=0, max_selections=None, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.min_selections = min_selections
          self.max_selections = max_selections
          if max_selections is None:
            self.max_selections = len(self.choices)

        @keyhandler.register(NEWLINE_KEY)
        def accept(self):
          if self.valid():
              return super().accept()

        def valid(self):
          return self.min_selections <= sum(1 for c in self.checked if c) <= self.max_selections

      logger.info("Please select cases...")
      cases_list=[]
      with open(test_list_name, 'r') as read_list:
        for i in read_list:
          cases_list.append(i.strip("\n"))
      # logger.info(cases_list)

      client = MinMaxCheck(
        prompt = "\nChoose 1 or more from the list, then type enter: ",
        min_selections = 1,
        max_selections = 999,
        return_index = True,
        choices = cases_list,
        check_color = colors.foreground["red"],
        check_on_switch = colors.foreground["red"],
        word_color = colors.foreground["black"],
        word_on_switch = colors.foreground["black"],
        background_color = colors.background["white"],
        background_on_switch = colors.background["yellow"]
      )

      logger.info('\n', end = '')
      result = client.launch()
      # logger.info(result)

      read_list.close()
      with open("%s_partial" % test_list_name, 'w') as partial_list:
        for r in result[0]:
          partial_list.write("%s\n" % r)
      partial_list.close()

      test_list_name = "%s_partial" % test_list_name
    else:
      logger.info("Run all tests in the list file: %s" % test_list_name)

    logger.info("TEST LIST:")
    os.system('cat %s' % test_list_name)
  else:
    mpi_install_path = '/usr/local/openmpi-3.0.1'
    nvshmem_install_path = '/usr/local/nvshmem'
    perftest_install_path = '/usr/local/nvshmem-perftest'
    launcher_choice = 0
    test_list_name = 'perftest-p2p.list'
    max_pes_per_node = 8
    hosts = 'localhost'
    timeout = 60
    enable_skip = 0
    extra_parameters_string = ""

    if (len(sys.argv) == 1) or (len(sys.argv) == 9) or (len(sys.argv) == 10):
      if (len(sys.argv) == 9) or (len(sys.argv) == 10):
        mpi_install_path = sys.argv[1]
        nvshmem_install_path = sys.argv[2]
        perftest_install_path = sys.argv[3]
        launcher_choice = int(sys.argv[4])
        test_list_name = sys.argv[5]
        max_pes_per_node = int(sys.argv[6])
        hosts = sys.argv[7]
        timeout = int(sys.argv[8])
        launcher=""

      if (len(sys.argv) == 10):
        extra_parameters_string = sys.argv[9]
        logger.info("Will add extra parameters: \"%s\" in perftest commands." % extra_parameters_string)

    else:
      logger.error('Include libmpi.so and libcudart.so in LD_LIBRARY_PATH and provide these 8 or 9 arguments - \
          1) MPI install path, 2) NVSHMEM install path, 3) perftest install path, 4) all launchers or single launcher (1/0), \
          5) test list name (P2P-PCIE, PCP-NVLink, IB) 6) maximum available GPUs per node to run test, 7) comma separated host list, 8) timeout (in seconds) per test, \
          9) Extra parameters(Eg. NCCL_INFO=WARN)')
      sys.exit()

  with open('perftest-'+time.strftime("%Y%m%d-%H%M%S")+launcher+'.out', 'w') as ftesto:
    with open('perftest-'+time.strftime("%Y%m%d-%H%M%S")+launcher+'.err', 'w') as fteste:

      perftestCommon.walk_dir(nvshmem_install_path, mpi_install_path, perftest_install_path,  launcher_choice, NPE_START_END_STEP, max_pes_per_node, hosts, timeout,
                              enable_skip, test_list_name, extra_parameters_string, ftesto, fteste)
      if perftestCommon.failed_binary_cmdlines_list:
        post_process(nvshmem_install_path, perftest_install_path, ftesto, fteste)

  logger.info("OUT file is %s" % ftesto.name)
  logger.info("ERR file is %s" % fteste.name)
