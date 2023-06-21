#!/bin/sh

if [ -n "$1" ]; then
  CXX=$1
fi

if [ -n "$2" ]; then
  CXXFLAGS=$2
fi

# use /tmp as fallback for systems without /dev/shm
# /tmp is slow so using it only if /dev/shm is not available
tmpdir=/tmp
if [ -w "/dev/shm" ]; then
  tmpdir=/dev/shm
fi
tmpfile="$(mktemp --suffix=.cpp ${tmpdir}/nvshmem.XXXXXXXXX)"

cat >${tmpfile} <<EOL
#include <iostream>
    int main(void) {
#if __cplusplus < 201103L
    std::cout << "ISO C++ 2011 or newer is required to build with NVTX support! (-std=c++11)" << std::endl;
#endif
    return 0;
}
EOL

${CXX} ${CXXFLAGS} ${tmpfile} -o conftest_cxx11
./conftest_cxx11

rm -f conftest_cxx11 ${tmpfile}
