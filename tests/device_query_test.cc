#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

#include "test_common/common.h"

TEST(DeviceTestQuery, Peer2PeerTest) {
  int access1from0, access0from1;
  CUDA_CALL(cudaDeviceCanAccessPeer(&access1from0, 0, 1));
  CUDA_CALL(cudaDeviceCanAccessPeer(&access0from1, 1, 0));

  LOG << "access1from0: " << access1from0 << " | access0from1: " << access0from1
      << "\n";
}

TEST(DeviceTestQuery, AsyncTest) {
  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  int device;
  for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;

    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));

    LOG << "Device " << device << " has " << deviceProp.asyncEngineCount
        << " async engines\n";
  }
}

TEST(DeviceTestQuery, HostMallocTest) {}
