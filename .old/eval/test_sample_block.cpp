#include <gflags/gflags.h>

#include "util/cuda.hpp"

#include "data/dataset.hpp"
#include "sampling/cuda/wrapper.hpp"

DEFINE_string(dataset_key, "papers100M","");
DEFINE_string(dataset_folder, "/graph-learning/preprocess/papers100M", "");

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("some usage message");
    gflags::SetVersionString("1.0.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(FLAGS_dataset_key, FLAGS_dataset_folder);

    auto csr = dataset->GetCSR();

    unsigned int num_input = 5;
    unsigned int seed = 6;
    unsigned int fanout = 5;
    uint32_t input [] = {0, 2, 4, 100};
    uint32_t *output = (uint32_t *) malloc(num_input * fanout * sizeof(uint32_t));

    uint32_t *d_indptr;
    uint32_t *d_indices;
    uint32_t *d_input;
    uint32_t *d_output;

    CUDA_CALL(cudaMalloc(&d_indptr, (csr->num_rows + 1) * sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc(&d_indices, csr->num_edges * sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc(&d_input, num_input * sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc(&d_output, num_input * fanout * sizeof(uint32_t)));

    CUDA_CALL(cudaMemcpy(d_indptr, csr->indptr, (csr->num_rows + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_indices, csr->indices, csr->num_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_input, input, num_input * sizeof(uint32_t), cudaMemcpyHostToDevice));

    sampleBlockWrapper(d_indptr, d_indices, d_input, d_output, num_input, fanout, seed);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaMemcpy(output, d_output, num_input * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_indptr));
    CUDA_CALL(cudaFree(d_indices));
    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_output));

    for (int i = 0; i < num_input; i++) {
        for (int j = 0; j < fanout; j++) {
            printf("%u\t", output[i * fanout + j]);
        }
        printf("\n");
    }

    free(output);
}
