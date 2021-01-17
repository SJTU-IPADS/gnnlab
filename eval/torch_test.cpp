#include <iostream>

#include <torch/torch.h>

int main() {
    torch::Tensor val = torch::ones(
        {10},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)
    ).to("cuda:0");

    std::cout << val << std::endl;
}
