#include <iostream>

#include <torch/torch.h>

int main() {
    torch::Tensor left = torch::ones(
        {10},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)
    ).to("cuda:0");

    torch::Tensor right = torch::ones(
        {10},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)
    ).to("cuda:0");

    torch::Tensor val = left + right;

    std::cout << val << std::endl;
}
