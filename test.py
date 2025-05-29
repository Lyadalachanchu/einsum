# Two inputs, no transposes
import time

import torch

import sum_reductions
from faster_einsum import faster_einsum
from subscript_parser import parse_einsum


def test_two_inputs_no_transposes_broadcasting():
    tensor_a = torch.rand(3)
    tensor_b = torch.rand(3, 2, 2)
    example_str = "j, jkl-> k"

    print("------------Test two inputs no transpose-------------")
    barebones_time, einsum_time = test_barebones(tensor_a, tensor_b, example_str)
    custom_time, _ = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])

    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration barebones einsum: {barebones_time}")
    print(f"Duration better barebones einsum: {custom_time}")

def test_two_inputs_duplicates():
    tensor_a = torch.rand(3, 3)
    tensor_b = torch.rand(3, 2)
    example_str = "ji, jk-> jik"

    print("------------Test two inputs no transpose-------------")
    custom_time, einsum_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])

    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration better barebones einsum: {custom_time}")

def test_two_inputs_no_transposes():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    example_str = "jilw, jekw-> ik"

    print("------------Test two inputs no transpose-------------")
    barebones_time, einsum_time = test_barebones(tensor_a, tensor_b, example_str)
    custom_time, _ = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])

    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration barebones einsum: {barebones_time}")
    print(f"Duration better barebones einsum: {custom_time}")



def test_two_inputs():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    example_str = "jilw, jekw-> ki"
    print("------------Test two inputs-------------")
    custom_time, einsum_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration better barebones einsum: {custom_time}")

def test_two_inputs_broadcasting():
    tensor_a = torch.rand(3, 2)
    tensor_b = torch.rand(3)
    example_str = "ij, i -> j"
    print("------------Test two inputs (with broadcasting)-------------")
    custom_time, einsum_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration better barebones einsum: {custom_time}")

def test_three_inputs_simple():
    tensor_a = torch.rand(3, 2)
    tensor_b = torch.rand(3, 3)
    tensor_c = torch.rand(3, 4)
    example_str = "ji, jk, jl -> ikl"

    # This should be split into contractions like:
    # ji, jk -> jik
    # j should not be contracted yet since the third tensor also has j
    # jik, jl -> ikl
    print("------------Test three inputs (simple)-------------")
    custom_time, einsum_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration better barebones einsum: {custom_time}")


def test_three_inputs_no_transpose():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    tensor_c = torch.rand(6, 5)
    example_str = "jilw, jekw, tk-> ik"
    print("------------Test three inputs no transpose-------------")
    custom_time, einsum_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration better barebones einsum: {custom_time}")

def test_three_inputs():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    tensor_c = torch.rand(6, 5)
    example_str = "jilw, jekw, tk-> kil"
    print("------------Test three inputs no transpose-------------")
    custom_time, einsum_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    print(f"Duration actual einsum: {einsum_time}")
    print(f"Duration better barebones einsum: {custom_time}")


def test_barebones(tensor_a, tensor_b, example_str):
    start_1 = time.perf_counter()
    result = sum_reductions.barebones_einsum(tensor_a, tensor_b,
                                             *parse_einsum(example_str, [tensor_a, tensor_b]))
    end_1 = time.perf_counter()

    start_2 = time.perf_counter()
    answer = torch.einsum(example_str, [tensor_a, tensor_b])
    end_2 = time.perf_counter()


    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05)
    return end_1-start_1, end_2-start_2

def test_custom_matmul_einsum(example_str, tensors):
    start_1 = time.perf_counter()
    answer = torch.einsum(example_str, tensors)
    end_1 = time.perf_counter()

    start_0 = time.perf_counter()
    result = faster_einsum(example_str, tensors)
    end_0 = time.perf_counter()

    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05)
    return end_0-start_0, end_1-start_1




# Two inputs, with transposes

# Three inputs, with transposes

if __name__ == "__main__":
    test_three_inputs()
    test_two_inputs_no_transposes_broadcasting()
    test_three_inputs_simple()
    test_two_inputs_duplicates()
    test_two_inputs()
    test_two_inputs_broadcasting()
    test_three_inputs_no_transpose()
