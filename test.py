# Two inputs, no transposes
import time

import torch

import sum_reductions
from faster_einsum import faster_einsum
from subscript_parser import parse_einsum
from two_tensor_faster_einsum import faster_einsum as two_tensor_faster_einsum


def test_two_inputs_no_transposes_broadcasting():
    tensor_a = torch.rand(3)
    tensor_b = torch.rand(3, 2, 2)
    example_str = "j, jkl-> k"

    print("------------Test two inputs no transpose-------------")
    barebones_time, einsum_time = test_barebones(tensor_a, tensor_b, example_str)
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    two_tensor_custom_time, _ = two_tensor_test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])

    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration barebones einsum: {barebones_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")
    print(f"Duration two tensor einsum: {two_tensor_custom_time}")

def test_two_inputs_duplicates():
    tensor_a = torch.rand(3, 3)
    tensor_b = torch.rand(3, 2)
    example_str = "ji, jk-> jik"

    print("------------Test two inputs duplicates-------------")
    
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])

    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")

def test_two_inputs_no_transposes():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    example_str = "jilw, jekw-> ik"

    print("------------Test two inputs no transpose-------------")
    barebones_time, einsum_time = test_barebones(tensor_a, tensor_b, example_str)
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    two_tensor_custom_time, _ = two_tensor_test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])

    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration barebones einsum: {barebones_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")
    print(f"Duration matrix mult. einsum: {two_tensor_custom_time}")


def test_two_inputs():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    example_str = "jilw, jekw-> ki"
    print("------------Test two inputs-------------")
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    two_tensor_custom_time, _ = two_tensor_test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")
    print(f"Duration two tensor einsum: {two_tensor_custom_time}")

def test_two_inputs_broadcasting():
    tensor_a = torch.rand(3, 2)
    tensor_b = torch.rand(3)
    example_str = "ij, i -> j"
    print("------------Test two inputs (with broadcasting)-------------")
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    two_tensor_custom_time, _ = two_tensor_test_custom_matmul_einsum(example_str, [tensor_a, tensor_b])
    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")
    print(f"Duration two tensor einsum: {two_tensor_custom_time}")

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
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")


def test_three_inputs_no_transpose():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    tensor_c = torch.rand(6, 5)
    example_str = "jilw, jekw, tk-> ik"
    print("------------Test three inputs no transpose-------------")
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")

def test_three_inputs():
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    tensor_c = torch.rand(6, 5)
    example_str = "jilw, jekw, tk-> kil"
    print("------------Test three inputs-------------")
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    print(f"Duration PyTorch einsum: {torch_time}")
    print(f"Duration left-to-right einsum: {basic_time}")
    print(f"Duration greedy einsum: {greedy_time}")


def test_barebones(tensor_a, tensor_b, example_str):
    start_1 = time.perf_counter()
    result = sum_reductions.barebones_einsum(tensor_a, tensor_b,
                                             *parse_einsum(example_str, [tensor_a, tensor_b]))

    # TODO: Try faster einsum (two tensors)
    end_1 = time.perf_counter()

    start_2 = time.perf_counter()
    answer = torch.einsum(example_str, [tensor_a, tensor_b])
    end_2 = time.perf_counter()


    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05)
    return end_1-start_1, end_2-start_2

def test_custom_matmul_einsum(example_str, tensors):
    # Test PyTorch einsum
    start_1 = time.perf_counter()
    answer = torch.einsum(example_str, tensors)
    end_1 = time.perf_counter()
    torch_time = end_1 - start_1

    # Test left-to-right strategy
    start_0 = time.perf_counter()
    result = faster_einsum(example_str, tensors, use_greedy=False)
    end_0 = time.perf_counter()
    basic_time = end_0 - start_0

    # Test greedy strategy
    start_2 = time.perf_counter()
    result_greedy = faster_einsum(example_str, tensors, use_greedy=True)
    end_2 = time.perf_counter()
    greedy_time = end_2 - start_2

    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05), "Left-to-right strategy gave incorrect result"
    assert torch.allclose(result_greedy, answer, rtol=1e-04, atol=1e-05), "Greedy strategy gave incorrect result"
    return basic_time, greedy_time, torch_time

def two_tensor_test_custom_matmul_einsum(example_str, tensors):
    start_1 = time.perf_counter()
    answer = torch.einsum(example_str, tensors)
    end_1 = time.perf_counter()

    start_0 = time.perf_counter()
    result = two_tensor_faster_einsum(example_str, tensors)
    end_0 = time.perf_counter()

    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05)
    return end_0-start_0, end_1-start_1




# Two inputs, with transposes

# Three inputs, with transposes

def test_greedy_optimization_case():
    # Create tensors where the contraction order matters significantly
    # A: (10, 2000, 30)  - large middle dimension
    # B: (30, 40, 10)    - shares small dimensions with A
    # C: (40, 50)        - shares dimension with B
    
    # If we contract A and B first (left-to-right):
    # 1. A x B -> (10, 2000, 40) - large intermediate result
    # 2. Result x C -> (10, 2000, 50) - final result
    
    # If we contract B and C first (greedy should find this):
    # 1. B x C -> (30, 10, 50) - smaller intermediate result
    # 2. A x Result -> (10, 2000, 50) - same final result
    
    tensor_a = torch.rand(10, 2000, 30)
    tensor_b = torch.rand(30, 40, 10)
    tensor_c = torch.rand(40, 50)
    example_str = "ijk,kli,lm->ijm"
    
    print("------------Test Greedy Optimization Case-------------")
    
    basic_time, greedy_time, torch_time = test_custom_matmul_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    
    print(f"Duration PyTorch einsum: {torch_time:.6f}s")
    print(f"Duration left-to-right einsum: {basic_time:.6f}s")
    print(f"Duration greedy einsum: {greedy_time:.6f}s")
    print(f"Speedup from greedy: {basic_time/greedy_time:.2f}x")

if __name__ == "__main__":
    test_greedy_optimization_case()
    test_greedy_optimization_case()
    test_two_inputs_no_transposes_broadcasting()
    test_three_inputs_simple()
    test_two_inputs_duplicates()
    test_two_inputs()
    test_two_inputs_no_transposes()
    test_two_inputs_broadcasting()
    test_three_inputs()
    test_three_inputs_no_transpose()
