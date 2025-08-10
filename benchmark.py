import time
import torch
from faster_einsum import faster_einsum

def time_einsum(example_str, tensors, num_runs=5):
    """Time all three implementations multiple times and return averages"""
    torch_times = []
    basic_times = []
    greedy_times = []
    
    for _ in range(num_runs):
        # Time PyTorch einsum
        start = time.perf_counter()
        _ = torch.einsum(example_str, tensors)
        torch_times.append(time.perf_counter() - start)
        
        # Time left-to-right strategy
        start = time.perf_counter()
        _ = faster_einsum(example_str, tensors, use_greedy=False)
        basic_times.append(time.perf_counter() - start)
        
        # Time greedy strategy
        start = time.perf_counter()
        _ = faster_einsum(example_str, tensors, use_greedy=True)
        greedy_times.append(time.perf_counter() - start)
    
    return (sum(torch_times)/num_runs, 
            sum(basic_times)/num_runs, 
            sum(greedy_times)/num_runs)

def benchmark_size_scaling():
    """Test how performance scales with tensor size (keeping number of tensors constant)"""
    sizes = [4, 8, 16, 32, 64, 128, 256]
    print("\n------------Benchmark Size Scaling (2 tensors)-------------")
    print("Testing with einsum 'ij,jk->ik' while increasing dimension sizes")
    print("Size | PyTorch | Left-to-Right | Greedy")
    print("-" * 50)
    
    for size in sizes:
        # Matrix multiplication case
        tensor_a = torch.rand(size, size)
        tensor_b = torch.rand(size, size)
        example_str = "ij,jk->ik"
        
        torch_time, basic_time, greedy_time = time_einsum(example_str, [tensor_a, tensor_b])
        print(f"{size:4d} | {torch_time:.6f} | {basic_time:.6f} | {greedy_time:.6f}")

def benchmark_num_tensors():
    """Test how performance scales with number of input tensors"""
    num_tensors = range(2, 7)  # Test with 2 to 6 tensors
    sizes = [8, 16, 32]  # Test with different sizes
    
    for size in sizes:
        print(f"\n------------Benchmark Number of Tensors (size={size})-------------")
        print("Testing with chain matrix multiplication pattern (ij,jk,kl,...)")
        print("Num | PyTorch | Left-to-Right | Greedy")
        print("-" * 50)
        
        for n in num_tensors:
            # Create n tensors of size (size,size)
            tensors = [torch.rand(size, size) for _ in range(n)]
            # Create einsum string like "ij,jk,kl,lm->im" for n tensors
            indices = [chr(ord('i') + i) + chr(ord('i') + i + 1) for i in range(n)]
            example_str = ",".join(indices) + "->" + chr(ord('i')) + chr(ord('i') + n)
            
            torch_time, basic_time, greedy_time = time_einsum(example_str, tensors)
            print(f"{n:3d} | {torch_time:.6f} | {basic_time:.6f} | {greedy_time:.6f}")

def benchmark_extreme_unequal():
    """Test extreme cases of dimension size differences"""
    print("\n------------Benchmark Extreme Unequal Dimensions-------------")
    print("Testing with three tensors: 'ij,jk,kl->il'")
    print("Case | PyTorch | Left-to-Right | Greedy | Description")
    print("-" * 80)
    
    test_cases = [
        # (i, j, k, l, description)
        ((2, 1024, 2, 1024), "Very thin tensors with one large dim"),
        ((1024, 2, 1024, 2), "Very wide tensors with one small dim"),
        ((2, 2, 1024, 1024), "Small to large progression"),
        ((1024, 1024, 2, 2), "Large to small progression"),
        ((2, 512, 512, 2), "Diamond shape (small-large-large-small)"),
        ((512, 2, 2, 512), "Hourglass shape (large-small-small-large)"),
    ]
    
    for dims, desc in test_cases:
        i, j, k, l = dims
        tensor_a = torch.rand(i, j)
        tensor_b = torch.rand(j, k)
        tensor_c = torch.rand(k, l)
        example_str = "ij,jk,kl->il"
        
        torch_time, basic_time, greedy_time = time_einsum(example_str, [tensor_a, tensor_b, tensor_c])
        print(f"{desc[:20]:20s} | {torch_time:.6f} | {basic_time:.6f} | {greedy_time:.6f} | {desc}")

def burn_in():
    """Run some operations to ensure PyTorch is fully loaded"""
    # Run a mix of different sized operations
    a = torch.rand(10, 100)
    b = torch.rand(100, 10)
    _ = torch.einsum('ij,jk->ik', [a, b])

if __name__ == "__main__":
    # Burn in run
    burn_in()
    
    print("Running benchmarks...")
    benchmark_size_scaling()
    benchmark_num_tensors()
    benchmark_extreme_unequal()