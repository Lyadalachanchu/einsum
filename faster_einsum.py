import math
import string
import time
import random
from collections import Counter

import torch


def parse_faster_einsum(einsum_str, tensors):
    if "->" in einsum_str:
        input, output = einsum_str.split("->")
        output_labels = list(output.strip())
    else:
        input = einsum_str
        output_labels = None

    input_labels = [list(op.strip()) for op in input.split(',')]
    assert len(input_labels) == len(tensors), "Number of inputs specified in str does not match number of tensors"
    return input_labels, output_labels

def einsum_pair(A, B, labels_A, labels_B, output_labels, global_count):
    # TODO: Highlight this interesting mistake
    # to_contract = [a for a in labels_A if a in labels_B]

    # Broadcasting logic
    # TODO: Can make more efficient by counting before adding
    shared = set(labels_A) & set(labels_B)

    contract_dims = [d for d in shared
               if global_count[d] == 2
               and (output_labels is None or d not in output_labels)]
    batch_dims = [d for d in shared if d not in contract_dims]
    free_A_dims = [d for d in labels_A
                   if d not in batch_dims
                   and d not in contract_dims]
    free_B_dims = [d for d in labels_B
                   if d not in batch_dims
                   and d not in contract_dims]

    for d in batch_dims:
        sa = A.shape[labels_A.index(d)]
        sb = B.shape[labels_B.index(d)]
        if not (sa == sb or sa == 1 or sb == 1):
            raise ValueError(f"Incompatible sizes for batch label '{d}': {sa} vs {sb}")

    batch_axes_A = [labels_A.index(d) for d in batch_dims]
    free_axes_A = [labels_A.index(d) for d in free_A_dims]
    contract_axes_A = [labels_A.index(d) for d in contract_dims]

    batch_axes_B = [labels_B.index(d) for d in batch_dims]
    contract_axes_B = [labels_B.index(d) for d in contract_dims]
    free_axes_B = [labels_B.index(d) for d in free_B_dims]



    # Change axes order to allow (future) matrix multiplication (ie. [Free axes, Contract axes] @ [Contract axes, Free axes])
    perm_A = A.permute(batch_axes_A + free_axes_A + contract_axes_A)
    perm_B = B.permute(batch_axes_B + contract_axes_B + free_axes_B)

    bA_shape = [perm_A.shape[i] for i in range(len(batch_axes_A))]
    bB_shape = [perm_B.shape[i] for i in range(len(batch_axes_B))]

    fA_shape = [perm_A.shape[len(batch_axes_A) + i] for i in range(len(free_axes_A))]
    c_shape  = [perm_A.shape[len(batch_axes_A) + len(free_axes_A) + i] for i in range(len(contract_axes_A))]
    fB_shape = [perm_B.shape[len(batch_axes_B) + len(contract_axes_B) + i] for i in range(len(free_axes_B))]

    fA_prod = math.prod(fA_shape) if fA_shape else 1
    c_prod  = math.prod(c_shape)  if c_shape  else 1
    fB_prod = math.prod(fB_shape) if fB_shape else 1

    A_mat = perm_A.reshape(*bA_shape, fA_prod, c_prod)   # (..., m, n)
    B_mat = perm_B.reshape(*bB_shape, c_prod, fB_prod)   # (..., n, p)

    C_mat = A_mat@B_mat
    C = C_mat.reshape(*C_mat.shape[:-2], *fA_shape, *fB_shape)
    C_labels = batch_dims + free_A_dims + free_B_dims

    return C, C_labels

# WARNING: The tensor updates are in-place. This might mean that tensors might be changed afterwards.
def estimate_contraction_cost(tensor_A_shape, tensor_B_shape, labels_A, labels_B, output_labels, global_count):
    shared = set(labels_A) & set(labels_B)
    contract_dims = [d for d in shared
                    if global_count[d] == 2
                    and (output_labels is None or d not in output_labels)]
    batch_dims = [d for d in shared if d not in contract_dims]
    free_A_dims = [d for d in labels_A if d not in batch_dims and d not in contract_dims]
    free_B_dims = [d for d in labels_B if d not in batch_dims and d not in contract_dims]
    
    # Calculate output tensor size
    output_size = 1
    # Batch dimensions
    for d in batch_dims:
        idx_A = labels_A.index(d)
        idx_B = labels_B.index(d)
        output_size *= max(tensor_A_shape[idx_A], tensor_B_shape[idx_B])
    
    # Free dimensions from both tensors
    for d in free_A_dims:
        output_size *= tensor_A_shape[labels_A.index(d)]
    for d in free_B_dims:
        output_size *= tensor_B_shape[labels_B.index(d)]
    
    return output_size

def find_best_contraction_pair(tensors, label_lists, output_labels, global_count):
    min_cost = float('inf')
    best_pair = (0, 1)
    
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            cost = estimate_contraction_cost(
                tensors[i].shape, tensors[j].shape,
                label_lists[i], label_lists[j],
                output_labels, global_count
            )
            if cost < min_cost:
                min_cost = cost
                best_pair = (i, j)
    
    return best_pair

def faster_einsum(einsum_str, tensors, use_greedy=True):
    input_labels, output_labels = parse_faster_einsum(einsum_str, tensors)
    tensors = list(tensors)  # Make a copy so we can modify the list
    label_lists = input_labels[:]
    
    if not use_greedy:
        # Original left-to-right strategy
        left_tensor, intermediate_labels = tensors[0], input_labels[0]
        for i in range(1, len(tensors)):
            global_count = Counter(sum(label_lists, []))
            left_tensor, intermediate_labels = einsum_pair(left_tensor, tensors[i], intermediate_labels, input_labels[i], output_labels, global_count)
            label_lists = [intermediate_labels] + label_lists[2:]
    else:
        # Greedy strategy
        while len(tensors) > 1:
            global_count = Counter(sum(label_lists, []))
            i, j = find_best_contraction_pair(tensors, label_lists, output_labels, global_count)
            
            # Contract the chosen pair
            result, new_labels = einsum_pair(tensors[i], tensors[j], label_lists[i], label_lists[j], output_labels, global_count)
            
            # Remove the contracted tensors and their labels (larger index first)
            tensors.pop(max(i, j))
            tensors.pop(min(i, j))
            label_lists.pop(max(i, j))
            label_lists.pop(min(i, j))
            
            # Add the result back
            tensors.insert(0, result)
            label_lists.insert(0, new_labels)
        
        left_tensor = tensors[0]
        intermediate_labels = label_lists[0]

    # See if there are extra axes to reduce based on output shape
    sum_reduce_axes = [i for i, contract in enumerate(intermediate_labels) if contract not in output_labels]
    if len(sum_reduce_axes) > 0:
        result = left_tensor.sum(dim=sum_reduce_axes)
    else:
        result = left_tensor
    # transpose to match the ordering specified in output_labels
    # eg. ikjr -> ikj -> jik
    for axis in sorted(sum_reduce_axes, reverse=True):
        del intermediate_labels[axis]
    return result.permute(*[output_labels.index(label) for label in intermediate_labels])


if __name__ == "__main__":
    # Compare different implementations and strategies
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    tensor_c = torch.rand(6, 5, 2, 1)
    example_str = "jilw, jekw, tkwu-> ie"

    # Test left-to-right strategy
    start_0 = time.perf_counter()
    result_basic = faster_einsum(example_str, [tensor_a, tensor_b, tensor_c], use_greedy=False)
    end_0 = time.perf_counter()
    time_basic = end_0 - start_0

    # Test greedy strategy
    start_1 = time.perf_counter()
    result_greedy = faster_einsum(example_str, [tensor_a, tensor_b, tensor_c], use_greedy=True)
    end_1 = time.perf_counter()
    time_greedy = end_1 - start_1

    # Compare with PyTorch's einsum
    start_2 = time.perf_counter()
    answer = torch.einsum(example_str, [tensor_a, tensor_b, tensor_c])
    end_2 = time.perf_counter()
    time_torch = end_2 - start_2

    print(f"Duration left-to-right strategy: {time_basic:.6f}s")
    print(f"Duration greedy strategy: {time_greedy:.6f}s")
    print(f"Duration PyTorch einsum: {time_torch:.6f}s")
    print(f"\nOutput shapes: {result_basic.shape}")
    
    # Verify all implementations give the same result
    assert torch.allclose(result_basic, answer, rtol=1e-04, atol=1e-05), "Left-to-right strategy gave incorrect result"
    assert torch.allclose(result_greedy, answer, rtol=1e-04, atol=1e-05), "Greedy strategy gave incorrect result"