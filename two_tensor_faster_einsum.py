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

def einsum_pair(A, B, labels_A, labels_B, output_labels):
    shared = set(labels_A) & set(labels_B)

    contract_dims = [d for d in shared
               if (output_labels is None or d not in output_labels)]
    free_A_dims = [d for d in labels_A
                   if d not in contract_dims]
    free_B_dims = [d for d in labels_B
                   if d not in contract_dims]

    free_axes_A = [labels_A.index(d) for d in free_A_dims]
    contract_axes_A = [labels_A.index(d) for d in contract_dims]

    contract_axes_B = [labels_B.index(d) for d in contract_dims]
    free_axes_B = [labels_B.index(d) for d in free_B_dims]



    # Change axes order to allow (future) matrix multiplication (ie. [Free axes, Contract axes] @ [Contract axes, Free axes])
    perm_A = A.permute(free_axes_A + contract_axes_A)
    perm_B = B.permute(contract_axes_B + free_axes_B)

    fA_shape = [perm_A.shape[i] for i in range(len(free_axes_A))]
    c_shape = [perm_A.shape[len(free_axes_A) + i]
               for i in range(len(contract_axes_A))]
    fB_shape = [perm_B.shape[len(contract_axes_B) + i]
                for i in range(len(free_axes_B))]

    fA_prod = math.prod(fA_shape)
    c_prod = math.prod(c_shape)
    fB_prod = math.prod(fB_shape)

    A_mat = perm_A.reshape(fA_prod, c_prod)
    B_mat = perm_B.reshape(c_prod, fB_prod)

    # Use the efficient matmul
    C_mat = A_mat@B_mat
    C = C_mat.reshape(*fA_shape, *fB_shape)
    C_labels = free_A_dims + free_B_dims

    return C, C_labels

# WARNING: The tensor updates are in-place. This might mean that tensors might be changed afterwards.
def faster_einsum(einsum_str, tensors):
    input_labels, output_labels = parse_faster_einsum(einsum_str, tensors)
    intermediate_tensor, intermediate_labels = einsum_pair(tensors[0], tensors[1], input_labels[0], input_labels[1], output_labels)

    # See if there are extra axes to reduce based on output shape
    sum_reduce_axes = [i for i, contract in enumerate(intermediate_labels) if contract not in output_labels]
    if len(sum_reduce_axes) > 0:
        for idx in sorted(sum_reduce_axes, reverse=True):
            del intermediate_labels[idx]
        result = intermediate_tensor.sum(dim=sum_reduce_axes)
    else:
        result = intermediate_tensor
    return result.permute(*[output_labels.index(label) for label in intermediate_labels])


if __name__ == "__main__":
    # Compare with two inputs speed across different implementations
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    example_str = "jilw, jekw -> ik"

    start_0 = time.perf_counter()
    result = faster_einsum(example_str, [tensor_a, tensor_b])
    end_0 = time.perf_counter()

    # start_1 = time.perf_counter()
    # result = sum_reductions.barebones_einsum(tensor_a, tensor_b,
    #                                 *parse_einsum(example_str, [tensor_a, tensor_b]))
    # end_1 = time.perf_counter()

    start_2 = time.perf_counter()
    answer = torch.einsum(example_str, [tensor_a, tensor_b])
    end_2 = time.perf_counter()

    print(f"Duration our einsum: {end_0 - start_0}")
    # print(f"Duration old einsum (barebones_einsum): {end_1 - start_1}")
    print(f"Duration actual einsum: {end_2 - start_2}")
    print(result.shape, answer.shape)
    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05)