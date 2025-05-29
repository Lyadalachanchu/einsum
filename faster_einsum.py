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
    both = set(labels_A) | set(labels_B)
    while len(labels_A) < len(labels_B):
        A.unsqueeze_(-1)
        rand_char = random.choice([ch for ch in string.ascii_lowercase if ch not in both])
        both.add(rand_char)
        labels_A.append(rand_char)

    while len(labels_B) < len(labels_A):
        B.unsqueeze_(-1)
        rand_char = random.choice([ch for ch in string.ascii_lowercase if ch not in both])
        both.add(rand_char)
        labels_B.append(rand_char)


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

    batch_axes_A = [labels_A.index(d) for d in batch_dims]
    free_axes_A = [labels_A.index(d) for d in free_A_dims]
    contract_axes_A = [labels_A.index(d) for d in contract_dims]

    batch_axes_B = [labels_B.index(d) for d in batch_dims]
    contract_axes_B = [labels_B.index(d) for d in contract_dims]
    free_axes_B = [labels_B.index(d) for d in free_B_dims]



    # Change axes order to allow (future) matrix multiplication (ie. [Free axes, Contract axes] @ [Contract axes, Free axes])
    perm_A = A.permute(batch_axes_A + free_axes_A + contract_axes_A)
    perm_B = B.permute(batch_axes_B + contract_axes_B + free_axes_B)

    b_shape = [perm_A.shape[i] for i in range(len(batch_axes_A))]
    fA_shape = [perm_A.shape[len(batch_axes_A) + i] for i in range(len(free_axes_A))]
    c_shape = [perm_A.shape[len(batch_axes_A) + len(free_axes_A) + i]
               for i in range(len(contract_axes_A))]
    fB_shape = [perm_B.shape[len(batch_axes_B) + len(contract_axes_B) + i]
                for i in range(len(free_axes_B))]

    fA_prod = math.prod(fA_shape)  # 2*4 = 8
    c_prod = math.prod(c_shape)  # 3
    fB_prod = math.prod(fB_shape)  # 6*1 = 6

    A_mat = perm_A.reshape(*b_shape, fA_prod, c_prod)
    B_mat = perm_B.reshape(*b_shape, c_prod, fB_prod)


    # print(f"perm_A.shape: {perm_A.shape}, labels_A: {labels_A}, free_labels_A: {free_A_dims}, labels_B: {labels_B}, free_labels_B: {free_B_dims}, perm_B.shape: {perm_B.shape}")
    # print(A_mat.shape, B_mat.shape)
    # Use the efficient matmul
    # TODO: Write a custom simple kernel for funsies?
    C_mat = A_mat@B_mat
    C = C_mat.reshape(*b_shape, *fA_shape, *fB_shape)
    C_labels = batch_dims + free_A_dims + free_B_dims

    return C, C_labels

# WARNING: The tensor updates are in-place. This might mean that tensors might be changed afterwards.
def faster_einsum(einsum_str, tensors):
    input_labels, output_labels = parse_faster_einsum(einsum_str, tensors)

    # For now, we just go left to right
    left_tensor, intermediate_labels = tensors[0], input_labels[0]
    label_lists = input_labels[:]
    for i in range(1, len(tensors)):
        global_count = Counter(sum(label_lists, []))
        # print(intermediate_labels, input_labels[i])
        left_tensor, intermediate_labels = einsum_pair(left_tensor, tensors[i], intermediate_labels, input_labels[i], output_labels, global_count)
        label_lists = [intermediate_labels] + label_lists[2:]

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

    # eg. ikj -> jik; i->1, k->2, j->0
    # print(intermediate_labels)
    return result.permute(*[output_labels.index(label) for label in intermediate_labels])


if __name__ == "__main__":
    # Compare with two inputs speed across different implementations
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    # No broadcasting support (yet?)
    tensor_c = torch.rand(6, 5, 2, 1)
    example_str = "jilw, jekw, tkwu-> ie"

    start_0 = time.perf_counter()
    result = faster_einsum(example_str, [tensor_a, tensor_b, tensor_c])
    end_0 = time.perf_counter()

    # start_1 = time.perf_counter()
    # result = sum_reductions.barebones_einsum(tensor_a, tensor_b,
    #                                 *parse_einsum(example_str, [tensor_a, tensor_b]))
    # end_1 = time.perf_counter()

    start_2 = time.perf_counter()
    answer = torch.einsum(example_str, [tensor_a, tensor_b, tensor_c])
    end_2 = time.perf_counter()

    print(f"Duration our einsum: {end_0 - start_0}")
    # print(f"Duration old einsum (barebones_einsum): {end_1 - start_1}")
    print(f"Duration actual einsum: {end_2 - start_2}")
    print(result.shape, answer.shape)
    assert torch.allclose(result, answer, rtol=1e-04, atol=1e-05)