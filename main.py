from itertools import product
from timeit import dummy_src_name

import torch

# def custom_einsum(a, b, ind_a_names, ind_b_names):
#     # eg. a.shape (2,3); b.shape (2,3); ind_a_names [i, k]; ind_b_names [j, k]
#     common_index_names = set(ind_a_names) & set(ind_b_names)
#     print(common_index_names)
#     dim_dict = {}
#     for common_index_name in common_index_names:
#         # Finds the first element. Might want to change this later for sum. eg. ii -> i
#         a_ind = ind_a_names.index(common_index_name)
#         b_ind = ind_b_names.index(common_index_name)
#         dim_dict[common_index_name] = (a_ind, b_ind)
#         ind_a_names[a_ind], ind_b_names[b_ind] = "?"
#
#     # initially it looked like [i, k], [j, k]. Now it should look like [i, ?], [j, ?]
#     # iterate through a's dimensions
#     for a_dim, a_dim_name in enumerate(ind_a_names):
#         if a_dim_name == "?": continue
#         for b_dim, b_dim_name in enumerate(ind_b_names):
#             if b_dim_name == "?": continue
#             for common_index_name in common_index_names:
#                 a_common_dim, b_common_dim = dim_dict[common_index_name]
#                 for i in range(a.shape[a_dim]):
#                     for j in range(b.shape[b_dim]):
#                         total = 0
#                         for k in range(a.shape[a_common_dim]):
#                             total += a[i, k] * b[j, k]
#



if __name__ == "__main__":
    a = torch.rand(2,3,4)
    b = torch.rand(1,3,1)
    result = torch.einsum("ikl, jkr-> ijlr", a, b)
    print(result)

    common_dim = 1
    result_shape = (2, 1, 4, 1)
    result = torch.zeros(result_shape)

    # Assume the common dim is in the same position for a and b
    dims = [range(*a.shape[:common_dim]), range(*a.shape[common_dim+1:]), range(*b.shape[:common_dim]), range(*b.shape[common_dim+1:])]

    for i,l,j,r in product(*dims):
        total = 0
        for k in range(a.shape[common_dim]):
            total += a[i,k,l] * b[j,k,r]
        result[i,j,l,r] = total
    print(result)

    # for i in range(a.shape[0]):
    #     for l in range(a.shape[2]):
    #         for j in range(b.shape[0]):
    #             for r in range(b.shape[2]):
    #                 total = 0
    #                 for k in range(a.shape[common_dim]):
    #                     total += a[i,k,l] * b[j,k,r]
    #                 result[i,j,l,r] = total
    # print(result)

    # result = torch.einsum("ik, jk -> ij", a, b)
    # common_dim = 1
    # result_shape = (2,1)
    # result = torch.zeros(result_shape)
    # for i in range(a.shape[0]):
    #     for j in range(b.shape[0]):
    #         total = 0
    #         for k in range(a.shape[common_dim]):
    #             total += a[i,k] * b[j,k]
    #         result[i,j] = total
    # print(result)