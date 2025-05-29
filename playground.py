from itertools import product
import torch


# Implementation of einsum without transposes and sums
if __name__ == "__main__":
    # Example input tensors
    tensor_a = torch.rand(1, 2)
    tensor_b = torch.rand(1, 3)

    # Index positions of common dimensions (shared axes)
    common_dims = [0]

    # For now, assume non-contracted dims of A appear before non-contracted dims of B
    output_shape = torch.Size((2, 3))

    # ----------------------- INPUT END ----------------------- #

    # Assume common dims are at the same position in both tensors
    common_dims_a, common_dims_b = common_dims, common_dims

    result = torch.zeros(output_shape)

    # Ranges for non-common dimensions (for iteration)
    a_noncommon_ranges = [range(size) for idx, size in enumerate(tensor_a.shape) if idx not in common_dims_a]
    b_noncommon_ranges = [range(size) for idx, size in enumerate(tensor_b.shape) if idx not in common_dims_b]

    # Cartesian products: all possible indices for non-common dims
    a_index_combinations = product(*a_noncommon_ranges)
    b_index_combinations = product(*b_noncommon_ranges)

    # Iterate over all combinations of non-contracted indices
    for a_indices, b_indices in product(a_index_combinations, b_index_combinations):
        common_index_ranges = product(*[range(tensor_a.shape[dim]) for dim in common_dims_a])
        total = 0
        for common_indices in common_index_ranges:
            # Build full indices including common dimensions
            full_a_indices = list(a_indices)
            full_b_indices = list(b_indices)
            for dim_idx, common_idx in enumerate(common_indices):
                full_a_indices.insert(common_dims_a[dim_idx], common_idx)
                full_b_indices.insert(common_dims_b[dim_idx], common_idx)

            total += tensor_a[*full_a_indices] * tensor_b[*full_b_indices]

        # Store the total in the result tensor
        result[a_indices + b_indices] = total

    print(result)
    print(torch.einsum("ij, ik -> jk", tensor_a, tensor_b))
