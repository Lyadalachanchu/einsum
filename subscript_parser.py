import torch

import sum_reductions


def parse_einsum(einsum_str, input_tensors):
    if "->" in einsum_str:
        input, output = einsum_str.split("->")
        output_labels = list(output.strip())
    else:
        input = einsum_str
        output_labels = None

    input_labels = [list(op.strip()) for op in input.split(',')]
    all_dim_chars = [dim_char for dim_chars in input_labels for dim_char in dim_chars]


    if output_labels is None:
        # infer: any label that appears exactly once across all inputs
        output_labels = [dim_char for dim_char in all_dim_chars if all_dim_chars.count(dim_char) == 1]

    # contains the corresponding sizes for each dim char
    dim_char_sizes = {}
    for T, dim_chars in zip(input_tensors, input_labels):
        # make sure tensors match their input sizes
        assert len(T.shape) == len(dim_chars), "number of dimensions between dim chars and tensors don't match"
        for axis, dim_char in enumerate(dim_chars):
            if dim_char in dim_char_sizes and dim_char_sizes[dim_char] != T.shape[axis]:
                raise ValueError("same dim char has two different sizes in input tensor")
            dim_char_sizes[dim_char] = T.shape[axis]

    # find which dimensions are contracted (appear more than once)
    contracted_dim_chars = [dim_char for dim_char in set(all_dim_chars) if all_dim_chars.count(dim_char) > 1]

    # for each tensor, record which axes are common (for now, assume same common axis(es?) and only two tensors)
    common_axes = [input_labels[0].index(dim_char) for dim_char in contracted_dim_chars]

    # build intermediate output shape (before the reductions; don't deal with transposes now)
    non_common_a = [dim_char for dim_char in input_labels[0] if dim_char not in contracted_dim_chars]
    non_common_b = [dim_char for dim_char in input_labels[1] if dim_char not in contracted_dim_chars]
    intermediate_dim_chars = non_common_a + non_common_b
    intermediate_output_shape = tuple(dim_char_sizes[dim_char] for dim_char in intermediate_dim_chars)

    # reduce dims
    # if its not in output_labels but in intermediate_dim_chars, add that axis to reduce_axis
    reduce_axes = [intermediate_dim_chars.index(dim_char) for dim_char in intermediate_dim_chars if dim_char not in output_labels]
    common_axes.sort()
    reduce_axes.sort()
    return common_axes, intermediate_output_shape, reduce_axes


if __name__ == "__main__":
    # Example input tensors
    tensor_a = torch.rand(1, 2)
    tensor_b = torch.rand(1, 3)

    # example str involving contracted dimensions (j) and reduction dims (i)
    # TODO: still assumes common dims are in the same axis (/es) for both tensors. Implement auto-transposes so they don't have to be same axis.
    example_str = "ji, jk -> k"
    assert parse_einsum(example_str, [tensor_a, tensor_b]) == ([0], (2, 3), [0])
    assert torch.allclose(torch.einsum(example_str, tensor_a, tensor_b), sum_reductions.barebones_einsum(tensor_a, tensor_b, *parse_einsum(example_str, [tensor_a, tensor_b])), rtol=1e-04, atol=1e-05)

    # example str involving contracted dimensions (j) and inferred output (ik)
    example_str = "ji, jk"
    assert parse_einsum(example_str, [tensor_a, tensor_b]) == ([0], (2, 3), [])
    assert torch.allclose(torch.einsum(example_str, tensor_a, tensor_b), sum_reductions.barebones_einsum(tensor_a, tensor_b, *parse_einsum(example_str, [tensor_a, tensor_b])), rtol=1e-04, atol=1e-05)

    # example str involving multiple contracted dimensions (j,w), and multiple reductions (l,e)
    # intermediate output shape should be ilek
    tensor_a = torch.rand(3, 2, 4, 2)
    tensor_b = torch.rand(3, 3, 5, 2)
    example_str = "jilw, jekw -> ik"
    assert parse_einsum(example_str, [tensor_a, tensor_b]) == ([0, 3], (2, 4, 3, 5), [1, 2])
    assert torch.allclose(torch.einsum(example_str, tensor_a, tensor_b), sum_reductions.barebones_einsum(tensor_a, tensor_b,
                                        *parse_einsum(example_str, [tensor_a, tensor_b])), rtol=1e-04, atol=1e-05)
