import torch


def toeplitz(c: torch.tensor, r: torch.tensor = None):
    """
    This function is similar to scipy.linalg.toeplitz, except for that not
    flatten is permormed on `c` and `r`. Instead these dimensions are handled
    as independent dimensions. Furthermore, this function cannot handle complex
    tensors at the moment.

    In the following is the docstring from scipy.linalg.toeplitz:

    Construct a Toeplitz matrix.
    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row.  If r is not given, ``r == conjugate(c)`` is
    assumed.
    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.
    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
    See also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0.  The behavior in previous
    versions was undocumented and is no longer supported.

    Examples
    --------
    >>> toeplitz([1,2,3], [1,4,5,6])
    tensor([[1, 4, 5, 6],
            [2, 1, 4, 5],
            [3, 2, 1, 4]])
    >>> c = [1,2,3]
    >>> toeplitz(c)
    tensor([[1, 2, 3],
            [2, 1, 2],
            [3, 2, 1]])
    >>> toeplitz([c, c])
    tensor([[[1, 2, 3],
             [2, 1, 2],
             [3, 2, 1]],
    <BLANKLINE>
            [[1, 2, 3],
             [2, 1, 2],
             [3, 2, 1]]])

    # Check reference implementation

    >>> from scipy.linalg import toeplitz as np_toeplitz
    >>> np_toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> np_toeplitz([1.0, 2+3j, 4-1j])
    array([[1.+0.j, 2.-3.j, 4.+1.j],
           [2.+3.j, 1.+0.j, 2.-3.j],
           [4.-1.j, 2.+3.j, 1.+0.j]])
    """
    c = torch.as_tensor(c)
    if r is None:
        r = c  # .conjugate()
    else:
        r = torch.as_tensor(r)

    assert len(c.shape) == len(r.shape), (c.shape, r.shape)
    assert r.shape[:-1] == c.shape[:-1], (r.shape, c.shape)

    vals = torch.cat((torch.flip(r[..., 1:], (-1,)), c), -1)
    stride = list(vals.stride())
    return torch.transpose(torch.flip(vals.as_strided(
        size=(*vals.shape[:-1], r.shape[-1], c.shape[-1]),
        stride=(*stride[:-1], stride[-1], stride[-1]),
    ), (-2,)), -2, -1)
