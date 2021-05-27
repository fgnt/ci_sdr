# Convolutive Transfer Function Invariant SDR

![Run python tests](https://github.com/fgnt/ci_sdr/workflows/Run%20python%20tests/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/ci_sdr)](https://pypi.org/project/ci-sdr)
[![codecov.io](https://codecov.io/github/fgnt/ci_sdr/coverage.svg?branch=main)](https://codecov.io/github/fgnt/ci_sdr?branch=main)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ci_sdr)](https://pypi.org/project/ci-sdr)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgnt/ci_sdr/blob/master/LICENSE)

<!-- ![Run python dependency test](https://github.com/fgnt/ci_sdr/workflows/Run%20python%20dependency%20test/badge.svg) -->


This repository contains an implementation for the `Convolutive transfer function Invariant Signal-to-Distortion Ratio` objective for PyTorch as described in the publication `Convolutive Transfer Function Invariant SDR training criteria for Multi-Channel Reverberant Speech Separation` ([link arXiv][arXiv]).

Here, a small example, how you can use this CI-SDR objective in your own source code:

```python
import torch
import ci_sdr

reference: torch.tensor = ...
# reference.shape: [speakers, samples]

estimation: torch.tensor = ...
# estimation shape: [speakers, samples]

sdr = ci_sdr.pt.ci_sdr_loss(estimation, reference)
# sdr shape: [speakers]
```

The idea of this objective function is based in the theory from `E. Vincent, R. Gribonval and C. Févotte, Performance measurement in blind audio source separation, IEEE Trans. Audio, Speech and Language Processing`, known as
`BSSEval`.
The original author provided MATLAB source code ([link](http://bass-db.gforge.inria.fr/bss_eval/)) and the package `mir_eval` ([link](http://craffel.github.io/mir_eval/#module-mir_eval.separation)) contains a python port. Some peoble refer to these implementations as `BSSEval v3` ([link](https://github.com/sigsep/bsseval)).

The PyTorch code in this package is tested to yield the same `SDR` values as `mir_eval` with the default parameters.

> **NOTE:** If you want to use `BSSEval v3 SDR` as metric, I recomment to use `mir_eval.separation.bss_eval_sources` and use as reference the clean/unreverberated source signals. The implementation in this repository has minor difference that makes it problematic to compare SDR values accorss different publications (e.g. here the permutation is calculated on the SDR, while `mir_eval` computes it based on the `SIR`.).



# Installation

Install it directly with Pip, if you just want to use it:

```bash
pip install ci-sdr
```

or to get the recent version:

```bash
pip install git+https://github.com/fgnt/ci_sdr.git
```

If you want to install it with `all` dependencies (test and doctest dependencies), run:

```bash
pip install git+https://github.com/fgnt/ci_sdr.git#egg=ci_sdr[all]
```

When you want to change the code, clone this repository and install it as `editable`:

```bash
git clone https://github.com/fgnt/ci_sdr.git
cd ci_sdr
pip install --editable .
# pip install --editable .[all]
```

# Citation

To cite this implementation, you can cite the following paper ([link][arXiv]):
```
@article{boeddeker2020convolutive,
  title   = {Convolutive Transfer Function Invariant {SDR} training criteria for Multi-Channel Reverberant Speech Separation},
  author  = {Boeddeker, Christoph and Zhang, Wangyou and Nakatani, Tomohiro and Kinoshita, Keisuke and Ochiai, Tsubasa and Delcroix, Marc and Kamo, Naoyuki and Qian, Yanmin and Haeb-Umbach, Reinhold},
  journal = {arXiv preprint arXiv:2011.15003},
  year    = {2020}
}
```



[arXiv]: https://arxiv.org/abs/2011.15003