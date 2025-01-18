import argparse
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.mixture import GaussianMixture

from gmm import gmm

_CLI_DOCSTRING = """
Fits a Guassian mixture model for different number of components and chooses the best one according to the AIC. 
"""


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=_CLI_DOCSTRING,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--filename",
        "-F",
        type=str,
        help="Name of the csv file containing the data.",
        required=True,
    )

    parser.add_argument(
        "--start",
        "-S",
        type=int,
        help="K starting value. Must be greater than 1.",
        default=2,
    )

    parser.add_argument(
        "--end",
        "-E",
        type=int,
        help="K ending value.",
        default=10,
    )

    return parser


@dataclass
class GMMSummary:
    fit: GaussianMixture
    aic: float
    llh: float

    


def calc_llh(
    data: np.ndarray, wgt: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> float:
    """Calculates log-likelihood."""

    densities = gmm.pdf(data, wgt=wgt, mu=mu, sigma=sigma)
    return np.sum(np.log(densities))


def calc_n_parameters(mu: np.ndarray) -> int:

    n_components, n_features = mu.shape
    cov_params = n_components * n_features * (n_features + 1) / 2.0
    mean_params = n_features * n_components
    return int(cov_params + mean_params + n_components - 1)


def calc_AIC(data: np.ndarray, wgt: np.ndarray, mu: np.ndarray, sigma) -> float:
    """Calculates aic."""
    k = calc_n_parameters(mu)
    llh = calc_llh(data=data, wgt=wgt, mu=mu, sigma=sigma)
    aic = 2 * k - 2 * llh
    return aic


def string_to_number(string: str) -> np.ndarray:
    """Converts string of number of numpy array."""
    return np.array(string.split(","), dtype=float).reshape(-1, 1)


def best_gmm(X: np.ndarray, start: int = 2, end: int = 10) -> GMMSummary:

    K_values = range(start, end + 1)

    best_aic: float = np.inf
    best_llh: float | None = None
    best_fit: GaussianMixture | None = None

    for K in K_values:
        fit = gmm.fit(X, K)

        # Compute likelihood.
        llh = calc_llh(
            X, wgt=fit.weights_, mu=fit.means_, sigma=fit.covariances_
        )

        # Compute AIC.
        # The lower the better.
        aic = calc_AIC(
            X, wgt=fit.weights_, mu=fit.means_, sigma=fit.covariances_
        )

        # Check `aic` criterion.
        if aic < best_aic:
            best_aic = aic
            best_fit = fit
            best_llh = llh

    return GMMSummary(best_fit, best_aic, best_llh)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Read file content and convert to Numpy array.
    with open(file=args.filename) as f:
        string = f.read()
        X = string_to_number(string)
    
    summary = best_gmm(X, start=args.start, end=args.end)
    print("The best model is: ")
    print(summary)


if __name__ == "__main__":
    main()
