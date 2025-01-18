from typing import Any, Literal

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

__all__ = ("gmm",)


def _multivariate_normal(
    x: np.ndarray,
    wgt: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    method: Literal["pdf", "cdf"] = "pdf",
) -> np.ndarray:
    """Mixture of multivatiate normal distributions.

    Parameters
    ----------
    x : 2-D array of shape (n_samples, n_dimensions)
        The input samples.

    wgt : 1-D array of shape (n_components,)
        The weights of each mixture components.

    mu : 2-D array of shape (n_components, n_dimensions)
        The mean of each mixture component.

    sigma : 3-D array of shape (n_components, n_dimensions, n_dimensions)
        The covariance matrix of each mixture component.

    method : str, {"pdf", "cdf"}, default="pdf"
        Wether to mix density ("pdf) or cumulative ("cdf").
    """

    method = (
        multivariate_normal.pdf if method == "pdf" else multivariate_normal.cdf
    )

    # Get number of components from `wgt`.
    wgt = np.array(wgt)
    n_components = wgt.shape[0]

    # Get number of samples from `x`.
    n_samples = x.shape[0]

    # Pre-allocate memory for N_i(x | mu, sigma) where
    # `i` in 1, 2, ..., `n_componenets`.
    N = np.ndarray(shape=(n_samples, n_components))

    for i in range(n_components):

        # Assign N_i(x | mu, sigma) to ith column.
        N[:, i] = method(x=x, mean=mu[i, :], cov=sigma[i, :, :])

    # Return a linear combination of the columns in `N` using as scalars
    # the given weights `wgt` normalized.
    W = np.array(wgt / wgt.sum()).reshape(-1, 1)
    return N @ W


class GaussianMixtureModelGen:
    """A Gaussian Mixture model."""

    def pdf(
        self,
        x: np.ndarray,
        wgt: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """Mixture probability density function.

        Parameters
        ----------
        x : array of shape (n_samples, n_dimensions)
            The input samples.

        wgt : array of shape (n_components,)
            The weights of each mixture components.

        mu : array of shape (n_components, n_dimensions)
            The mean of each mixture component.

        sigma : array of shape (n_components, n_dimensions, n_dimensions)
            The covariance matrix of each mixture component.
        """
        return _multivariate_normal(
            x=x, wgt=wgt, mu=mu, sigma=sigma, method="pdf"
        )

    def cdf(
        self,
        x: np.ndarray,
        wgt: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """Mixture probability cumulative function.

        Parameters
        ----------
        x : array of shape (n_samples, n_dimensions)
            The input samples.

        wgt : array of shape (n_components,)
            The weights of each mixture components.

        mu : array of shape (n_components, n_dimensions)
            The mean of each mixture component.

        sigma : array of shape (n_components, n_dimensions, n_dimensions)
            The covariance matrix of each mixture component.
        """
        return _multivariate_normal(
            x=x, wgt=wgt, mu=mu, sigma=sigma, method="cdf"
        )

    def rvs(
        self,
        wgt: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        size: int | None = None,
    ) -> np.ndarray:
        """Sampling from multivariate Normal mixture model.

        Parameters
        ----------
        wgt : array of shape (n_components,)
            The weights of each mixture components.

        mu : array of shape (n_components, n_dimensions)
            The mean of each mixture component.

        sigma : array of shape (n_components, n_dimensions, n_dimensions)
            The covariance matrix of each mixture component.

        size : int or None, default=None
            Sapmle size. If None, size=1.
        """
        wgt = np.array(wgt)
        wgt = wgt / wgt.sum()

        # Randomly select components using `wgt` as probability distribution.
        # The size of the sample is given by `size`.
        n_components = wgt.shape[0]
        chosen_components = np.random.choice(n_components, size=size, p=wgt)

        # Pre-allocate memory for returned samples.
        n_dimensions = mu.shape[1]
        samples = np.ndarray(shape=(size, n_dimensions))

        # Generate samples for each index.
        for i, component in enumerate(chosen_components):
            samples[i, :] = multivariate_normal.rvs(
                mean=mu[component],
                cov=sigma[component]
            )

        return samples

    def fit(self, data: np.ndarray, K: int, **kwargs) -> GaussianMixture:
        """Fit Gaussian Mixture to data.

        Parameters
        ----------
        K : int
            Number of components.

        **kwargs : key-word arguments.
            Extra arguments passed to :class:`sklearn.mixture.GaussianMixture`.
        """
        return GaussianMixture(K, covariance_type="full", **kwargs).fit(data)


# Singleton instance.
# This instance is meant to be used by users.
# Example:
# >>> from gmm import gmm
# >>> gmm.pdf(...)
gmm = GaussianMixtureModelGen()


def test_pdf():
    wgt = np.array([0.5, 0.5])
    mu = np.array([[0, 0], [2, 2]])
    sigma = np.array([np.eye(2), np.eye(2)])
    x = np.array([[0, 0], [1, 1], [2, 2]])

    expected = np.sum(
        [
            multivariate_normal.pdf(x, mean=mu[i], cov=sigma[i]) * wgt[i]
            for i in range(2)
        ],
        axis=0,
    )

    actual = gmm.pdf(x=x, wgt=wgt, mu=mu, sigma=sigma).flatten()

    # Check pdf.
    # The weighted individual pdfs should be close to the mixture pdf.
    assert np.allclose(actual, expected), "PDF calculation failed."


def test_rvs():
    wgt = np.array([0.7, 0.3])
    mu = np.array([[0, 0], [3, 3]])
    sigma = np.array([np.eye(2), np.eye(2)])
    size = 1000

    samples = gmm.rvs(wgt=wgt, mu=mu, sigma=sigma, size=size)

    # Check sample shape
    assert samples.shape == (size, 2), "Sample shape mismatch."

    # Check approximate means
    # Sample mean should be close to the weighted mean.
    means = samples.mean(axis=0)
    assert np.allclose(means, np.dot(wgt, mu), atol=0.2), "Mean mismatch."


if __name__ == "__main__":
    test_rvs()
    test_pdf()
    print("All tests were passed successfully!")
