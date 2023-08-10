    def _cn_matrix(fmat, dmat, γ: float, α: float = 1.0) -> np.ndarray:
        """
        Calculate the common neighbours matrix.

        Parameters
        ----------
        γ : float
            Decay exponent.
        α : float, optional
            The power on the (directed) Adamic-Adar index

        Returns
        -------
        np.array
            nxn unconstrained gravity matrix.

        """
        in_degs = fmat.sum(axis=0)
        out_degs = fmat.sum(axis=1)

        degs = in_degs + out_degs
        try:
            deg_mat = 1 / np.log(degs)
            deg_mat[deg_mat == np.inf] = 0.0
        except RuntimeError:
            raise ValueError(
                f"Zero degrees encountered for node {np.where(degs==0)[0][0]},"
                " cannot perform inverse log operation")
        # ndarray because matrices only take integer power in numpy
        smat = np.array(fmat @ deg_mat @ fmat)

        with np.errstate(divide="ignore", invalid="raise"):
            dmat_power = dmat ** (-γ)
            dmat_power[dmat_power == np.inf] = 0.0

        out = (smat ** α) * dmat_power
        return out
