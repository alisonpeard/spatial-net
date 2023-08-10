"""Class for common-neighbours null model.

    To construct a doubly-constrained null model:
    >> cn = CommonNeighborsModel(constraint="doubly")
    # unconstrained matrix with fitted α, γ
    >> prediction = cn.fit_transform(locs)
    >> doubly = DoublyConstrained()
    >> doubly_cn = doubly.fit_transform(locs, prediction)  # constrained matrix
"""
import warnings
from typing import Optional, Callable, Tuple, Dict
from operator import itemgetter

import numpy as np
from scipy import optimize
from sklearn import linear_model

from spatial_nets.locations import LocationsDataClass
from spatial_nets.base import Model
from spatial_nets.metrics import CPC
from spatial_nets.models.constraints import UnconstrainedModel,\
    ProductionConstrained, AttractionConstrained, DoublyConstrained


# A couple of utility functions
first, second = itemgetter(0), itemgetter(1)
GetterType = Callable[[float], float]


def zero(*args):
    return 0


def kwargs_from_vec(
    x: np.ndarray,
    template: Dict[str, GetterType],
) -> Dict[str, float]:
    kwargs = {}
    for k, v in template.items():
        kwargs[k] = v(x)
    return kwargs


class CommonNeighboursModel(Model):
    def __init__(
        self,
        constraint: Optional[str] = None,
        coef: Tuple[float, float, float] = None,
        method: str = "nlls",
        use_log: bool = False,
        maxiters: int = 500,
        verbose: bool = False
    ):
        super().__init__(constraint=constraint)
        self.verbose = verbose
        self.routine = None

        if method is None:
            if coef is None:
                raise ValueError(
                    "when `method` is set to None `coef` should be provided"
                )
            if len(coef) == 2:
                if not isinstance(coef, dict):
                    template = {"γ": first, "α": second}
                    self.coef_ = kwargs_from_vec(coef, template)
                elif all(k in coef for k in ["γ", "α"]):
                    self.coef_ = coef
                else:
                    raise ValueError("invalid keys for coefficients")

            else:
                raise ValueError("invalid number of coefficients")

        elif method not in ("nlls", "cpc", "linreg"):
            raise ValueError("invalid method")

        else:
            if method == "nlls":
                routine = getattr(self, "_nonlinear_leastsquares")
            elif method == "cpc":
                routine = getattr(self, "_max_cpc")
            else:  # "linreg"
                routine = getattr(self, "_linreg")

            self.routine = routine
            self.use_log = use_log

            # Auxiliary model for constraints that will during the call to fit
            if constraint is None:
                aux_constraint = UnconstrainedModel()
            elif constraint == "production":
                aux_constraint = ProductionConstrained()
            elif constraint == "attraction":
                aux_constraint = AttractionConstrained()
            else:  # doubly
                aux_constraint = DoublyConstrained(
                    maxiters=maxiters, verbose=verbose)

            self.aux_constraint = aux_constraint

    def __str__(self):
        string = "Common neighbors null model"
        return string

    def transform(self, mat=None, coef=None):
        """
        coef : dict
        """
        if coef is None:
            matrix = self._cn_matrix(**self.coef_)
        else:
            matrix = self._cn_matrix(**coef)
        return matrix

    def _cn_matrix(self, γ: float, α: float = 1.0) -> np.ndarray:
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
        fmat = self.flow_data
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
        # ndarray because matrices only take integer powere in numpy
        smat = np.array(fmat @ deg_mat @ fmat)

        with np.errstate(divide="ignore", invalid="raise"):
            dmat_power = self.dmat ** (-γ)
            dmat_power[dmat_power == np.inf] = 0.0

        out = (smat ** α) * dmat_power
        return out

    def fit(self, data: LocationsDataClass):
        """
        Fit the model to the observations and compute the model parameters.

        This method sets the `coef_` attribute and overwrites any data already
        stored there.

        wARNING: this method overrides the parameters already stored under `_coef`

        Parameters
        ----------
        data : LocationsDataClass
            The custom object which we defined to store the data. Note that
            this object needs to have its `flow_data` attribute set.

        Returns
        -------
        self, with coef_ attributes assigned

        """
        super().fit(data)  # method from Model
        self.dmat = data.dmat

        if self.routine is not None:  # i.e. self.routine="nlls"
            self.aux_constraint.fit(data)  # just gets Oi, Dj vecs etc.
            try:
                self.coef_ = self.routine(
                    constraint=self.constraint,
                    use_log=self.use_log,
                    verbose=self.verbose
                )
            except AssertionError as e:
                warnings.warn(str(e))

        return self

    def _nonlinear_leastsquares(
        self,
        constraint: str,
        use_log=False,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Calibrate the model using nonlinear least squares."""

        def cost_fun(x, template, y, idx, use_log):
            """"Compute cost func. for x-vector of arguments.

            Parameters:
            -----------
            x : input vector of arguments to be optimised
            template : dictionary of argument names
            """
            kwargs = kwargs_from_vec(x, template)
            mat = self._cn_matrix(**kwargs)
            predict = self.aux_constraint.transform(mat)
            diff = np.log(
                y) - np.log(predict[idx]) if use_log else y - predict[idx]
            return diff

        if constraint == "doubly":
            x0 = [2]
            bounds = (-np.inf, np.inf)
            template_args = {"γ": first, "α": zero}
        else:
            x0 = [1, 1]
            bounds = ([-np.inf, 0], [np.inf, np.inf])
            template_args = {"γ": first, "α": second}

        # The observations
        idx = self.flow_data.nonzero()
        y = np.asarray(self.flow_data[idx]).flatten()
        res = optimize.least_squares(
            cost_fun, x0, bounds=bounds, args=(template_args, y, idx, use_log)
        )
        st = res.status
        assert st > 0, "optimization routine failed"

        st_dict = {
            -1: "improper input parameters status returned from MINPACK.",
            0: "the maximum number of function evaluations is exceeded.",
            1: "gtol termination condition is satisfied.",
            2: "ftol termination condition is satisfied.",
            3: "xtol termination condition is satisfied.",
            4: "Both ftol and xtol termination conditions are satisfied.",
        }

        if verbose:
            print("Status: ", st_dict[st])

        return kwargs_from_vec(res.x, template_args)

    def nlls_cost_fun(self, x, use_log=False):
        """x : (α, γ) pair to get loss for common neighbours model.

        x : vector
            vector of coefficients (α,γ) or just (γ,)
        constraint : string
            if constraint is doubly then optimisation is only for (γ,)

        """
        if self.aux_constraint == "doubly":
            template = {"γ": first, "α": zero}
        else:
            template = {"γ": first, "α": second}
        idx = self.flow_data.nonzero()
        y = np.asarray(self.flow_data[idx]).flatten()
        kwargs = kwargs_from_vec(x, template)
        mat = self._cn_matrix(**kwargs)
        predict = self.aux_constraint.transform(mat)

        # compute vector of residuals
        diff = np.log(
            y) - np.log(predict[idx]) if use_log else y - predict[idx]

        # loss = 0.5 sum squared residuals
        loss = 0.5 * sum(diff ** 2)
        return loss
