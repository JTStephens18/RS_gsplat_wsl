from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import sdf_duplicate, sdf_remove, reset_opa, sdf_split, sdf_reset_opa_simple


@dataclass
class SDFStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 5000 #sdf2gs_from
    #refine_start_iter: int = 300
    refine_stop_iter: int = 30000 #sdf2gs_end
    #reset_every: int = 3000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    geo_interval: int = 100
    percent_dense: float = 0.01
    sdf_prune_threshold = 0.05
    sdf_densification_threshold = 0.9
    sdf_split_threshold = 0.85

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        sdfval: callable,
        packed: bool = False
        #neuRISRunner: Any = None,  # Placeholder for neuRISRunner if needed
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, step, packed=packed)

        # for name, optimizer in optimizers.items():
        #   optim_cls = type(optimizer)
        #   optimizers[name] = optim_cls(neuRISRunner.parameters(), lr=optimizer.defaults['lr'])

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            #and step % self.geo_interval == 3
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, sdfval)
            if self.verbose:
                print(
                    f"Step {step}: SDF: {n_dupli} GSs duplicated, SDF: {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step, sdfval)
            if self.verbose:
                print(
                    f"Step {step}: SDF: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )


            # max_grad = self.grow_grad2d
            # min_opacity = self.prune_opa
            # extent = state["scene_scale"]
            # max_screen_size = None

            # grads = state["grad2d"] / state["count"].clamp_min(1)

            # n_dupli, n_split, n_prune = self.sdf_densify_and_prune_with_ops(
            #     params=params,
            #     optimizers=optimizers,
            #     state=state,
            #     grads=grads,
            #     max_grad=max_grad,
            #     min_opacity=min_opacity,
            #     extent=extent,
            #     max_screen_size=max_screen_size,
            #     sdf_val=sdfval,  # Use the gaussian function defined above
            # )

            if self.verbose:
                print(
                    f"Step {step}: SDF: {n_dupli} GSs duplicated, SDF: {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )
                print(
                    f"Step {step}: SDF: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        # if step % self.reset_every == 0:
        #     sdf_reset_opa_simple(
        #         params=params,
        #         optimizers=optimizers,
        #         state=state,
        #         value=self.prune_opa * 2.0,
        #     )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        step: int,
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
        # if step >= self.refine_start_iter:
        #     print("[SDFStrategy] Grad shape ", state["grad2d"].shape)
        #     print("[SDFStrategy] gs_ids shape ", gs_ids.shape)
        #     print("[SDFStrategy] gs_ids max ", max(gs_ids))
        #     missing_ids = gs_ids[gs_ids >= state["grad2d"].shape[0]]
        #     print(f"[SDFStrategy] Missing ids shape ",  missing_ids.shape[0])
        #     print(f"[SDFStrategy] Missing ids ",  missing_ids)
        #     valid_ids = gs_ids[gs_ids < state["grad2d"].shape[0]]
        #     print(f"[SDFStrategy] Valid ids shape ",  valid_ids.shape[0])
        #     print(f"[SDFStrategy] Valid ids max ",  max(valid_ids))
        valid_mask = gs_ids < state["grad2d"].shape[0]
        gs_ids = gs_ids[valid_mask]
        #grads = grads[valid_mask]
        state["grad2d"].index_add_(0, gs_ids, grads[valid_mask].norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )



    def gaussian_fun(self, s, sigma):
        # return (-s**2)/(2*sigma**2)
        return torch.exp((-s**2)/(2*torch.square(sigma)))

    def cal_percentile(self, x):
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        percentile_values = []
        for p in percentiles:
            k = int(x.numel() * p / 100)  # 计算百分位对应的位置
            percentile_val, _ = torch.kthvalue(x.flatten(), k)
            percentile_values.append(percentile_val.item())
        return percentile_values

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        sdfval: callable
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        #is_dupli = self.gaussian_fun(sdfval(params["means"]), torch.sigmoid(params["opacities"]).squeeze()) > self.sdf_densification_threshold
        #n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        #is_split = self.gaussian_fun(sdfval(params["means"]), torch.sigmoid(params["opacities"]).squeeze()) > self.sdf_split_threshold
        #n_split = is_split.sum().item() 

        # first duplicate
        if n_dupli > 0:
            sdf_duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            sdf_split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        sdfval: callable,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        
        #prune_guidance = -(1 - self.gaussian_fun(sdfval(params["means"]), torch.sigmoid(params["opacities"]).squeeze()))
        prune_guidance = self.gaussian_fun(sdfval(params["means"]), torch.sigmoid(params["opacities"]).squeeze())
        #is_prune = prune_guidance < self.sdf_prune_threshold
        #n_prune = is_prune.sum().item()
        if n_prune == params["means"].shape[0]:
            print(
                f"[SDFStrategy] Warning: All {n_prune} GSs are attempted to be pruned. "
            )
            return 0
        elif n_prune > 0:
            sdf_remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    # def cal_percentile(self, x):
    #     percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    #     percentile_values = []
    #     for p in percentiles:
    #         k = int(x.numel() * p / 100)  # 计算百分位对应的位置
    #         percentile_val, _ = torch.kthvalue(x.flatten(), k)
    #         percentile_values.append(percentile_val.item())
    #     return percentile_values


    # def sdf_densify_and_split_with_ops(self, params, optimizers, state, sdf_densify_mask, grads, grad_threshold, scene_extent):
    #     n_init_points = params["means"].shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device=params["means"].device)
    #     if len(grads.shape) > 1:
    #         padded_grad[:grads.shape[0]] = torch.norm(grads, dim=-1)
    #     else:
    #         padded_grad[:grads.shape[0]] = grads
        
    #     selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                         torch.exp(params["scales"]).max(dim=1).values > self.percent_dense*scene_extent)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask, sdf_densify_mask)

    #     n_split = selected_pts_mask.sum().item()
        
    #     # Use the split function from ops
    #     if n_split > 0:
    #         split(params=params, optimizers=optimizers, state=state, mask=selected_pts_mask, revised_opacity=False)
        
    #     return n_split

    # def sdf_densify_and_clone_with_ops(self, params, optimizers, state, sdf_densify_mask, grads, grad_threshold, scene_extent):
    #     # Extract points that satisfy the gradient condition
    #     if len(grads.shape) > 1:
    #         grad_norm = torch.norm(grads, dim=-1)
    #     else:
    #         grad_norm = grads
        
    #     selected_pts_mask = torch.where(grad_norm >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                         torch.exp(params["scales"]).max(dim=1).values <= self.percent_dense*scene_extent)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask, sdf_densify_mask)
        
    #     n_dupli = selected_pts_mask.sum().item()
        
    #     # Use the duplicate function from ops
    #     if n_dupli > 0:
    #         duplicate(params=params, optimizers=optimizers, state=state, mask=selected_pts_mask)
        
    #     return n_dupli

    # def sdf_densify_and_prune_with_ops(self, params, optimizers, state, grads, max_grad, min_opacity, extent, max_screen_size, sdf_val):
    #     """
    #     Modified version that returns counts for verbose output and handles gradient information properly.
    #     """
    #     with torch.no_grad():
    #         #PruningThreshold = -0.002  # 裁90%
    #         #PruningThreshold = 0.1
    #         PruningThreshold = -0.01
    #         DesificationThreshold = 0.95  # 70% 以上靠近的进行致密

    #         n_dupli = 0
    #         n_split = 0
    #         n_prune = 0

    #         #-----------densify------------------
    #         sdf_densify_guidance = self.gaussian_fun(sdf_val(params["means"]), torch.sigmoid(params["opacities"]).squeeze())
    #         torch.cuda.empty_cache()
    #         sdf_densify_mask = (sdf_densify_guidance > DesificationThreshold).squeeze()
            
    #         # Clone/duplicate operation
    #         print("[SDFStrategy] densify and clone ")
    #         n_dupli = self.sdf_densify_and_clone_with_ops(params, optimizers, state, sdf_densify_mask, grads, max_grad, extent)

    #         # Recalculate guidance after cloning (params may have changed)
    #         sdf_densify_guidance = self.gaussian_fun(sdf_val(params["means"]), torch.sigmoid(params["opacities"]).squeeze())
    #         torch.cuda.empty_cache()
    #         sdf_densify_mask = (sdf_densify_guidance > DesificationThreshold).squeeze()
            
    #         # Split operation
    #         print("[SDFStrategy] densify and split ")
    #         n_split = self.sdf_densify_and_split_with_ops(params, optimizers, state, sdf_densify_mask, grads, max_grad, extent)

    #         #-----------prune------------------
    #         sdf_prune_guidance = -(1 - self.gaussian_fun(sdf_val(params["means"]), torch.sigmoid(params["opacities"]).squeeze()))
    #         torch.cuda.empty_cache()
    #         sdf_prune_mask = (sdf_prune_guidance < PruningThreshold).squeeze()
            
    #         # Count gaussians to be pruned
    #         n_prune += sdf_prune_mask.sum().item()
            
    #         # Use the remove function for SDF-based pruning
    #         if n_prune > 0:
    #             sdf_remove(params=params, optimizers=optimizers, state=state, mask=sdf_prune_mask)

    #         # Additional pruning based on opacity and size
    #         print("[SDFStrategy] additional pruning ")
    #         prune_mask = (torch.sigmoid(params["opacities"]) < min_opacity).squeeze()
    #         if max_screen_size and hasattr(self, 'max_radii2D'):
    #             big_points_vs = self.max_radii2D > max_screen_size
    #             big_points_ws = torch.exp(params["scales"]).max(dim=1).values > 0.1 * extent
    #             prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
    #         # Count additional pruning
    #         additional_prune = prune_mask.sum().item()
    #         n_prune += additional_prune
            
    #         # Use the remove function for additional pruning
    #         if additional_prune > 0:
    #             sdf_remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
            
    #         torch.cuda.empty_cache()

    #         print("[SDFStrategy] End function ")
            
    #         return n_dupli, n_split, n_prune