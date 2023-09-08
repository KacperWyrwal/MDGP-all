import sklearn
import torch 
import gpytorch 
import geometric_kernels.torch 
import os 
import math 
import warnings

from plotly.graph_objects import Figure
from lightning.fabric.utilities.cloud_io import get_filesystem
from gpytorch.metrics import mean_absolute_error, mean_squared_error, mean_standardized_log_loss, standardized_mean_squared_error, quantile_coverage_error, negative_log_predictive_density
from geometric_kernels.spaces import Hypersphere 

from lightning.pytorch import seed_everything
from tqdm.autonotebook import tqdm 
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.loggers import CSVLogger as FabricCSVLogger
from mdgp.utils import rotate
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure

from mdgp.utils import spherical_antiharmonic, spherical_harmonic, sphere_uniform_grid, sphere_meshgrid
from mdgp.models.deep_gps import GeometricManifoldDeepGP, EuclideanManifoldDeepGP, EuclideanDeepGP

from functools import partial 
from tqdm.autonotebook import tqdm 


pio.templates.default = "plotly_dark"
torch.set_default_dtype(torch.float64)
seed_everything(42, workers=True)


# hyperparams settings to run experiments with 
NUM_TRAIN_DENSE = 400
NUM_TRAIN_SPARSE = 80
NUM_INDUCING_DENSE = 60 
NUM_INDUCING_SPARSE = 12
NUM_EIGENFUNCTIONS_MANY = 20
NUM_EIGENFUNCTIONS_FEW = 10
NUM_HIDDEN_LIST = [0, 1, 2, 3]
NUM_VAL = 1000
NUM_TEST = 10000
NUM_PLOT = 40000


class CSVLogger(FabricCSVLogger):
    def _get_next_version(self) -> int:
        root_dir = os.path.join(self.root_dir, self.name)

        if not self._fs.isdir(root_dir):
            warnings.warn(f"Missing logger folder: {root_dir}")
            return 0

        existing_versions = []
        for d in self._fs.listdir(root_dir):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if self._fs.isdir(full_path) and name.startswith("version_"):
                existing_versions.append(int(name.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class PlotlyLogger:

    def __init__(self, root_dir: str, name: str, version=None): 
        self.root_dir = os.path.abspath(root_dir)
        self.name = name 
        self._version = version
        self._fs = get_filesystem(root_dir)
        if not os.path.exists(self.log_dir):
            warnings.warn(f"Logging directory {self.log_dir} does not exist. Creating new directory.")
            os.makedirs(self.log_dir)

    def save_html(self, fig: Figure, title: str, replace_existing=True) -> None: 
        # maybe add file extension 
        if not title.endswith('.html'): 
            title += '.html'
        file_path = os.path.join(self.log_dir, title)
        if os.path.exists(file_path) and not replace_existing:
            raise RuntimeError(f"File at {file_path} already exists and replace_existing is False.")
        fig.write_html(file_path)

    def save_json(self, fig: Figure, title: str, replace_existing=True) -> None: 
        # maybe add file extension 
        if not title.endswith('.json'): 
            title += '.json'
        file_path = os.path.join(self.log_dir, title)
        if os.path.exists(file_path) and not replace_existing:
            raise RuntimeError(f"File at {file_path} already exists and replace_existing is False.")
        fig.write_json(file_path)

    def save_png(self, fig: Figure, title: str, replace_existing=True) -> None: 
        # maybe add file extension 
        if not title.endswith('.png'): 
            title += '.png'
        file_path = os.path.join(self.log_dir, title)
        if os.path.exists(file_path) and not replace_existing:
            raise RuntimeError(f"File at {file_path} already exists and replace_existing is False.")
        fig.write_image(file_path, width=fig.layout.width, height=fig.layout.height, format='png')
    
    def save_pdf(self, fig: Figure, title: str, replace_existing=True) -> None: 
        # maybe add file extension 
        if not title.endswith('.pdf'): 
            title += '.pdf'
        file_path = os.path.join(self.log_dir, title)
        if os.path.exists(file_path) and not replace_existing:
            raise RuntimeError(f"File at {file_path} already exists and replace_existing is False.")
        fig.write_image(file_path, width=fig.layout.width, height=fig.layout.height, format='pdf')

    @property 
    def scale(self) -> float: 
        # Get the DPI of the Jupyter notebook display
        dpi = pio.kaleido.scope().get("dpi", 96)  # Default to 96 DPI if not available

        # Calculate the scale factor based on the DPI
        scale = int(dpi) / 96
        return scale 
        
    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, self.name, version)

    @property
    def version(self):
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self) -> int:
        root_dir = os.path.join(self.root_dir, self.name)

        if not self._fs.isdir(root_dir):
            warnings.warn(f"Missing logger folder: {root_dir}")
            return 0

        existing_versions = []
        for d in self._fs.listdir(root_dir):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if self._fs.isdir(full_path) and name.startswith("version_"):
                existing_versions.append(int(name.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


def plot_prediction(model, inputs, targets, inducing_points=True, train_points=None, sample='naive', template='plotly', subplot_width=400, subplot_height=600, no_background=True):
    with torch.no_grad():
        s = math.isqrt(inputs.size(0))
        x, y, z = inputs.view(s, s, 3).unbind(-1)
        with gpytorch.settings.fast_pred_var():
            mean = model(inputs, mean=True).view_as(x)
        stddev = model(inputs, sample_hidden=sample, sample_output=False).stddev.mean(0).view_as(x)
        with gpytorch.settings.num_likelihood_samples(1):
            sample = model(inputs, sample_hidden=sample, sample_output=sample)[0].view_as(x)
        targets = targets.view_as(x)

    fig = make_subplots(
        rows=1, cols=4, specs=[[{'type': 'surface'}] * 4], 
        subplot_titles=["Posterior Standard Deviation", "Posterior Mean", "Posterior Sample", "Ground Truth"], 
        horizontal_spacing=0.02)

    fig.add_trace(
        go.Surface(
            x=x, 
            y=y, 
            z=z, 
            surfacecolor=stddev,
            colorscale="viridis", 
            colorbar=dict(
                x=-0.1,
            ),
        ),
        row=1, col=1, 
    )

    fig.add_trace(
        go.Surface(
            x=x, 
            y=y, 
            z=z, 
            surfacecolor=mean,
            coloraxis="coloraxis1", 
        ),
        row=1, col=2, 
    )

    fig.add_trace(
        go.Surface(
            x=x, 
            y=y, 
            z=z, 
            surfacecolor=sample,
            coloraxis="coloraxis1", 
        ),
        row=1, col=3, 
    )

    fig.add_trace(
        go.Surface(
            x=x, 
            y=y, 
            z=z, 
            surfacecolor=targets,
            coloraxis="coloraxis1", 
        ),
        row=1, col=4, 
    )

    # Maybe plot inducing points 
    if inducing_points is True:
        inducing_points = model.output_layer.variational_strategy.inducing_points 
    if inducing_points is not None and inducing_points is not False:
        x, y, z = (inducing_points * 1.02).unbind(-1)
        for i in range(1, 4): 
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color='rgba(0, 0, 0, 0.25)',
                    # opacity=0.05, 
                    line=dict(
                        width=5,
                        color='black',
                    ),
                ),
                name='inducing points',
                legendgroup='inducing points',
                showlegend=i == 1
            ),col=i, row=1)

    # Maybe plot training points 
    if train_points is not None and train_points is not False: 
        x, y, z = (train_points * 1.02).unbind(-1)
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(0, 0, 0, 0.05)',
                opacity=0.6,
                line=dict(
                    width=4,
                    color='white',
                ),
            ),
            name='training points',
            legendgroup='training points',
        ),col=4, row=1)

    fig.update_layout(
        template=template,
        coloraxis1_colorscale='plasma',
        width=4 * subplot_width, 
        height=subplot_height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.125,
            xanchor="center",
            x=0.5, 
            bgcolor='lightgrey',
            tracegroupgap=200,
            # bordercolor="lightgrey",
            # borderwidth=1,
        ),
        margin=dict(l=0, r=0)#, t=20, b=30),
    )

    if no_background:
        scene_settings = dict(
            xaxis=dict(showbackground=False, gridcolor='lightgrey', showticklabels=False, title_text="X"),
            yaxis=dict(showbackground=False, gridcolor='lightgrey', showticklabels=False, title_text="Y"),
            zaxis=dict(showbackground=False, gridcolor='lightgrey', showticklabels=False, title_text="Z"),
            bgcolor='rgba(0,0,0,0)',  # this makes the background transparent
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        )
        fig.update_layout(
            scene=scene_settings,
            scene2=scene_settings,
            scene3=scene_settings,
            scene4=scene_settings,
        )

    return fig 


def plot_transformation(model, inputs, subplot_height=400, subplot_width=450, template='plotly', no_background=True):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        s = math.isqrt(inputs.size(0))
        hidden, outputs = model.forward_return_hidden(inputs, mean=True) 
        inputs = inputs.view(s, s, 3)
        outputs = outputs.view(s, s)
        coords = [inputs, *map(lambda x: x['manifold'].view(s, s, 3), hidden)]

    rows = len(coords)
    fig = make_subplots(
        rows=len(coords), cols=1, specs=[[{'type': 'surface'}]] * len(coords), 
        subplot_titles=[r"$\mathbb{{S}}^2\rightarrow \mathbb{{R}}$"]+[rf"$F_{{1:{i}}}(\mathbb{{S}}^2)\rightarrow \mathbb{{R}}$" for i in range(1, rows + 1)], 
        vertical_spacing = 0.05,
    )
    for layer, xyz in enumerate(coords, 1):
        x, y, z = xyz.unbind(-1)
        fig.add_trace(
            go.Surface(
                x=x, 
                y=y, 
                z=z, 
                surfacecolor=outputs,
            ),
            row=layer, col=1, 
        )

    fig.update_layout(
        height=subplot_height * len(coords), 
        width=subplot_width,
        template=template,
        margin=dict(l=40, r=0)
    )

    if no_background:
        scene_settings = dict(
            xaxis=dict(showbackground=False, gridcolor='lightgrey', showticklabels=False, title_text="X"),
            yaxis=dict(showbackground=False, gridcolor='lightgrey', showticklabels=False, title_text="Y"),
            zaxis=dict(showbackground=False, gridcolor='lightgrey', showticklabels=False, title_text="Z"),
            bgcolor='rgba(0,0,0,0)',  # this makes the background transparent
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        )
        fig.update_layout(
            {f"scene{i if i > 1 else ''}": scene_settings for i in range(1, rows + 1)}
        )

    return fig 


def kwargs_to_string(sep='-', no_keys=True, **kwargs):
    if no_keys: 
        return sep.join(f"{val}" for val in kwargs.values())
    return sep.join(f"{key}={val}" for key, val in kwargs.items())


def log(loggers, metrics, step=None):
    for logger in loggers: 
        logger.log_metrics(metrics=metrics, step=step)


def finalize(loggers): 
    for logger in loggers: 
        logger.finalize("Done")


def train_step(model, inputs, targets, criterion, sample_hidden='naive', loggers=None, step=None): 
    model.train() 
    outputs = model(inputs, sample_hidden=sample_hidden)
    loss = criterion(outputs, targets)
    if loggers is not None: 
        log(loggers=loggers, metrics={'elbo': loss}, step=step)
    return loss 


def test_step(model, inputs, targets, sample_hidden='naive', train_targets=None, loggers=None, step=None):
    with torch.no_grad():
        model.eval() 
        outputs_f = model(inputs, sample_hidden=sample_hidden)
        outputs_y = model.likelihood(outputs_f)
        metrics = {
            'expected_log_probability': model.likelihood.expected_log_prob(targets, outputs_f).mean(), 
            'mean_absolute_error': mean_absolute_error(outputs_y, targets).mean(0), 
            'mean_squared_error': mean_squared_error(outputs_y, targets).mean(0), 
            'standardized_mean_squared_error': standardized_mean_squared_error(outputs_y, targets).mean(0), 
            'mean_standardized_log_loss': mean_standardized_log_loss(outputs_y, targets, train_y=train_targets).mean(0), 
            'quantile_coverage_error': quantile_coverage_error(outputs_y, targets).mean(0), 
            'negative_log_predictive_density': negative_log_predictive_density(outputs_y, targets).mean(0)
        }
    if loggers is not None: 
        log(loggers=loggers, metrics=metrics, step=step)
    return metrics 


def fit(model, train_inputs, train_targets, criterion, optimizer, val_inputs=None, val_targets=None, val_every_n_epochs=50, sample_hidden='naive', num_epochs=1000, train_loggers=None, val_loggers=None): 
    pbar = tqdm(range(1, num_epochs + 1), desc="Fitting")
    metrics = {'elbo': None, 'nlpd': None, 'smse': None}
    for epoch in pbar:
        # training
        optimizer.zero_grad(set_to_none=True)
        loss = train_step(model=model, inputs=train_inputs, targets=train_targets, criterion=criterion, sample_hidden=sample_hidden, loggers=train_loggers, step=epoch)
        loss.backward()
        optimizer.step() 
        metrics.update({'elbo': loss.item()})
        # maybe validation
        if val_inputs is not None and val_targets is not None and (epoch % val_every_n_epochs == 0 or epoch == 1):
            val_metrics = test_step(model=model, inputs=val_inputs, targets=val_targets, sample_hidden=sample_hidden, train_targets=train_targets, loggers=val_loggers, step=epoch)
            metrics.update({'nlpd': val_metrics['negative_log_predictive_density'].item(), 'smse': val_metrics['standardized_mean_squared_error'].item()})
        # show metrics
        pbar.set_postfix(metrics)
    return model 


def get_target_function(name='smooth', degree=1, order=2):
    if name == 'smooth': 
        return partial(spherical_harmonic, m=degree, n=order)
    if name == 'singularity': 
        return partial(spherical_antiharmonic, m=degree, n=order)
    if name == 'singularity_rotated': 
        def target_fnc(x): 
            return spherical_antiharmonic(rotate(x=x, roll=math.pi / 2), m=degree, n=order)
        return target_fnc
    if name == 'singularity_hard': 
        def target_fnc(x): 
            z = spherical_antiharmonic(x, m=degree, n=order)
            # y = spherical_antiharmonic(rotate(x=x, roll=math.pi / 2), m=degree, n=order)
            x = spherical_antiharmonic(rotate(x=x, roll=math.pi / 2, pitch=math.pi / 2), m=degree, n=order)
            return  (z + x) / 1
        return target_fnc
    if name == 'singularity_vhard': 
        def target_fnc(x): 
            z = spherical_antiharmonic(x, m=degree, n=order)
            y = spherical_antiharmonic(rotate(x=x, roll=math.pi / 2), m=degree, n=order)
            x = spherical_antiharmonic(rotate(x=x, roll=math.pi / 2, pitch=math.pi / 2), m=degree, n=order)
            return  (z + y + x) / 2
        return target_fnc
    raise NotImplementedError


def get_data(n, target_fnc, noise_std=0.01, arrangement='uniform', meshgrid_eps=10e-6):
    if arrangement == 'uniform': 
        inputs = sphere_uniform_grid(n=n)
    elif arrangement == 'meshgrid': 
        s = math.isqrt(n)
        if s * s != n: 
            warnings.warn(f"{n} is not a perfect square. Generating meshgrid with {s*s} points")
        inputs = sphere_meshgrid(s, s, meshgrid_eps).view(-1, 3)
    outputs = target_fnc(inputs)
    return inputs, outputs + torch.randn_like(outputs) * noise_std


def get_outputscale_prior(outputscale_mean: float = 1.0):
    return gpytorch.priors.GammaPrior(1.0, 1 / outputscale_mean) 


def create_model(name, num_hidden, num_inducing, learn_inducing_locations=False, optimize_nu=False, num_eigenfunctions=20, outputscale_prior=None, project_to_tangent='intrinsic', nu=2.5, tangent_to_manifold='exp'):
    inducing_points = sphere_uniform_grid(num_inducing)
    if name == 'geometric_manifold':
        return GeometricManifoldDeepGP(
            space=Hypersphere(dim=2), 
            num_hidden=num_hidden, 
            num_eigenfunctions=num_eigenfunctions, 
            learn_inducing_locations=learn_inducing_locations, 
            optimize_nu=optimize_nu, 
            inducing_points=inducing_points, 
            nu=nu,
            project_to_tangent=project_to_tangent, 
            outputscale_prior=outputscale_prior, 
            tangent_to_manifold=tangent_to_manifold,
        )
    if name == 'euclidean_manifold': 
        return EuclideanManifoldDeepGP(
            space=Hypersphere(dim=2),
            num_hidden=num_hidden, 
            learn_inducing_locations=learn_inducing_locations, 
            inducing_points=inducing_points,
            nu=nu,
            project_to_tangent=project_to_tangent, 
            outputscale_prior=outputscale_prior,
            tangent_to_manifold=tangent_to_manifold,
        )
    if name == 'euclidean': 
        return EuclideanDeepGP(
            num_hidden=num_hidden, 
            inducing_points=inducing_points, 
            nu=nu, 
            learn_inducing_locations=learn_inducing_locations, 
            outputscale_prior=outputscale_prior,
        )
    raise NotImplementedError


def main(name, num_train, model_name, num_hidden, num_inducing, num_eigenfunctions, target_name, num_val=500, num_test=2000, num_plot=40000, learn_inducing_locations=False, optimize_nu=False, nu=2.5, tangent_to_manifold='exp', degree=1, order=2, sample_hidden='naive', num_epochs=1000, val_every_n_epochs=50, outputscale_mean=1.0, noise_std=0.01, meshgrid_eps=10e-6, project_to_tangent='intrinsic'):
    # 1. Create target function
    target_fnc = get_target_function(name=target_name, degree=degree, order=order)

    # 2. Create data 
    train_inputs, train_targets = get_data(n=num_train, target_fnc=target_fnc, noise_std=noise_std, arrangement='uniform') 
    val_inputs, val_targets = get_data(n=num_val, target_fnc=target_fnc, noise_std=noise_std, arrangement='uniform') 
    test_inputs, test_targets = get_data(n=num_test, target_fnc=target_fnc, noise_std=noise_std, arrangement='uniform') 
    plot_inputs, plot_targets = get_data(n=NUM_PLOT, target_fnc=target_fnc, noise_std=noise_std, arrangement='meshgrid', meshgrid_eps=meshgrid_eps)

    # 3. Create model, criterion, and optimizer 
    outputscale_prior = get_outputscale_prior(outputscale_mean=outputscale_mean)
    model = create_model(name=model_name, tangent_to_manifold=tangent_to_manifold, num_hidden=num_hidden, num_inducing=num_inducing, num_eigenfunctions=num_eigenfunctions, learn_inducing_locations=learn_inducing_locations, optimize_nu=optimize_nu, outputscale_prior=outputscale_prior, nu=nu, project_to_tangent=project_to_tangent)
    elbo = gpytorch.mlls.DeepApproximateMLL(gpytorch.mlls.VariationalELBO(likelihood=model.likelihood, model=model, num_data=num_train))
    optimizer = torch.optim.Adam(model.parameters(), maximize=True, lr=0.01)
    
    # 4. Train and validate model
    print("Training...")
    fit_tensorboard_logger = TensorBoardLogger(root_dir='logs/tensor_board/', name=name) 
    train_csv_logger = CSVLogger(root_dir='logs/csv/train/', name=name) 
    val_csv_logger = CSVLogger(root_dir='logs/csv/val/', name=name) 
    train_loggers = [fit_tensorboard_logger, train_csv_logger]
    val_loggers = [fit_tensorboard_logger, val_csv_logger]
    model = fit(model=model, train_inputs=train_inputs, train_targets=train_targets, val_inputs=val_inputs, val_targets=val_targets, sample_hidden=sample_hidden, train_loggers=train_loggers, val_loggers=val_loggers, num_epochs=num_epochs, val_every_n_epochs=val_every_n_epochs, criterion=elbo, optimizer=optimizer)
    # make sure logger files are saved
    finalize(loggers=[*val_loggers, *train_loggers])

    # 5. Test model 
    print("Testing...")
    test_csv_logger = CSVLogger(root_dir='logs/csv/test/', name=name, prefix='test')
    test_loggers = [test_csv_logger]
    test_metrics = test_step(model=model, inputs=test_inputs, targets=test_targets, sample_hidden=sample_hidden, loggers=test_loggers, train_targets=train_targets)
    # make sure logger files are saved
    finalize(loggers=test_loggers)
    print(test_metrics)

    # 6. plot 
    print("Plotting...")
    plotly_logger = PlotlyLogger('./plots/', name=name)
    fig = plot_prediction(model=model, inputs=plot_inputs, targets=plot_targets, inducing_points=True, train_points=train_inputs, sample=sample_hidden, no_background=True)
    plotly_logger.save_html(fig=fig, title='prediction.html')
    plotly_logger.save_json(fig=fig, title='prediction.html')
    fig = plot_transformation(model=model, inputs=plot_inputs, no_background=True)
    plotly_logger.save_html(fig=fig, title='transformation.json')
    plotly_logger.save_json(fig=fig, title='transformation.json')

    print("Done!")
    return model 


NUM_TRAIN_VERY_DENSE = 800
target_name = "singularity_vhard"
model_name = "geometric_manifold"
sample_hidden = 'naive'
project_to_tangent='intrinsic'

for run in range(10):
    for num_hidden in range(1,2): 
        for outputscale_mean in [0.01]:
            for nu in [2.5]:
                for sample_hidden in ["naive", "pathwise"]:
                    name = f'{model_name}-{target_name}-{sample_hidden}-{project_to_tangent}-mean={outputscale_mean}-nu={nu}-num_hidden={num_hidden}'
                    print("="*60, f"Experiment: {num_hidden=}, {outputscale_mean=}, {run=}, {nu=}, {sample_hidden=}", "="*60)
                    main(
                        name=name,
                        num_train=NUM_TRAIN_DENSE, 
                        num_val=500, 
                        model_name=model_name,
                        num_inducing=NUM_INDUCING_DENSE, 
                        num_eigenfunctions=NUM_EIGENFUNCTIONS_MANY, 
                        target_name=target_name,
                        tangent_to_manifold="exp",
                        num_hidden=num_hidden, 
                        sample_hidden=sample_hidden,
                        outputscale_mean=outputscale_mean, 
                        project_to_tangent=project_to_tangent, 
                        num_epochs=1000,
                        nu=nu,
                    )
