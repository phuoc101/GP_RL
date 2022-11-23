import os
import numpy as np
import gpytorch
import time
import torch
from models.BatchIndependentMultitaskGPModel import BatchIndependentMultitaskGPModel
from loguru import logger
from utils.data_loading import load_training_data, load_test_data, load_data, save_data

float_type = torch.float32
torch.set_default_dtype(torch.float32)


class GPModel:
    def __init__(self, **kwargs):
        super(GPModel, self).__init__()
        self.data_fields = kwargs["GPModel_datafields"]
        self.verbose = kwargs["verbose"]
        # verbose level: Basic/Critical
        if self.verbose > 3:
            logger.info("Configuring model with parameters:")
        for key, value in kwargs.items():
            if key in self.data_fields:
                setattr(self, key, value)
                # verbose level: Trace Full
                if self.verbose > 3:
                    logger.info(f"attribute {key}: {value}")
        self.configs = kwargs

        # set device
        if not self.Force_CPU:
            self.set_processor()
        else:
            logger.info("Forcing CPU as processor...")
            self.set_processor_cpu()

    def initialize_model(self, path_train_data, path_model=""):
        if not os.path.isfile(path_model) or self.force_train:
            # initialize models, train, save
            # load train data
            (
                self.X_train,
                self.y_train,
                self.mean_states,
                self.std_states,
                self.x_lb,
                self.x_ub,
            ) = load_training_data(
                data_path=path_train_data, output_torch=True, normalize=True
            )
            # convert to GPU if required
            self.X_train = self.X_train.to(self.device, self.dtype)
            self.y_train = self.y_train.to(self.device, self.dtype)
            logger.debug(f"X train shape: {self.X_train.shape}")
            logger.debug(f"y train shape: {self.y_train.shape}")
            if self.verbose > 0:
                logger.info(
                    f"Loaded training dataset {path_train_data} with {len(self.X_train)} datapoints"
                )

            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=2
            ).to(device=self.device, dtype=self.dtype)
            self.model = BatchIndependentMultitaskGPModel(
                X_train=self.X_train, y_train=self.y_train, likelihood=self.likelihood
            ).to(device=self.device, dtype=self.dtype)
            self.train()
            model_info = {
                "model": self.model,
                "likelihood": self.likelihood,
            }
            os.makedirs("./results/", exist_ok=True)
            save_data(path_model, model_info)

        else:  # load models
            model_info = load_data(path_model)
            logger.info(f"Loaded model from {path_model}")
            self.model = model_info["model"].to(device=self.device, dtype=self.dtype)
            self.likelihood = model_info["likelihood"].to(
                device=self.device, dtype=self.dtype
            )
            if self.verbose > 3:
                logger.info(f"Model: {self.model}")
                logger.info(f"Likelihood: {self.likelihood}")

    def train(self):
        if self.verbose > 0:
            logger.info("Training GP models on data...")
        # ---- Optimize GP ----- #
        # Time the training process
        start_model_training = time.perf_counter()
        training_iter = self.GP_training_iter

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(
            [
                # Includes all submodel and all likelihood parameters
                {"params": self.model.parameters()},
            ],
            lr=0.1,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.X_train)
            # Calculate loss and backprop gradients
            loss = -mll(output, self.y_train)
            loss.backward()
            if self.verbose > 0:
                logger.info(
                    f"Iter {i+1}/{training_iter} - Loss: {loss:.3f} noise: {self.model.likelihood.noise.item():.3f}"
                )
            optimizer.step()
        end_model_training = time.perf_counter()
        elapsed_model_training = end_model_training - start_model_training
        if self.verbose > 0:
            logger.info(
                f"GP Models trained in {elapsed_model_training:.3f}s, with {len(self.X_train)} data points"
            )
        # set models into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

    def eval(self, path_test_data):
        """
        quantify how good it is agaist test data
        """
        # Load test data
        self.X_test, self.y_test = load_test_data(
            path_test_data, output_torch=True, normalize=True
        )
        self.X_test = self.X_test.to(self.device, self.dtype)
        self.y_test = self.y_test.to(self.device, self.dtype)

        if self.verbose > 2:
            print("Starting batch querying GP with test data")
        t1_GPQueryingBatch = time.perf_counter()
        X_test = self.X_test
        y_test = self.y_test
        logger.debug(f"X test shape: {X_test.shape}")
        logger.debug(f"y test shape: {y_test.shape}")
        y_pred = self.predict(X_test)
        t2_GPQueryingBatch = time.perf_counter()
        elapsed_GPQueryingBatch = t2_GPQueryingBatch - t1_GPQueryingBatch
        if self.verbose > 1:
            logger.info(
                f"GP models queried in {elapsed_GPQueryingBatch:.3f} seconds, with {len(X_test)} data points"
            )
        # calculate MSE
        y_actual = y_test.cpu().numpy()

        def calc_MSE(x1, x2):
            return sum(np.sqrt(x1**2 + x2**2))

        y_pred_mean = y_pred.mean.detach().cpu().numpy()
        y_pred_conf = y_pred.confidence_region()
        y_pred_conf = np.array([c.detach().cpu().numpy() for c in y_pred_conf])

        logger.debug(f"actual y shape: {y_actual.shape}")
        logger.debug(f"Number of tasks: {y_pred.num_tasks}")
        logger.debug(f"pred y mean shape: {y_pred_mean.shape}")
        logger.debug(f"pred y conf shape: {y_pred_conf.shape}")
        MSE_1 = calc_MSE(y_pred_mean[:, 0], y_actual[:, 0])
        MSE_2 = calc_MSE(y_pred_mean[:, 1], y_actual[:, 1])
        logger.info(f"MSE_1= {MSE_1:.2f}, MSE_2= {MSE_2:.4f}")
        return y_pred_mean, y_pred_conf

    def predict(self, X):
        """
        predict the output from input X* using GP model
        """
        if self.verbose > 3:
            # //TODO: try passing tensors around to see if it improves speed
            logger.info("getting prediction(s) from GP Model:")
        with gpytorch.settings.fast_pred_var():  # torch.no_grad(),
            observed_pred = self.likelihood(self.model(X))
        return observed_pred

    def set_processor(self):
        self.is_cuda = torch.cuda.is_available()
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        if self.verbose > 0:
            logger.info(
                f"using GPU: {self.is_cuda} - using processor: *({self.device})"
            )

    def set_processor_cpu(self):
        self.is_cuda = False
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        if self.verbose > 0:
            logger.info(f"Forcing CPU... using processor: *({self.device})")
