from .algo_callback import AlgoCallback


class ParetoHyperNetwork(AlgoCallback):
    def configure_model(self, model, validate_models, reinit_flag, p, rank):
        raise NotImplementedError
