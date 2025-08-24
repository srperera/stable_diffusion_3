from architectures.diff_model import MMDiT2D
from architectures.pretrained_models.vae import VAE
from architectures.time_components import TimeSampler
from architectures.pretrained_models.encoders import CaptionEncoder


#############################################################################################
# Function: Build Architecture Function
#############################################################################################
def build_architecture(config):
    # build all the components and return them
    # TODO: Might have to refactor later
    model_components = {
        "vae": VAE(),
        "model": MMDiT2D(config),
        "time_sampler": TimeSampler(),
        "caption_encoder": CaptionEncoder(config),
    }
    return model_components


def build_architecture_v2(config):
    # build all the components and return them
    # TODO: Might have to refactor later
    model_components = {
        # "vae": VAE(),
        "model": MMDiT2D(config),
        "time_sampler": TimeSampler(),
        # "caption_encoder": CaptionEncoder(config),
    }
    return model_components
