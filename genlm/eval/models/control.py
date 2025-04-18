import time
import numpy as np
from functools import cached_property
from collections import OrderedDict
from genlm.control import PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.eval.core import ModelAdaptor, ModelOutput, ModelResponse


class ControlModelAdaptor(ModelAdaptor):
    """Adaptor for genlm-control models."""

    def __init__(
        self,
        model_name,
        max_tokens,
        n_particles,
        ess_threshold,
        resampling_method="multinomial",
        lm_args=None,
        sampler_cache_size=0,
        critic_cache_size=0,
    ):
        """Initialize the controlled LLM adaptor.

        Args:
            model_name (str): Name of the model to use
            max_tokens (int): Maximum number of tokens to generate
            n_particles (int): Number of particles for sampling
            ess_threshold (float): ESS threshold for resampling
            resampling_method (str): Resampling method to use. See `llamppl.inference.smc_standard` for options.
            lm_args (dict): Arguments for the language model
            sampler_cache_size (int): Size of the sampler cache. 0 means no caching.
            critic_cache_size (int): Size of the critic cache. 0 means no caching.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.lm_args = lm_args or {}

        self.sampler_cache_size = sampler_cache_size
        self.critic_cache_size = critic_cache_size
        self._sampler_cache = OrderedDict()
        self._critic_cache = OrderedDict()

    @cached_property
    def llm(self):
        """Lazily initialize and return the language model.

        Returns:
            (PromptedLLM): The initialized language model instance.
        """
        return self.make_llm(self.model_name, self.lm_args)

    def fetch_or_create_critic(self, instance):
        """Fetch or create the critic for the model with LRU caching support."""
        if self.critic_cache_size == 0:
            return self.make_critic(instance)

        cache_key = self.get_critic_cache_key(instance)
        if cache_key in self._critic_cache:
            self._critic_cache.move_to_end(cache_key)
            return self._critic_cache[cache_key]

        if len(self._critic_cache) >= self.critic_cache_size:
            self._critic_cache.popitem(last=False)

        critic = self.make_critic(instance)
        self._critic_cache[cache_key] = critic
        return critic

    def fetch_or_create_sampler(self, instance):
        """Fetch or create the sampler for the model with LRU caching support."""
        if self.sampler_cache_size == 0:
            return self.make_sampler(instance)

        cache_key = self.get_sampler_cache_key(instance)
        if cache_key in self._sampler_cache:
            self._sampler_cache.move_to_end(cache_key)
            return self._sampler_cache[cache_key]

        if len(self._sampler_cache) >= self.sampler_cache_size:
            self._sampler_cache.popitem(last=False)

        sampler = self.make_sampler(instance)
        self._sampler_cache[cache_key] = sampler
        return sampler

    def make_llm(self, model_name, lm_args):
        """Create a new PromptedLLM instance.

        Override this method to customize the language model configuration, like
        setting the temperature or eos tokens.

        Args:
            model_name (str): Name of the model to initialize
            lm_args (dict): Additional arguments for model initialization

        Returns:
            (PromptedLLM): The initialized language model
        """
        return PromptedLLM.from_name(model_name, **lm_args)

    def get_sampler_cache_key(self, instance):
        """Generate a cache key for the sampler instances.

        Override this method to provide custom cache keys.

        Args:
            instance (DatasetInstance): The dataset instance to generate a cache key for

        Returns:
            (Any): A unique cache key for the instance
        """
        return instance.instance_id

    def get_critic_cache_key(self, instance):
        """Generate a cache key for the critic instances.

        Override this method to provide custom cache keys.

        Args:
            instance (DatasetInstance): The dataset instance to generate a cache key for

        Returns:
            (Any): A unique cache key for the instance
        """
        return instance.instance_id

    def make_sampler(self, instance):
        """Create a `genlm.control.TokenSampler` instance given a dataset instance.

        This abstract method should be implemented by subclasses to create
        a sampler appropriate for the provided instance.

        Args:
            instance: The dataset instance

        Returns:
            (genlm.control.TokenSampler): A sampler instance configured for the model
        """
        raise NotImplementedError("Subclasses must implement `make_sampler`")

    def make_critic(self, instance):
        """Create a `genlm.control.Potential` instance given a dataset instance.

        This abstract method should be implemented by subclasses to create
        a potential appropriate for the provided instance. If no critic is needed, return None.

        Args:
            instance: The dataset instance

        Returns:
            (genlm.control.Potential): A potential instance configured for the model, or None if no critic is needed
        """
        raise NotImplementedError("Subclasses must implement `make_critic`")

    def make_prompt_ids(self, instance):
        """Create prompt token IDs for the model.

        This abstract method should be implemented by subclasses to convert
        the input instance into token IDs that can be used to prompt the
        language model before generation.

        Args:
            instance (DatasetInstance): The dataset instance

        Returns:
            (List[int]): A list of token IDs representing the prompt
        """
        raise NotImplementedError("Subclasses must implement `make_prompt_ids`")

    def chat_template_formatter(self, system_prompt, few_shot_examples, query):
        messages = [{"role": "system", "content": system_prompt}]
        for input, output in few_shot_examples:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": query})
        return messages

    async def generate(self, instance, record_path=None):
        """Asynchronous generation method.

        Args:
            instance: The dataset instance.
            record_path: The path to save a record of the inference run.

        Returns:
            ModelOutput: The generated responses and metadata.
        """
        self.llm.prompt_ids = self.make_prompt_ids(instance)
        sampler = self.fetch_or_create_sampler(instance)
        critic = self.fetch_or_create_critic(instance)

        start_time = time.time()
        sequences = await sampler.smc(
            n_particles=self.n_particles,
            max_tokens=self.max_tokens,
            ess_threshold=self.ess_threshold,
            critic=critic,
            json_path=record_path,
            resampling_method=self.resampling_method,
        )
        runtime = time.time() - start_time
        responses = self.postprocess_sequences(sequences)

        return ModelOutput(
            responses=responses,
            runtime_seconds=runtime,
            metadata=self.metadata(),
        )

    def postprocess_sequences(self, sequences):
        """Convert a sequence of particles into a list of ModelResponse objects.

        Args:
            sequences (Sequences): The sequences object containing particles and their probabilities

        Returns:
            (List[ModelResponse]): A list of ModelResponse objects.
        """
        responses = []
        # TODO: maybe this should be decoded_posterior?
        for sequence, prob in sequences.posterior.items():
            if np.isnan(prob):
                continue

            if isinstance(sequence[-1], EndOfSequence):
                sequence = sequence[:-1]
            else:
                continue

            try:
                text = b"".join(sequence).decode("utf-8")
                responses.append(
                    ModelResponse(
                        text=text,
                        prob=prob,
                        metadata={
                            # Convert bytes into a json-serializable format
                            "context": repr(sequence),
                        },
                    )
                )
            except UnicodeDecodeError:
                continue

        return responses

    def metadata(self):
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "n_particles": self.n_particles,
            "ess_threshold": self.ess_threshold,
        }
