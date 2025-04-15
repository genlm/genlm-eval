import time
import numpy as np
from functools import cached_property
from collections import OrderedDict
from genlm.control import PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.eval.core import ModelAdaptor, ModelOutput, ModelResponse


class ControlModelAdaptor(ModelAdaptor):
    """Model adaptor for controlled LLM generation."""

    def __init__(
        self,
        model_name,
        max_tokens,
        n_particles,
        ess_threshold,
        lm_args=None,
        sampler_cache_size=0,
        critic_cache_size=0,
    ):
        """Initialize the controlled LLM adaptor.

        Args:
            model_name: Name of the model to use
            max_tokens: Maximum number of tokens to generate
            n_particles: Number of particles for sampling
            ess_threshold: ESS threshold for resampling
            lm_args: Arguments for the language model
            sampler_cache_size: Size of the sampler cache. 0 means no caching.
            critic_cache_size: Size of the critic cache. 0 means no caching.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.lm_args = lm_args or {}

        self.sampler_cache_size = sampler_cache_size
        self.critic_cache_size = critic_cache_size
        self._sampler_cache = OrderedDict()
        self._critic_cache = OrderedDict()

    @cached_property
    def llm(self):
        # Only create the LLM if needed
        return self.make_llm(self.model_name, self.lm_args)

    def make_llm(self, model_name, lm_args):
        return PromptedLLM.from_name(model_name, **lm_args)

    def get_sampler_cache_key(self, instance):
        """Get cache key for sampler. Override this method to customize the cache key."""
        return str(instance)

    def get_critic_cache_key(self, instance):
        """Get cache key for critic. Override this method to customize the cache key."""
        return str(instance)

    def fetch_or_create_sampler(self, instance):
        """Fetch or create the sampler for the model with caching support."""
        if self.sampler_cache_size == 0:
            return self.make_sampler(instance)

        cache_key = self.get_sampler_cache_key(instance)
        if cache_key in self._sampler_cache:
            self._sampler_cache.move_to_end(cache_key)
            return self._sampler_cache[cache_key]

        sampler = self.make_sampler(instance)
        if len(self._sampler_cache) >= self.sampler_cache_size:
            self._sampler_cache.popitem(last=False)
        self._sampler_cache[cache_key] = sampler
        return sampler

    def fetch_or_create_critic(self, instance):
        """Fetch or create the critic for the model with caching support."""
        if self.critic_cache_size == 0:
            return self.make_critic(instance)

        cache_key = self.get_critic_cache_key(instance)
        if cache_key in self._critic_cache:
            self._critic_cache.move_to_end(cache_key)
            return self._critic_cache[cache_key]

        critic = self.make_critic(instance)
        if len(self._critic_cache) >= self.critic_cache_size:
            self._critic_cache.popitem(last=False)
        self._critic_cache[cache_key] = critic
        return critic

    def make_sampler(self, instance):
        """Make the sampler for the model."""
        raise NotImplementedError("Subclass must implement this method")

    def make_critic(self, instance):
        """Make the critic for the model."""
        raise NotImplementedError("Subclass must implement this method")

    def make_prompt_ids(self, instance):
        raise NotImplementedError("Subclass must implement this method")

    def chat_template_formatter(self, system_prompt, few_shot_examples, query):
        messages = [{"role": "system", "content": system_prompt}]
        for input, output in few_shot_examples:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": query})
        return messages

    async def generate(self, instance, record_path):
        """Asynchronous generation method.

        Args:
            instance: The instance to generate examples for.

        Returns:
            ModelOutput: The generated responses and metadata.
        """
        self.llm.prompt_ids = self.make_prompt_ids(instance)
        sampler = self.fetch_or_create_sampler(instance)
        critic = self.make_critic(instance)

        start_time = time.time()
        sequences = await sampler.smc(
            n_particles=self.n_particles,
            max_tokens=self.max_tokens,
            ess_threshold=self.ess_threshold,
            critic=critic,
            json_path=record_path,
        )
        runtime = time.time() - start_time

        responses = []
        for sequence, prob in sequences.posterior.items():
            if np.isnan(prob):
                prob = float("-inf")

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

        return ModelOutput(
            responses=responses,
            runtime_seconds=runtime,
            metadata=self.metadata(),
        )

    def metadata(self):
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "n_particles": self.n_particles,
            "ess_threshold": self.ess_threshold,
        }
