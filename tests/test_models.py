import pytest
from unittest.mock import Mock, AsyncMock
from genlm.eval.models.control import ControlModelAdaptor

from genlm.control import PromptedLLM
from genlm.control.constant import EndOfSequence


@pytest.fixture
def control_model():
    return ControlModelAdaptor(
        model_name="test_model",
        max_tokens=10,
        n_particles=100,
        ess_threshold=0.5,
        sampler_cache_size=2,
        critic_cache_size=2,
    )


def test_init(control_model):
    assert control_model.model_name == "test_model"
    assert control_model.max_tokens == 10
    assert control_model.n_particles == 100
    assert control_model.ess_threshold == 0.5
    assert control_model.sampler_cache_size == 2
    assert control_model.critic_cache_size == 2


def test_llm_lazy_initialization(control_model):
    # Mock the make_llm method
    control_model.make_llm = Mock(return_value=Mock(spec=PromptedLLM))

    # First access should create the LLM
    _ = control_model.llm
    control_model.make_llm.assert_called_once_with("test_model", {})

    # Second access should use cached property
    control_model.make_llm.reset_mock()
    _ = control_model.llm
    control_model.make_llm.assert_not_called()


def test_sampler_caching(control_model):
    # Mock make_sampler
    control_model.make_sampler = Mock()

    # Test cache hit
    instance1 = "test1"
    instance2 = "test2"
    instance3 = "test3"

    sampler1 = control_model.fetch_or_create_sampler(instance1)
    control_model.fetch_or_create_sampler(instance2)

    # Should return cached sampler
    assert control_model.fetch_or_create_sampler(instance1) == sampler1

    # Should evict instance1 when cache is full
    control_model.fetch_or_create_sampler(instance3)
    assert len(control_model._sampler_cache) == 2
    assert instance2 not in control_model._sampler_cache


def test_critic_caching(control_model):
    # Mock make_critic
    control_model.make_critic = Mock()

    # Test cache hit
    instance1 = "test1"
    instance2 = "test2"
    instance3 = "test3"

    critic1 = control_model.fetch_or_create_critic(instance1)
    control_model.fetch_or_create_critic(instance2)

    # Should return cached critic
    assert control_model.fetch_or_create_critic(instance1) == critic1

    # Should evict instance1 when cache is full
    control_model.fetch_or_create_critic(instance3)
    assert len(control_model._critic_cache) == 2
    assert instance2 not in control_model._critic_cache


@pytest.mark.asyncio
async def test_generate(control_model):
    # Mock dependencies
    control_model.make_prompt_ids = Mock(return_value=[1, 2, 3])
    control_model.make_sampler = Mock()
    control_model.make_critic = Mock()
    control_model.llm = Mock(spec=PromptedLLM)

    # Mock sampler.smc
    mock_sampler = Mock()
    mock_sampler.smc = AsyncMock()
    mock_sequence = [b"Hello", b" World"]
    mock_sampler.smc.return_value.posterior = {
        tuple(mock_sequence + [EndOfSequence()]): 0.8,
        tuple([b"Invalid"] + [EndOfSequence()]): float("nan"),
        tuple([b"Invalid2"]): 0.1,  # no eos
        tuple(
            ["😎".encode("utf-8")[:2]] + [EndOfSequence()]
        ): 0.1,  # for unicode decoding error
    }
    control_model.make_sampler.return_value = mock_sampler

    # Test generate
    result = await control_model.generate("test_instance")

    # Verify prompt_ids were set
    assert control_model.llm.prompt_ids == [1, 2, 3]

    # Verify sampler was called
    mock_sampler.smc.assert_called_once_with(
        n_particles=100,
        max_tokens=10,
        ess_threshold=0.5,
        critic=control_model.make_critic.return_value,
        json_path=None,
    )

    # Check response
    assert len(result.responses) == 1
    assert result.responses[0].text == "Hello World"
    assert result.responses[0].prob == 0.8
    assert isinstance(result.runtime_seconds, float)


def test_metadata(control_model):
    metadata = control_model.metadata()
    assert metadata == {
        "model_name": "test_model",
        "max_tokens": 10,
        "n_particles": 100,
        "ess_threshold": 0.5,
    }


def test_abstract_methods(control_model):
    with pytest.raises(NotImplementedError):
        control_model.make_sampler("test")

    with pytest.raises(NotImplementedError):
        control_model.make_critic("test")

    with pytest.raises(NotImplementedError):
        control_model.make_prompt_ids("test")
