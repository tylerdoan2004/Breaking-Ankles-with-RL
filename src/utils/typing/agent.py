"""
A module for typing agents.
"""
from typing import Any, Generic, Protocol, TypeVar


ProtocolObsType = TypeVar("ProtocolObsType", contravariant = True)
ProtocolActType = TypeVar("ProtocolActType", covariant = True)
class Agent(Protocol[ProtocolObsType, ProtocolActType]):
    """
    A protocol for representing an agent.
    """
    def predict(self, observation: ProtocolObsType, *, deterministic: bool = True) -> ProtocolActType:
        """
        Predicts an action given an observation.
        
        :param observation: The observation used to predict the next action.
        :param deterministic: Whether the agent predicts actions deterministically.
        :return: The predicted action.
        """
        ...


class StableBaselines3Model(Protocol[ProtocolObsType, ProtocolActType]):
    """
    A protocol for representing a Stable Baselines 3 model.
    """
    def predict(self, observation: ProtocolObsType, state: Any = None, episode_start: Any = None, deterministic: bool = False) -> tuple[ProtocolActType, Any]:
        """
        Predicts an action given an observation.
        
        :param observation: The observation used to predict the next action.
        :param deterministic: Whether the agent predicts actions deterministically.
        :return: The predicted (action, state) tuple.
        """
        ...


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
class StableBaselines3Agent(Generic[ObsType, ActType]):
    """
    A class for representing a Stable Baselines 3 agent.
    """
    def __init__(self, model: StableBaselines3Model[ObsType, ActType]) -> None:
        """
        Creates a StableBaselines3Agent object.
        
        :param model: The Stable Baselines 3 model used by the agent.
        :return: None.
        """
        self.model = model

    def predict(self, observation: ObsType, *, deterministic: bool = True) -> ActType:
        """
        Predicts an action given an observation.
        
        :param observation: The observation used to predict the next action.
        :param deterministic: Whether the agent predicts actions deterministically.
        :return: The predicted action.
        """
        action, _ = self.model.predict(observation, deterministic = deterministic)
        return action
