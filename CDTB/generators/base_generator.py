from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs):
        """Generate output based on the given prompt."""
        pass
        