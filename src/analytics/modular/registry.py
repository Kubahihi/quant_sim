from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable


ModelRunner = Callable[[Any, Dict[str, Any]], Any]
SignalRunner = Callable[[Dict[str, Any], Dict[str, Any]], Any]


@dataclass
class RegistryEntry:
    name: str
    family: str
    runner: Callable[..., Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}

    def register(self, name: str, family: str, runner: ModelRunner) -> None:
        self._entries[name] = RegistryEntry(name=name, family=family, runner=runner)

    def items(self) -> Iterable[RegistryEntry]:
        return self._entries.values()

    def run_all(self, series: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for entry in self.items():
            outputs[entry.name] = entry.runner(series, context)
        return outputs


class SignalRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}

    def register(self, name: str, family: str, runner: SignalRunner) -> None:
        self._entries[name] = RegistryEntry(name=name, family=family, runner=runner)

    def items(self) -> Iterable[RegistryEntry]:
        return self._entries.values()

    def run_all(self, models: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        for entry in self.items():
            outputs[entry.name] = entry.runner(models, context)
        return outputs
