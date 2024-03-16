from multiprocessing import Process
import typing

import ezmsg.core as ez

from .unit import ShMemCircBuffSettings, ShMemCircBuff


class AttachShmProcess(Process):
    settings: ShMemCircBuffSettings

    def __init__(
        self,
        settings: ShMemCircBuffSettings,
        address: typing.Optional[typing.Tuple[str, int]] = None,
    ) -> None:
        super().__init__()
        self._graph_address = address
        self.settings = settings

    def run(self) -> None:
        components = {"VISBUFF": ShMemCircBuff(self.settings)}
        ez.run(
            components=components,
            connections=(
                (
                    self.settings.topic,
                    components["VISBUFF"].INPUT_SIGNAL,
                ),
            ),
            graph_address=self._graph_address,
        )
