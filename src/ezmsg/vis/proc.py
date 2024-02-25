from multiprocessing import Process

import ezmsg.core as ez

from .unit import ShMemCircBuffSettings, ShMemCircBuff


class AttachShmProcess(Process):
    settings: ShMemCircBuffSettings

    def __init__(self, settings: ShMemCircBuffSettings) -> None:
        super().__init__()
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
        )
