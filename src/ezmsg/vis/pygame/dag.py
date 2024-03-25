from pathlib import Path
import tempfile
import typing

import pygame

from ..dag import get_graph, pgv2pd


SCROLL_STEP = 50


class VisDAG:
    def __init__(
        self,
        tl_offset: typing.Tuple[int, int] = (0, 0),
        screen_height: int = 1440,
        graph_ip: str = "127.0.0.1",
        graph_port: int = 25978,
    ):
        self._screen_height = screen_height
        G = get_graph((graph_ip, graph_port))
        G.layout(prog="dot")
        # Create SVG to get the correct coordinates
        svg_path = Path(tempfile.gettempdir()) / "ezmsg-vis.svg"
        G.draw(svg_path, format="svg:cairo")
        # Get the graph details as dataframe
        self._node_df = pgv2pd(G)
        # Unfortunately, pygame cannot render svg very well, so we render as png for display
        img_path = Path(tempfile.gettempdir()) / "ezmsg-vis.png"
        G.draw(img_path)
        self._image = pygame.image.load(img_path)
        self._image_rect = self._image.get_rect(topleft=tl_offset)
        self._min_y = -(self._image_rect.height - screen_height)
        # Scale the svg coordinates by png size / svg size
        _svg = pygame.image.load(svg_path)
        self._node_df["y"] *= self._image_rect.height / _svg.get_rect().height
        self._node_df["x"] *= self._image_rect.width / _svg.get_rect().width

        self._image_y = 0  # Initial position of the image
        self._b_update = True

    @property
    def size(self) -> typing.Tuple[int, int]:
        return self._image_rect.size

    def handle_event(self, event: pygame.event.Event) -> typing.Optional[str]:
        clicked_node_path = None
        if event.type in [pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN]:
            mouse_pos = pygame.mouse.get_pos()
            if self._image_rect.left <= mouse_pos[0] <= self._image_rect.right:
                if event.type == pygame.MOUSEWHEEL:
                    if event.y > 0:
                        # scroll graph up
                        self._image_y = min(0, self._image_y + SCROLL_STEP)
                    elif event.y < 0:
                        # scroll graph down
                        self._image_y = max(self._min_y, self._image_y - SCROLL_STEP)
                    self._b_update = True

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Mouse events
                    if event.button == 1:
                        # Clicked on a node
                        graph_pos = (
                            mouse_pos[0],
                            self._image_rect.height - (mouse_pos[1] - self._image_y),
                        )
                        min_row = (
                            (self._node_df.x - graph_pos[0]) ** 2
                            + (self._node_df.y - graph_pos[1]) ** 2
                        ).argmin()
                        clicked_node_path = f"{self._node_df.iloc[min_row]['upstream']}"
        return clicked_node_path

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        res = []
        if self._b_update:
            surface.blit(self._image, (0, self._image_y))
            pygame.display.update(self._image_rect)
            res.append(self._image_rect)
            self._b_update = False
        return res
