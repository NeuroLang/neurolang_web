from collections import defaultdict
from functools import partial
from ipywidgets import (
    BoundedFloatText,
    Checkbox,
    Dropdown,
    FloatSlider,
    FloatText,
    HBox,
    Layout,
    VBox,
)
from neurolang_ipywidgets import NlPapayaViewer

from plotly.graph_objects import Figure, FigureWidget, Histogram

from traitlets import Any

from ..util import debounce


class PapayaWidget(HBox):
    """A widget class that displays a papaya viewer (NlPapayaViewer) and config widget (PapayaConfigWidget) side by side."""

    current_config = Any()
    current_colorbar = Any()

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        **kwargs:
            config_visible: str
                Depending on the value, config widget will be visible or hidden upon initialization. Possible values are "hidden" or "visible"
        """
        super().__init__(*args, **kwargs)

        self._viewer = NlPapayaViewer(
            layout=Layout(width="70%", height="auto", border="1px solid black")
        )

        self._config = PapayaConfigWidget(
            self._viewer,
            layout=Layout(width="30%", height="auto", border="1px solid black"),
        )

        self._config.layout.visibility = kwargs.get("config_visible", "hidden")

        self.current_config = None
        self.current_colorbar = None

        self.children = [self._viewer, self._config]

    def show_image_config(self, image, show=True):
        if show:
            self._config.layout.visibility = "visible"
            self.current_config = image
            self._config.set_image(image)
        else:
            if self.current_config is not None and image.id == self.current_config.id:
                self._config.layout.visibility = "hidden"
                self._config.set_image(None)

    def show_image_colorbar(self, image):
        self.current_colorbar = image
        self._viewer.show_image_colorbar(image)

    def can_add(self, images):
        return self._viewer.can_add(images)

    def add(self, images):
        self._viewer.add(images)
        self.current_colorbar = self._viewer.get_colorbar_image()

    def remove(self, images):
        self._viewer.remove(images)
        self.current_colorbar = self._viewer.get_colorbar_image()

    def set_images(self):
        self._viewer.set_images()

    def set_center(self, widget, image):
        self._viewer.set_center(widget, image)

    def reset_center(self):
        self._viewer.reset_center()

    def set_error(self, error):
        self._viewer.set_error(error)

    def reset(self):
        self._viewer.reset()
        self._config.reset()

    def get_hex_for_lut(self, lut):
        return self._viewer.get_hex_for_lut(lut)


class PapayaConfigWidget(VBox):
    """A widget that displays widgets to adjust NLPapayaViewer image parameters."""

    lut_options = [
        "Grayscale",
        "Red Overlay",
        "Green Overlay",
        "Blue Overlay",
        "Gold",
        "Spectrum",
        "Overlay (Positives)",
        "Overlay (Negatives)",
    ]

    def __init__(self, viewer, *args, **kwargs):
        """
        Parameters
        ----------
        viewer: NlPapayaViewer
            associated viewer.
        """
        super().__init__(*args, **kwargs)

        self._viewer = viewer
        self._init_widgets()

        self.children = [
            VBox(
                [
                    VBox(
                        [self._hist],
                        layout=Layout(
                            height="auto",
                            margin="0px 0px 0px 0px",
                            padding="5px 5px 5px 5px",
                        ),
                    ),
                    VBox(
                        [
                            self._alpha,
                            self._lut,
                            self._nlut,
                            self._min,
                            self._minp,
                            self._max,
                            self._maxp,
                            self._sym,
                        ],
                        layout=Layout(width="230px"),
                    ),
                ]
            )
        ]

    def _init_widgets(self):
        """Initializes all configuration widgets. Possible image config parameters are:"""
        layout = Layout(width="200px", max_width="200px")

        self._alpha = FloatSlider(
            value=1,
            min=0,
            max=1.0,
            step=0.1,
            description="alpha:",
            description_tooltip="Overlay image alpha level (0 to 1).",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            layout=layout,
        )

        self._lut = Dropdown(
            options=PapayaConfigWidget.lut_options,
            value="Red Overlay",
            description="lut:",
            description_tooltip="The color table name.",
            layout=layout,
        )

        self._nlut = Dropdown(
            options=PapayaConfigWidget.lut_options,
            value="Red Overlay",
            description="negative-lut:",
            description_tooltip="The color table name used by the negative side of the parametric pair.",
            layout=layout,
        )

        self._min = FloatText(
            value=None,
            description="min:",
            description_tooltip="The display range minimum.",
            step=0.01,
            continuous_update=True,
            disabled=False,
            layout=layout,
        )

        self._minp = BoundedFloatText(
            value=None,
            min=0,
            max=100,
            step=1,
            continuous_update=True,
            description="min %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=layout,
        )

        self._max = FloatText(
            value=None,
            description="max:",
            description_tooltip="The display range maximum.",
            step=0.01,
            continuous_update=True,
            disabled=False,
            layout=layout,
        )

        self._maxp = BoundedFloatText(
            value=None,
            min=0,
            max=100,
            step=1,
            continuous_update=True,
            description="max %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=layout,
        )

        self._sym = Checkbox(
            value=False,
            description="symmetric",
            description_tooltip="When selected, sets the negative range of a parametric pair to the same size as the positive range.",
            disabled=False,
            layout=layout,
        )

        # figure to display histogram of image data
        fig = Figure()
        fig.update_layout(
            height=300,
            margin=dict(l=15, t=15, b=15, r=15, pad=4),
            showlegend=True,
            legend_orientation="h",
        )

        self._hist = FigureWidget(fig)
        self._hist.add_trace(
            Histogram(x=[], name="All image data", visible="legendonly")
        )
        self._hist.add_trace(Histogram(x=[], name="Image data without 0s"))

        self._handlers = defaultdict()

    def _set_values(self, config, range, data):
        """Sets config values from the specified `config` and creates histogram for `data`.

        Parameters
        ----------
        config : dict
            configuration parameters for the image. Possible keywords are:
            alpha : int
                the overlay image alpha level (0 to 1).
            lut : str
                the color table name.
            negative_lut : str
                the color table name used by the negative side of the parametric pair.
            max : int
                the display range maximum.
            maxPercent : int
                the display range maximum as a percentage of image max.
            min : int
                the display range minimum.
            minPercent : int
                the display range minimum as a percentage of image min.
           symmetric : bool
                if true, sets the negative range of a parametric pair to the same size as the positive range.
        range: float
            range of image values.
        data: []
           flattened image data.
        """
        self._alpha.value = config.get("alpha", 1)
        self._lut.value = config.get("lut", PapayaConfigWidget.lut_options[1])
        self._nlut.value = config.get("negative_lut", PapayaConfigWidget.lut_options[1])
        self._min.value = config.get("min", 0)
        self._minp.value = self._get_per_from_value(range, config.get("min", 0))
        self._max.value = config.get("max", 0.1)
        self._maxp.value = self._get_per_from_value(range, config.get("max", 0.1))
        self._sym.value = config.get("symmetric", "false") == "true"

        # set histogram data
        self._hist.data[0].x = data
        # leave out 0 values
        self._hist.data[1].x = [] if (data == [] or data is None) else data[data != 0]

    def _add_handlers(self, image):
        """Add config widget event handlers to change the config values for the specified `image`.

        Parameters
        ----------
        image: neurolang_ipywidgets.PapayaImage
            image whose config values will be viewed/modified using this config widget.
        """

        # Dropdown does not support resetting event handlers after Dropdown.unobserve_all is called
        # So handlers are stored to be removed individually
        # github issue https://github.com/jupyter-widgets/ipywidgets/issues/1868

        self._handlers["alpha"] = partial(
            self._config_changed, image=image, name="alpha"
        )
        self._handlers["lut"] = partial(self._config_changed, image=image, name="lut")
        self._handlers["nlut"] = partial(
            self._config_changed, image=image, name="negative_lut"
        )
        self._handlers["min"] = partial(self._config_changed, image=image, name="min")
        self._handlers["minp"] = partial(
            self._set_min_max, image=image, name="minPercent"
        )
        self._handlers["max"] = partial(self._config_changed, image=image, name="max")
        self._handlers["maxp"] = partial(
            self._set_min_max, image=image, name="maxPercent"
        )
        self._handlers["sym"] = partial(
            self._config_bool_changed, image=image, name="symmetric"
        )

        self._alpha.observe(self._handlers["alpha"], names="value")

        self._lut.observe(self._handlers["lut"], names="value")

        self._nlut.observe(self._handlers["nlut"], names="value")

        self._min.observe(self._handlers["min"], names="value")

        self._minp.observe(self._handlers["minp"], names="value")

        self._max.observe(self._handlers["max"], names="value")

        self._maxp.observe(self._handlers["maxp"], names="value")

        self._sym.observe(self._handlers["sym"], names="value")

    def _remove_handlers(self):
        """Removes all event handlers set for the config widgets."""
        if len(self._handlers):
            self._alpha.unobserve(self._handlers["alpha"], names="value")
            self._lut.unobserve(self._handlers["lut"], names="value")
            self._nlut.unobserve(self._handlers["nlut"], names="value")
            self._min.unobserve(self._handlers["min"], names="value")
            self._minp.unobserve(self._handlers["minp"], names="value")
            self._max.unobserve(self._handlers["max"], names="value")
            self._maxp.unobserve(self._handlers["maxp"], names="value")
            self._sym.unobserve(self._handlers["sym"], names="value")

            self._handlers = defaultdict()

    @debounce(0.5)
    def _config_changed(self, change, image, name):
        if name == "min":
            self._minp.unobserve(self._handlers["minp"], names="value")
            self._minp.value = self._get_per_from_value(image.range, change.new)
            self._minp.observe(self._handlers["minp"], names="value")
        elif name == "max":
            self._maxp.unobserve(self._handlers["maxp"], names="value")
            self._maxp.value = self._get_per_from_value(image.range, change.new)
            self._maxp.observe(self._handlers["maxp"], names="value")

        self._set_config(image, name, change.new)

    @debounce(0.5)
    def _set_min_max(self, change, image, name):
        if name == "minPercent":
            self._min.unobserve(self._handlers["min"], names="value")
            self._min.value = self._get_value_from_per(image.range, change.new)
            self._set_config(image, "min", self._min.value)
            self._min.observe(self._handlers["min"], names="value")
        elif name == "maxPercent":
            self._max.unobserve(self._handlers["max"], names="value")
            self._max.value = self._get_value_from_per(image.range, change.new)
            self._set_config(image, "max", self._max.value)
            self._max.observe(self._handlers["max"], names="value")

    def _config_bool_changed(self, change, image, name):
        value = "false"
        if change.new:
            value = "true"
        self._set_config(image, name, value)

    def _set_config(self, image, key, value):
        image.config[key] = value
        self._viewer.set_images()

    def _get_per_from_value(self, range, value):
        return round(value * 100 / range, 0)

    def _get_value_from_per(self, range, per):
        return round(per * range / 100, 2)

    def set_image(self, image):
        """Sets the image whose config values will be viewed/modified using this config widget.
        If image is `None`, all config values are reset.

        Parameters
        ----------
        image: neurolang_ipywidgets.PapayaImage
            image whose config values will be viewed/modified using this config widget.
        """
        if image:
            self._remove_handlers()
            self._set_values(
                image.config, image.range, image.image.get_fdata().flatten()
            )
            self._add_handlers(image)
        else:
            self.reset()

    def reset(self):
        """Resets values for all config widgets."""
        self._remove_handlers()
        self._set_values({}, 100, [])
        self.layout.visibility = "hidden"
