from collections import defaultdict
from functools import partial
from ipywidgets import (
    BoundedIntText,
    Checkbox,
    Dropdown,
    FloatSlider,
    FloatText,
    HBox,
    Layout,
    VBox,
)
from neurolang_ipywidgets import NlPapayaViewer, NlVBoxOverlay

from plotly.graph_objects import Figure, FigureWidget, Histogram


class PapayaWidget(HBox):
    """A widget class that displays a papaya viewer (NlPapayaViewer) and config widget (PapayaConfigWidget) side by side.


    Implements all methods of NlPapayaViewer.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        **kwargs:
            config_visible: str
                Depending on the value, config widget will be visible or hidden upon initialization. Possible values are "hidden" or "visible"
        """
        super().__init__(*args, **kwargs)

        self._current_widget = None

        self._viewer = NlPapayaViewer(layout=Layout(width="70%", height="auto"))

        self._config = PapayaConfigWidget(
            self._viewer,
            layout=Layout(width="30%", height="auto", border="1px solid black"),
        )

        self._config.layout.visibility = kwargs.get("config_visible", "hidden")

        self.children = [self._viewer, self._config]

    def show_image_config(self, cell_widget, show=True):
        self._config.layout.visibility = "visible" if show else "hidden"
        if show:
            if self._current_widget is not None:
                self._current_widget.reset_config()
            self._config.set_image(cell_widget.image)
            self._current_widget = cell_widget
        else:
            self._config.set_image(None)
            self._current_widget = None

    def can_add(self, images):
        return self._viewer.can_add(images)

    def add(self, images):
        self._viewer.add(images)

    def remove(self, images):
        self._viewer.remove(images)

    def set_images(self):
        self._viewer.set_images()

    def set_center(self, widget, image):
        self._viewer.set_center(widget, image)

    def set_error(self, error):
        self._viewer.set_error()

    def reset(self):
        self._viewer.reset()
        self._config.reset()


class PapayaConfigWidget(NlVBoxOverlay):
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
        """Initializes all configuration widgets. Possible image config parameters are:
        """
        layout = Layout(width="200px", max_width="200px")

        self._alpha = FloatSlider(
            value=1,
            min=0,
            max=1.0,
            step=0.1,
            description="alpha:",
            description_tooltip="Overlay image alpha level (0 to 1).",
            disabled=False,
            continuous_update=False,
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
            disabled=False,
            layout=layout,
        )

        self._minp = BoundedIntText(
            value=None,
            min=0,
            max=100,
            step=1,
            description="min %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=layout,
        )

        self._max = FloatText(
            value=None,
            description="max:",
            description_tooltip="The display range maximum.",
            disabled=False,
            layout=layout,
        )

        self._maxp = BoundedIntText(
            value=None,
            min=0,
            max=100,
            step=1,
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
            legend_orientation="h",
        )

        self._hist = FigureWidget(fig)
        self._hist.add_trace(Histogram(x=[], name="All image data"))
        self._hist.add_trace(Histogram(x=[], name="Image data without 0s"))

        self._handlers = defaultdict()

    def _set_values(self, config, data):
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
        data: []
           flattened image data.
        """
        self._alpha.value = config.get("alpha", 1)
        self._lut.value = config.get("lut", PapayaConfigWidget.lut_options[1])
        self._nlut.value = config.get("lut", PapayaConfigWidget.lut_options[1])
        self._min.value = config.get("min", 0)
        self._minp.value = config.get("minPercent", 100)
        self._max.value = config.get("max", 0.1)
        self._maxp.value = config.get("maxPercent", 100)
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
        print("adding handlers")

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
            self._config_changed, image=image, name="minPercent"
        )
        self._handlers["max"] = partial(self._config_changed, image=image, name="max")
        self._handlers["maxp"] = partial(
            self._config_changed, image=image, name="maxPercent"
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
        """Removes all event handlers set for the config widgets.
        """
        print("removing handlers")
        if len(self._handlers):
            self._alpha.unobserve_all()
            self._lut.unobserve(self._handlers["lut"])
            self._nlut.unobserve(self._handlers["nlut"])
            self._min.unobserve(self._handlers["min"])
            self._minp.unobserve(self._handlers["minp"])
            self._max.unobserve(self._handlers["max"])
            self._maxp.unobserve(self._handlers["maxp"])
            self._sym.unobserve(self._handlers["sym"])
            self._handlers = defaultdict()

    def _config_changed(self, change, image, name):
        print("entered")
        image.config[name] = change.new
        self._viewer.set_images()

    def _config_bool_changed(self, change, image, name):
        value = "false"
        if change.new:
            value = "true"
        image.config[name] = value
        self._viewer.set_images()

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
            self._set_values(image.config, image.image.get_fdata().flatten())
            self._add_handlers(image)
        else:
            self.reset()

    def reset(self):
        """Resets values for all config widgets.
        """
        self._set_values({}, [])
        self._remove_handlers()
