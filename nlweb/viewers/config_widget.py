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
from neurolang_ipywidgets import NlVBoxOverlay, PapayaNiftiImage

import nibabel as nib

from functools import partial


from plotly.graph_objects import Figure, FigureWidget, Histogram

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


def init():
    papaya_image = PapayaNiftiImage(nib.load("overlay.nii"), dict())
    vb = PapayaConfigWidget(papaya_image)
    return vb


class PapayaConfigWidget(NlVBoxOverlay):
    """A widget that displays widgets to adjust NLPapayaViewer image parameters."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout.width = "720px"

        self._create_widgets(parent.image)

        self.children = [
            HBox(
                [
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
                        layout=Layout(width="220px"),
                    ),
                    VBox(
                        [self._hist],
                        layout=Layout(
                            width="500px", height="250px", margin="5px 5px 5px 5px"
                        ),
                    ),
                ]
            )
        ]

    def _create_widgets(self, image):

        config = image.config

        layout = Layout(width="200px", max_width="200px")

        self._alpha = FloatSlider(
            value=config.get("alpha", 1),
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
            options=lut_options,
            value=config.get("lut", "Red Overlay"),
            description="lut:",
            description_tooltip="The color table name.",
            layout=layout,
        )

        self._nlut = Dropdown(
            options=lut_options,
            value=config.get("lut", "Red Overlay"),
            description="negative-lut:",
            description_tooltip="The color table name used by the negative side of the parametric pair.",
            layout=layout,
        )

        self._min = FloatText(
            value=config.get("min", None),
            description="min:",
            description_tooltip="The display range minimum.",
            disabled=False,
            layout=layout,
        )

        self._minp = BoundedIntText(
            value=config.get("minPercent", None),
            min=0,
            max=100,
            step=1,
            description="min %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=layout,
        )

        self._max = FloatText(
            value=config.get("max", None),
            description="max:",
            description_tooltip="The display range maximum.",
            disabled=False,
            layout=layout,
        )

        self._maxp = BoundedIntText(
            value=config.get("maxPercent", None),
            min=0,
            max=100,
            step=1,
            description="max %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=layout,
        )

        self._sym = Checkbox(
            value=config.get("symmetric", "false") == "true",
            description="symmetric",
            description_tooltip="When selected, sets the negative range of a parametric pair to the same size as the positive range.",
            disabled=False,
            layout=layout,
        )

        self._hist = self._create_hist(image)

        # add handlers
        self._alpha.observe(
            partial(self._config_changed, image=image, name="alpha"), names="value"
        )

        self._lut.observe(
            partial(self._config_changed, image=image, name="lut"), names="value"
        )

        self._nlut.observe(
            partial(self._config_changed, image=image, name="negative_lut"),
            names="value",
        )

        self._min.observe(
            partial(self._config_changed, image=image, name="min"), names="value"
        )

        self._minp.observe(
            partial(self._config_changed, image=image, name="minPercent"), names="value"
        )

        self._max.observe(
            partial(self._config_changed, image=image, name="max"), names="value"
        )

        self._maxp.observe(
            partial(self._config_changed, image=image, name="maxPercent"), names="value"
        )

        self._sym.observe(
            partial(self._config_bool_changed, image=image, name="symmetric"),
            names="value",
        )

    def _config_changed(self, change, image, name):
        image.config[name] = change.new
        self._parent.update_config()

    def _config_bool_changed(self, change, image, name):
        value = "false"
        if change.new:
            value = "true"
        image.config[name] = value
        self._parent.update_config()

    def _create_hist(self, image):
        # flatten image data
        data = image.image.get_fdata().flatten()

        # leave out 0 values
        data0 = data[data != 0]

        fig = Figure()
        fig.add_trace(Histogram(x=data, name="All image data"))
        fig.add_trace(Histogram(x=data0, name="Image data without 0s"))

        fig.update_layout(width=500, height=250, margin=dict(l=15, t=15, b=15, r=15))

        return FigureWidget(fig)
