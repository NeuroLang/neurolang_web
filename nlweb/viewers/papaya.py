from . import ColumnViewer

import base64  # type: ignore

from ipywidgets import HTML

import html  # type: ignore

import json  # type: ignore

import nibabel as nib  # type: ignore

import numpy as np  # type: ignore


def encode_images(images):
    encoded_images = []
    image_txt = []
    for i, image in enumerate(images):
        nifti_image = nib.Nifti2Image(image.get_fdata(), affine=image.affine)
        encoded_image = base64.encodebytes(nifti_image.to_bytes())
        del nifti_image
        image_txt.append(f"image{i}")
        enc = encoded_image.decode("utf8").replace("\n", "")
        encoded_images.append(f'var {image_txt[-1]}="{enc}";')

    encoded_images = "\n".join(encoded_images)
    return encoded_images, image_txt


class PapayaViewerWidget(HTML, ColumnViewer):
    """A viewer that overlays multiple label maps.
    
    Number of label maps to overlay is limited to 8. ??
    """

    encoder = json.JSONEncoder()

    # papaya parameters
    params = {"kioskMode": False, "worldSpace": True, "fullScreen": False}

    # html necessary to embed papaya viewer
    papaya_html = """
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml" lang="en">
            <head>
                <link rel="stylesheet" type="text/css" href="https://raw.githack.com/rii-mango/Papaya/master/release/current/standard/papaya.css" />
                <script type="text/javascript" src="https://raw.githack.com/rii-mango/Papaya/master/release/current/standard/papaya.js"></script>
                <title>Papaya Viewer</title>

                <script type="text/javascript">

                    {encoded_images}

                    var params={params};
                </script>
            </head>

            <body>
                <div class="papaya" data-params="params"></div>
            </body>
        </html>
    """

    def __init__(self, atlas="avg152T1_brain.nii.gz", *args, **kwargs):
        """Initializes the widget with the specified `atlas`.
        
        Parameters
        ----------
        atlas: str
            path for the image file to be used as atlas.
        """
        super().__init__(*args, **kwargs)

        # load atlas and add it to image list
        self.atlas_image = nib.load(atlas)
        self.images = [self.atlas_image]
        self._center = None
        self._center_coords = None

        # initially plot the atlas
        self.plot()

    def add(self, images):
        """Adds the specified `image` to the image list of this viewer.
        
        Parameters
        ----------
        images: list
            images to be added to the list of this viewer.
        """
        for image in images:
            self.images.append(image)
        self.plot()

    def remove(self, images):
        """Removes the specified `images` from the image list of this viewer.
        
        Parameters
        ----------
        images: list
            image to be removed from the image list of this viewer.
        """

        for image in images:
            self.images.remove(image)
        self.plot()

    def set_center(self, widget, image):
        """"""
        if self._center is not None:
            self._center.remove_center()
            self._center_coords = None

        # think of this
        if image is not None:
            self._center = widget
            self._center_coords = PapayaViewerWidget.calculate_coords(image)
        self.plot()

    def plot(self):
        """Plots all images in the image list of this viewer.
        
        Note
        ----
        As papaya has a limit of 8 images, it can display only 8 images overlaid. Selection of images depends on the implementation of papaya.
        """

        # set center_image as the last appended image if not specified
        if self._center is None and len(self.images) > 0:
            self._center_coords = PapayaViewerWidget.calculate_coords(self.images[-1])

        # encode images
        encoded_images, image_names = encode_images(self.images)

        # set params variable for papaya
        params = dict()
        params.update(PapayaViewerWidget.params)
        params["encodedImages"] = image_names
        if self._center_coords is not None:
            params["coordinate"] = self._center_coords

        for image_name in image_names[1:]:
            params[image_name] = {"min": 0, "max": 10, "lut": "Red Overlay"}

        escaped_papaya_html = html.escape(
            PapayaViewerWidget.papaya_html.format(
                params=PapayaViewerWidget.encoder.encode(params),
                encoded_images=encoded_images,
            )
        )
        iframe = (
            f'<iframe srcdoc="{escaped_papaya_html}" id="papaya" '
            f'width="700px" height="600px"></iframe>'
        )
        self.value = iframe

    @staticmethod
    def calculate_coords(image):
        """Calculates coordinates for the specified `image`."""
        coords = np.transpose(image.get_fdata().nonzero()).mean(0).astype(int)
        coords = nib.affines.apply_affine(image.affine, coords)
        return [int(c) for c in coords]

    def reset(self):
        self.images = [self.atlas_image]
        self._center = None
        self.plot()
