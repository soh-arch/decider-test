"""Frame a rendered figure as a rounded card: transparent margin, light border, rounded corners."""
import sys

from PIL import Image, ImageDraw

MARGIN, RADIUS, BORDER, SCALE = 16, 36, 2, 4
BORDER_COLOR = (0xD8, 0xD3, 0xCB, 255)


def card(src, dst):
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    size = (w + 2 * MARGIN, h + 2 * MARGIN)
    box = (MARGIN * SCALE, MARGIN * SCALE, (MARGIN + w) * SCALE - 1, (MARGIN + h) * SCALE - 1)
    big = (size[0] * SCALE, size[1] * SCALE)

    mask = Image.new("L", big, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, RADIUS * SCALE, fill=255)
    ring = Image.new("L", big, 0)
    ImageDraw.Draw(ring).rounded_rectangle(box, RADIUS * SCALE, outline=255, width=BORDER * SCALE)
    mask, ring = mask.resize(size, Image.LANCZOS), ring.resize(size, Image.LANCZOS)

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.paste(im, (MARGIN, MARGIN))
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(layer, (0, 0), mask)
    out = Image.composite(Image.new("RGBA", size, BORDER_COLOR), out, ring)
    out.save(dst)


if __name__ == "__main__":
    card(sys.argv[1], sys.argv[2])
