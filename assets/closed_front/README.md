# Garments worn closed

`<garment>.json` holds `{"closed": true}` for a garment that was scanned open but is worn closed
in the release video (00136_outer: a denim jacket). `GarmentRenderer.CLOSE_FRONT = "auto"` then
brings its two front panels together at bind (`tryon/garment/skinning.py::front_closure`). A
garment without a file is worn as scanned. The repository copy wins over
`<checkpoint>/stage1/closed_front.json`.

Whether a wearer wears a jacket closed cannot be known from one frame, so this is a per-garment
choice, not something the system detects. It measured +0.038 garment IoU on 00136_outer
(docs/LOG.md).
