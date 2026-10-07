"""The live application.

    cli.py             command line (python -m tryon.app)
    pipeline.py        one frame end to end: run(), the synchronous loop
    person.py          keypoints, visibility gates, tracking step
    compositing.py     garment over the frame, face / hair / hands in front
    smooth_display.py  the fluid display modes (--smooth, --delayed)
    layer_warp.py      moving a finished garment layer with the torso
    harmonize.py       optional colour harmonisation to the room
"""
