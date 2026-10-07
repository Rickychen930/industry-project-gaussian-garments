"""The garment on the body.

    skinning.py        bind the captured garment to the wearer (linear blend skinning from the scan pose)
    cloth.py           springs toward the skinned pose (LightCloth)
    collision.py       keep the garment outside the body
    renderer.py        GarmentRenderer: pose the mesh, place the splats, shade, draw
    appearance.py      the checkpoint's own shading network (the teacher)
    shading_student.py its small distilled student, cheap enough for every frame
"""
