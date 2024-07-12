import openslide

def read_slide(path):
    #Slide properties
    slide = openslide.OpenSlide(path)
    if "tiff.XResolution" in slide.properties.keys():
        spacing = 1 / (float(slide.properties["tiff.XResolution"]) / 10000)
    elif "openslide.mpp-x" in slide.properties.keys():
        spacing = float(slide.properties["openslide.mpp-x"])
    else:
        print("using default spacing value")
        spacing = 0.25
        # raise ValueError("Not able to find spacing")
    print(f"Slide spacing: {spacing}")
    return slide, spacing