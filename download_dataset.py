"""
https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
"""

import requests, zipfile, io, os

if __name__ == "__main__":
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    # download images
    for ds in ["train", "val", "test"]:
        folder = os.path.join("dataset", ds)
        if not os.path.exists(folder):
            print("Downloading", ds, "images.")
            os.mkdir(folder)
            with requests.get("http://images.cocodataset.org/zips/{}2017.zip".format(ds)) as r:
                if r.ok:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    z.extractall(folder)
                else:
                    print("error")
        else:
            print("Already downloaded", ds, "images.")

    # download annotations
    annotation_folder = os.path.join("dataset", "annotations")
    if not os.path.exists(annotation_folder):
        os.mkdir(annotation_folder)
        print("Downloading annotations.")
        with requests.get("http://images.cocodataset.org/annotations/annotations_trainval2017.zip") as r:
            if r.ok:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(annotation_folder)
            else:
                print("error")
    else:
        print("Already downloaded annotations.")
