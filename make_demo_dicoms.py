"""Creates the synthetic DICOM demo cases in samples/dicom/ from the PNG samples.
Patient names and IDs are fictional; the images come from the PneumoniaMNIST test set.
Usage: python make_demo_dicoms.py
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dicom_io import make_dicom  # noqa: E402

S = Path(__file__).parent / "samples"
CASES = [  # file name, source png, patient id, name, birth date, sex, study date, referring physician, description
    ("TR-100231_2026-10-02_prior.dcm", "pneumonia_3.png", "TR-100231", "DEMO^Ayse", "20210314", "F", "20261002", "DEMO^Dr Kaya", "CHEST PA - fever, cough"),
    ("TR-100231_2026-11-24_followup.dcm", "normal_1.png", "TR-100231", "DEMO^Ayse", "20210314", "F", "20261124", "DEMO^Dr Kaya", "CHEST PA - follow-up after treatment"),
    ("TR-100587_2026-11-25.dcm", "pneumonia_1.png", "TR-100587", "DEMO^Mehmet", "20200607", "M", "20261125", "DEMO^Dr Aydin", "CHEST PA - shortness of breath"),
    ("TR-100902_2026-11-25.dcm", "pneumonia_2.png", "TR-100902", "DEMO^Zeynep", "20220120", "F", "20261125", "DEMO^Dr Aydin", "CHEST PA - suspected pneumonia"),
    ("TR-100344_2026-11-23.dcm", "normal_2.png", "TR-100344", "DEMO^Can", "20190911", "M", "20261123", "DEMO^Dr Kaya", "CHEST PA - pre-operative"),
]

if __name__ == "__main__":
    out = S / "dicom"; out.mkdir(exist_ok=True)
    for fn, png, pid, name, birth, sex, date, ref, desc in CASES:
        img = Image.open(S / png).convert("L").resize((512, 512), Image.BICUBIC)
        make_dicom(img, pid, name, birth, sex, date, referring=ref, description=desc).save_as(out / fn, enforce_file_format=True)
        print("wrote", out / fn)
