"""DICOM support: read chest X-rays in the hospital standard format and de-identify them.

- read_dicom(): pixel data -> 8-bit grayscale image (applies VOI windowing / rescale and
  inverts MONOCHROME1), plus the clinically relevant header fields.
- anonymize(): removes direct identifiers following the spirit of the DICOM PS3.15 Basic
  Application Level Confidentiality Profile and replaces the Patient ID with a stable pseudonym
  (salted SHA-256), so repeated studies of the same patient can still be linked.
"""
import hashlib
import io

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_voi_lut

# Direct identifiers removed by anonymize() (subset of PS3.15 Table E.1-1 relevant to plain radiographs)
IDENTIFYING_TAGS = [
    "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
    "OtherPatientNames", "PatientMotherBirthName", "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "InstitutionName", "InstitutionAddress", "AccessionNumber", "StationName",
]
PSEUDO_SALT = "comp463-pneumoscan"


def pseudonym(patient_id: str) -> str:
    return "PS-" + hashlib.sha256((PSEUDO_SALT + str(patient_id)).encode()).hexdigest()[:8].upper()


def _fmt_date(d):
    d = str(d or "")
    return f"{d[6:8]}.{d[4:6]}.{d[:4]}" if len(d) == 8 else "-"


def read_dicom(file_or_bytes):
    raw = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(np.float32)
    if "WindowCenter" in ds or "VOILUTSequence" in ds:  # use the display window chosen by the modality
        arr = apply_voi_lut(arr, ds).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
    else:  # no window: map the full stored bit depth linearly to 0-255
        lo, hi = 0.0, float(2 ** int(ds.get("BitsStored", 8)) - 1)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 1.0 - arr  # MONOCHROME1 stores bone as dark
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    birth = str(ds.get("PatientBirthDate", "") or "")
    meta = {
        "PatientID": str(ds.get("PatientID", "") or ""),
        "PatientName": str(ds.get("PatientName", "") or ""),
        "BirthYear": int(birth[:4]) if birth[:4].isdigit() else None,
        "PatientSex": str(ds.get("PatientSex", "") or ""),
        "StudyDate": _fmt_date(ds.get("StudyDate")),
        "Modality": str(ds.get("Modality", "") or ""),
        "BodyPart": str(ds.get("BodyPartExamined", "") or ""),
        "View": str(ds.get("ViewPosition", "") or ""),
        "Institution": str(ds.get("InstitutionName", "") or ""),
        "ReferringPhysician": str(ds.get("ReferringPhysicianName", "") or ""),
        "StudyDescription": str(ds.get("StudyDescription", "") or ""),
        "Size": f"{ds.get('Columns', '?')}×{ds.get('Rows', '?')}",
    }
    return img, meta, ds


def anonymize(ds):
    """Returns (de-identified dataset copy, list of removed tag names)."""
    ds = ds.copy()
    removed = []
    for tag in IDENTIFYING_TAGS:
        if tag in ds:
            removed.append(tag)
            delattr(ds, tag)
    ds.PatientID = pseudonym(ds.get("PatientID", ""))
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "PS3.15 basic profile subset; pseudonymised ID"
    ds.remove_private_tags()
    return ds, removed


def make_dicom(img: Image.Image, patient_id, patient_name, birth_date, sex, study_date,
               institution="MEF Demo Hospital", referring="", description="CHEST PA"):
    """Wraps a grayscale image in a minimal valid DX DICOM file (used to create the demo cases)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"  # Digital X-Ray Image Storage - For Presentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientID, ds.PatientName, ds.PatientBirthDate, ds.PatientSex = patient_id, patient_name, birth_date, sex
    ds.StudyDate, ds.Modality, ds.BodyPartExamined, ds.ViewPosition = study_date, "DX", "CHEST", "PA"
    ds.InstitutionName, ds.ReferringPhysicianName, ds.StudyDescription = institution, referring, description
    ds.AccessionNumber = "ACC" + patient_id[-5:]
    a = np.array(img.convert("L"), dtype=np.uint16) * 16  # 12-bit range like a real detector
    ds.Rows, ds.Columns = a.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = a.tobytes()
    return ds
