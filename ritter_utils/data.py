# nitorch
from nitorch.data import *
from nitorch.transforms import *


default_transforms = Compose(
    [ToTensor(), IntensityRescale(masked=False, on_gpu=True)]
)


def get_idss(
    path,
    bs=8,
    test_size=0.15,
    z_factor=None,
    num_training_samples=None,
    labels_to_keep=["AD", "CVD"],
    transforms=default_transforms,
    random_state=None,
    balance=False,
    **kwargs,
):
    """Load a pre-defined databunch object for the iDSS data.
     Builds dataloaders with a certain batch size and returns databunch object """
    db = DataBunch(
        source_dir="/analysis/ritter/data/iDSS",
        table="tables/mri_complete_4_class_minimal.csv",
        path=path,
        mask=f"/analysis/ritter/data/PPMI/Mask/mask_T1.nii",  # same mask as T1 ppmi scans
        labels_to_keep=labels_to_keep,
        transforms=transforms,
        random_state=random_state,
        balance=balance,
        test_size=test_size,
        z_factor=z_factor,
        num_training_samples=num_training_samples,
        **kwargs,
    )
    db.build_dataloaders(bs=bs)
    return db


def get_adni(
    path,
    bs=8,
    test_size=0.15,
    z_factor=None,
    num_training_samples=None,
    labels_to_keep=["Dementia", "CN"],
    transforms=default_transforms,
    mask="/analysis/ritter/data/ADNI/binary_brain_mask.nii.gz",
    grouped=True,
    balance=False,
    **kwargs,
):
    """Load a pre-defined databunch object for the ADNI data.
     Builds dataloaders with a certain batch size and returns databunch object """

    db = DataBunch(
        source_dir="/analysis/ritter/data/ADNI",
        image_dir="ADNI_2Yr_15T_quick_preprocessed",
        table="ADNI_tables/customized/DxByImgClean_CompleteAnnual2YearVisitList_1_5T.csv",
        path=path,
        mask=mask,
        labels_to_keep=labels_to_keep,
        transforms=transforms,
        random_state=1337,
        grouped=grouped,
        balance=balance,
        test_size=test_size,
        num_training_samples=num_training_samples,
        z_factor=z_factor,
        **kwargs,
    )
    db.build_dataloaders(bs=bs)
    return db


def get_ppmi(
    path,
    bs=8,
    test_size=0.15,
    mri_type="T2",
    z_factor=None,
    num_training_samples=None,
    labels_to_keep=["PD", "HC"],
    transforms=default_transforms,
    random_state=None,
    balance=False,
    **kwargs,
):
    """Load a pre-defined databunch object for the PPMI data.
     Builds dataloaders with a certain batch size and returns databunch object """

    mri_type = mri_type.upper()
    assert mri_type in [
        "T1",
        "T2",
    ], "Argument mri_type has to be one of T1 or T2"
    if mri_type == "T2":
        z_factor = 0.87

    db = DataBunch(
        source_dir="/analysis/ritter/data/PPMI",
        table=f"tables/PPMI_{mri_type}.csv",
        path=path,
        mask=f"/analysis/ritter/data/PPMI/Mask/mask_{mri_type}.nii",
        labels_to_keep=labels_to_keep,
        random_state=random_state,
        balance=balance,
        test_size=test_size,
        num_training_samples=num_training_samples,
        z_factor=z_factor,
        transforms=transforms,
        **kwargs,
    )
    db.build_dataloaders(bs=bs)
    return db


def get_biobank(
    path,
    label_col,
    bs=8,
    test_size=0.2,
    mri_type="T2",
    z_factor=None,
    num_samples=None,
    labels_to_keep=None,
    transforms=default_transforms,
    random_state=None,
    balance=False,
    **kwargs,
):
    """Load a pre-defined databunch object for the UKBiobank data.
     Builds dataloaders with a certain batch size and returns databunch object """

    mri_type = mri_type.upper()
    assert mri_type in [
        "T1",
        "T2",
    ], "Argument mri_type has to be one of T1 or T2"

    db = DataBunch(
        source_dir="/analysis/ritter/data/UKbiobank/UKBiobank_BIDS",
        table=f"../tables/participants_transMRI_{mri_type}.csv",
        path=path,
        mask=None,
        label_col=label_col,
        labels_to_keep=labels_to_keep,
        random_state=random_state,
        balance=balance,
        test_size=test_size,
        num_samples=num_samples,
        z_factor=z_factor,
        transforms=transforms,
        **kwargs,
    )
    db.build_dataloaders(bs=bs)
    return db
