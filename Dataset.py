from tools import *

class BrainMRIDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, meta_data, transform=None):

        self.IDs = meta_data["subject_id"]
        self.image_paths = {}
        self.ages = {}
        for i, ID in enumerate(self.IDs):
            mg_path = data_dir + 'greymatter/wc1sub-' + ID + '_T1w.nii.gz'
            self.image_paths[ID] = mg_path
            self.ages[ID] = meta_data["age"][i]
        self.transform = transform

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, item):
        ID = self.IDs[item]
        MRI = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[ID]))
        MRI = NormalizeGreyMatter(MRI)

        if self.transform is not None:
            MRI = self.transform(PIL.Image.fromarray(MRI))
        else:
            MRI = torch.tensor(MRI, dtype=torch.float32).unsqueeze(0)

        age = torch.tensor(self.ages[ID], dtype=torch.float32).unsqueeze(0)

        return (MRI, age)
