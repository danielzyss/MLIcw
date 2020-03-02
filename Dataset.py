from tools import *

class BrainMRIDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, meta_data, test=False, transform=None, chunk=True, n_chunk=10):

        self.test = test

        self.IDs = meta_data["subject_id"]
        self.image_paths = {}
        self.ages = {}
        for i, ID in enumerate(self.IDs):
            mg_path = data_dir + 'greymatter/wc1sub-' + ID + '_T1w.nii.gz'
            self.image_paths[ID] = mg_path
            self.ages[ID] = meta_data["age"][i]
        self.transform = transform
        self.chunk = chunk
        self.n_chunk = n_chunk

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, item):
        ID = self.IDs[item]
        MRI = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[ID]))
        MRI = NormalizeGreyMatter(MRI)

        if self.chunk and not self.test:
            MRI = self.ChunkDownImage(MRI)
            MRI = MRI[np.random.randint(0, MRI.shape[0], 1)]
        if self.chunk and self.test:
            MRI = self.ChunkDownImage(MRI)

        if self.transform is not None:
            MRI = self.transform(PIL.Image.fromarray(MRI))
        else:
            if self.chunk and self.test:
                MRI = torch.tensor(MRI, dtype=torch.float32).unsqueeze(1)
            else:
                MRI = torch.tensor(MRI, dtype=torch.float32).unsqueeze(0)

        age = torch.tensor(self.ages[ID], dtype=torch.float32).unsqueeze(0)

        return (MRI, age)

    def ChunkDownImage(self, MRI):

        chunks = []
        for i in range(0, self.n_chunk):
            chunks.append(MRI[:,:,i*int(MRI.shape[2]/self.n_chunk):(i+1)*int(MRI.shape[2]/self.n_chunk)])
        return np.array(chunks)
