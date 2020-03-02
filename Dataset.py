from tools import *

class BrainMRIDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, meta_data, test=False, augment=False, chunk=True, n_chunk=10):

        self.test = test

        self.IDs = meta_data["subject_id"]
        self.image_paths = {}
        self.ages = {}
        for i, ID in enumerate(self.IDs):
            mg_path = data_dir + 'greymatter/wc1sub-' + ID + '_T1w.nii.gz'
            self.image_paths[ID] = mg_path
            self.ages[ID] = meta_data["age"][i]
        self.augment = augment
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

        if self.augment:
            MRI = self._transform(MRI)

        if self.chunk and self.test:
            MRI = torch.tensor(MRI, dtype=torch.float32).unsqueeze(1)
        elif self.chunk:
            MRI = torch.tensor(MRI, dtype=torch.float32)
        else:
            MRI = torch.tensor(MRI, dtype=torch.float32).unsqueeze(0)

        age = torch.tensor(self.ages[ID], dtype=torch.float32).unsqueeze(0)

        return (MRI, age)

    def ChunkDownImage(self, MRI):

        chunks = []
        for i in range(0, self.n_chunk):
            chunks.append(MRI[:,:,i*int(MRI.shape[2]/self.n_chunk):(i+1)*int(MRI.shape[2]/self.n_chunk)])
        return np.array(chunks)

    def _transform(self, MRI, p_flip=0.5):

        # random flip
        if random.random() > p_flip:
            MRI = np.flip(MRI, -1)

        # random crop
        crop = random.randint(0,10)
        x,y,z = MRI.shape
        MRI = MRI[crop:-crop, crop:-crop, crop:-crop]
        MRI = zoom(MRI, (x/(x-2*crop), y/(y-2*crop), z/(z-2*crop)))

        # random shift
        shift_val = random.randint(-8, 8)
        MRI = shift(MRI, shift_val)

        return MRI
