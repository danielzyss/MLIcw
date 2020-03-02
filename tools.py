from parameters import *


def DataPeek():
    meta_data_all = pd.read_csv(data_dir + 'meta/meta_data_all.csv')
    meta_data_all.head()  # show the first five data entries
    meta_data = meta_data_all

    sns.catplot(x="gender_text", data=meta_data, kind="count")
    plt.title('Gender distribution')
    plt.xlabel('Gender')
    plt.show()

    sns.distplot(meta_data['age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.show()

    plt.scatter(range(len(meta_data['age'])), meta_data['age'], marker='.')
    plt.grid()
    plt.xlabel('Subject')
    plt.ylabel('Age')
    plt.show()


def wl_to_lh(window, level):
    low = level - window / 2
    high = level + window / 2
    return low, high


def display_image(img, x=None, y=None, z=None, window=None, level=None, colormap='gray', crosshair=False):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)

    # Get image dimensions in millimetres
    size = img.GetSize()
    spacing = img.GetSpacing()
    width = size[0] * spacing[0]
    height = size[1] * spacing[1]
    depth = size[2] * spacing[2]

    if x is None:
        x = np.floor(size[0] / 2).astype(int)
    if y is None:
        y = np.floor(size[1] / 2).astype(int)
    if z is None:
        z = np.floor(size[2] / 2).astype(int)

    if window is None:
        window = np.max(img_array) - np.min(img_array)

    if level is None:
        level = window / 2 + np.min(img_array)

    low, high = wl_to_lh(window, level)

    # Display the orthogonal slices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    ax1.imshow(img_array[z, :, :], cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(img_array[:, y, :], origin='lower', cmap=colormap, clim=(low, high), extent=(0, width, 0, depth))
    ax3.imshow(img_array[:, :, x], origin='lower', cmap=colormap, clim=(low, high), extent=(0, height, 0, depth))

    # Additionally display crosshairs
    if crosshair:
        ax1.axhline(y * spacing[1], lw=1)
        ax1.axvline(x * spacing[0], lw=1)
        ax2.axhline(z * spacing[2], lw=1)
        ax2.axvline(x * spacing[0], lw=1)
        ax3.axhline(z * spacing[2], lw=1)
        ax3.axvline(y * spacing[1], lw=1)

    plt.show()


def interactive_view(img):
    size = img.GetSize()
    img_array = sitk.GetArrayFromImage(img)
    interact(display_image, img=fixed(img),
             x=(0, size[0] - 1),
             y=(0, size[1] - 1),
             z=(0, size[2] - 1),
             window=(0, np.max(img_array) - np.min(img_array)),
             level=(np.min(img_array), np.max(img_array)))


def CheckingDataRetrieval(meta_data):
    # Subject with index 0
    ID = meta_data['subject_id'][0]
    age = meta_data['age'][0]

    # Image
    image_filename = data_dir + 'images/sub-' + ID + '_T1w_unbiased.nii.gz'
    img = sitk.ReadImage(image_filename)

    # Mask
    mask_filename = data_dir + 'masks/sub-' + ID + '_T1w_brain_mask.nii.gz'
    msk = sitk.ReadImage(mask_filename)

    # Grey matter map
    gm_filename = data_dir + 'greymatter/wc1sub-' + ID + '_T1w.nii.gz'
    gm = sitk.ReadImage(gm_filename)
    print(sitk.GetArrayFromImage(gm).shape)

    print('Imaging data of subject ' + ID + ' with age ' + str(age))

    print('\nMR Image (used in part A)')
    display_image(img, window=400, level=200)

    print('Brain mask (used in part A)')
    display_image(msk)

    print('Spatially normalised grey matter maps (used in part B and C)')
    display_image(gm)

def NormalizeGreyMatter(image):

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        normalised = (image - mean) / std
    else:
        normalised = image-mean

    return normalised


def plot_regression_scatter(y_true, y_pred):
    plt.scatter(y_pred, label="ground truths")
    plt.scatter(y)