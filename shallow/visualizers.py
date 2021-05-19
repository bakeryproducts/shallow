import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def show_images(image_batch, by_channels=False):
    columns = image_batch.shape[1] if by_channels else 1
    batch_size = image_batch.shape[0] if len(image_batch.shape) == 4 else 1
    #image_batch = image_batch.squeeze()


    fig = plt.figure(figsize = (32,32//batch_size))
    gs = gridspec.GridSpec(batch_size, columns)
    for i in range(batch_size):
        if by_channels:
            for j in range(columns):
                k = j + i*columns
                plt.subplot(gs[k])
                plt.axis("off")
                plt.imshow(image_batch[i,j])
        else:
            plt.subplot(gs[i])
            plt.axis("off")
            img = image_batch[i].squeeze()
            if len(img.shape) > 2:
                if img.shape[0] == 3:
                    img = np.transpose(img, (1,2,0))
            plt.imshow(img)
