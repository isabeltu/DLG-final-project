# DLG-final-project

link to input dataset mask_crops: https://drive.google.com/file/d/129rzroyDJMvFSkbRF5R5vnGG13ulXTQw/view?usp=sharing

To run, download and unzip mask_crops from the link above and place it in the same directory. Run the first two cells on first launch to sort crops into folders and create file metadata.

Change the object folder in cell three to choose which object you want to train the GAN on.

The fourth cell augments the data for that object if necessary.

The fifth cell does the initial 200 epochs of training and saves the pth files as well as samples from every tenth epoch.

The sixth cell can be utilized if additional training after the intial 200 epochs is needed.
