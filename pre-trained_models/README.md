## Add here your pre trained models

To download Bert pretrained models from Huggingface repository you can use git [lfs](https://git-lfs.com/).
```
    # Install git large file storage https://git-lfs.com/
    # On ubuntu 22.04 see https://installati.one/install-git-lfs-ubuntu-22-04/
    sudo apt update
    sudo apt -y install git-lfs
    git lfs install

    ## Download the model
    ## For Airbnb we use the main tree of https://huggingface.co/bert-base-uncased
    cd pre-trained_models/
    git clone https://huggingface.co/bert-base-uncased

```
