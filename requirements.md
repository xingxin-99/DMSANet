```
# Directory Structure
|__ code
    |__ centralRepo
    |__ classify
    |__ netInitModels
    |__ visible
|__ data
    |__ bci42a
    |__ originalData
        |__ A01E.gdf
        |__ A01E.mat
        |__ ...
    |__ hgd
    |__ originalData
        |__ A01E.mat
        |__ A01T.mat
        |__ ...
```

## BCI-IV-2A Dataset Download
- **Download Link:** [https://www.bbci.de/competition/iv/#download](https://www.bbci.de/competition/iv/#download)
- **Instructions:** Download the dataset files and label files from the provided link. Place them into the `DMSANet/data/bci42a/originalData` directory.
- **Preprocessing:** Use the `DMSANet/code/savaData/parseBci42aFile` script for preprocessing.

## HGD Dataset Download
- **Download Link:** [https://gin.g-node.org/robintibor/high-gamma-dataset/src/master/data](https://gin.g-node.org/robintibor/high-gamma-dataset/src/master/data)
- **Instructions:** Download the dataset files and label files from the provided link. Place them into the `DMSANet/data/hgd/originalData` directory.
- **Preprocessing:** Use the `DMSANet/code/savaData/parseHGDFile` script for preprocessing.

**Note:** The file naming format is "A + subject number + E/T", where "T" indicates the training set and "E" indicates the testing set. For example, "A01T" represents the training set data for the first subject. To ensure uniform processing of dataset files, the files under the `originalData` directory should follow this naming convention. If the file naming does not conform to this convention, modifications are required.

## Using DMSANet
- The `cv.py` and `ho.py` scripts in `DMSANet/codes/classify/` serve as the entry points for using DMSANet.
- Prepare the dataset, and then run `ho.py` to obtain experimental results. The `ho.py` script has two key parameters:
    - `datasetId`: Specifies which dataset to use for the experiment (0: BCI-IV-2a data, 1: HGD data).
    - `network`: Specifies which network to use (default is DMSANet).
