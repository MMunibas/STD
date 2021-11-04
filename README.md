#  State-to-Distribution (STD) Model

In this repository we provide exemplary code on how to construct and evaluate a state-to-distribution (STD) model for a reactive atom-diatom collision system.

## Requirements

- python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.7
- TensorFlow <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4
- SciKit-learn <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.20

### Setting up the environment

We recommend to use [ Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for the creation of a virtual environment. 

Once in miniconda, you can create a virtual enviroment called *StD* from the `.yml` file with the following command

``` 
conda env create --file StD.yml
```
 
On the same file, there is a version of the required packages. Additionally, a `.txt` file is included, if this is used the necessary command for the creation of the environment is:

```
conda create --file StD.txt 
```

To activate the virtual environment use the command:

```
conda activate StD
```

You are ready to run the code.

## Predict product state distributions

### Initial Conditions

To predict product state distributions for initial conditions from the test set (77 data sets).  Go to the `evaluation_InitialCondition` folder.

*Don't remove (external_plotting directory).* 
```diff
! You need to explain why the external_plotting folder is required and if necessary put it with the provided data. Otherwise when submitting the code set the flag `external_plotting = False`.
```

```
python3 evaluate.py 
```

**It does not run: 
Error: *FileNotFoundError: [Errno 2] No such file or directory: './external_plotting/NN_pe362_NN.txt'* 
**

The `evaluate.py` file predicts product state distributions for all initial conditions within the test set and compares them with reference data obtained from quasi-classical trajectory similations (QCT).

Edit the code `evaluation.py` in the folder `evaluation_InitialCondition` to specify whether accuracy measures should be calculated for "QCT" for NN evaluation and then interpolation to QCT grid or "NN" for evaluation and comparison in the NN grid.  Then run the code to obtain a file containing the desired accuracy measures, as well as a PDF with the corresponding plots.  The evaluations are compared with available QCT data  on `QCT_Data/Initial_Condition_Data`

### Temperatures

For temperature evaluation.  Go to the `evaluation_Temperature` folder

Edit the code `evaluation.py` in the folder `evaluation_Temperature`, to specify which of the (4) studied cases:

 - `Ttrans=Trot=Tvib (indices_set1.txt)`
 - `Ttrans != Tvib =Trot (indices_set2.txt)`
 - `Ttrans=Tvib != Trot (indices_set3.txt)` 
 - `Ttrans != Tvib != Trot (indices_set4.txt)` 
 
 you want to compare.

Then run the code with the following command to obtain a file containing the desired accuracy measures, as well as a PDF with the corresponding plots for (3) example temperatures. 

```
python3 evaluate.py
```

**It does not run: 
Error: *FileNotFoundError: [Errno 2] No such file or directory: './external_plotting/NN_pe362_NN.txt'* 
**

 The evaluations are compared with available QCT data inside `QCT_Data/Temp_Data`

The complete list of temperatures <img src="https://render.githubusercontent.com/render/math?math=T_{rot}"> and <img src="https://render.githubusercontent.com/render/math?math=T_{vib}"> can be read from the file `tinput.dat` in `data_preprocessing/TEMP/tinput.dat` .


## Cite as: 
 
 Julian Arnold, Debasish Koner, Juan Carlos San Vicente, Narendra Singh, Raymond J. Bemish, and Markus Meuwly,
 ```diff
 !*Complete name of paper or do you want to cite the repository? Also, add an email or responsable*
 ```


