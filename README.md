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

### For specific initial conditions

To predict product state distributions for fixed nitial conditions from the test set (77 data sets).  Go to the `evaluation_InitialCondition` folder.

*Don't remove (external_plotting directory).* 


```
python3 evaluate.py 
```


The `evaluate.py` file predicts product state distributions for all initial conditions within the test set and compares them with reference data obtained from quasi-classical trajectory similations (QCT).

Edit the code `evaluation.py` in the folder `evaluation_InitialCondition` to specify whether accuracy measures should be calculated based on comparison of the NN predictions and QCT data solely at the grid points where the NN places its predictions (flag "NN") or at all points where QCT data is available (flag "QCT") based on linear interpolation. Then run the code to obtain a file containing the desired accuracy measures, as well as a PDF with the corresponding plots. The evaluations are compared with available QCT data located in `QCT_Data/Initial_Condition_Data`.

### For thermal reactant state dsitributions

To predict product state distributions from thermal reactant state distributions go to the `evaluation_Temperature` folder.

Edit the code `evaluation.py` in the folder `evaluation_Temperature`, to specify which of the four studied cases

 - `Ttrans=Trot=Tvib (indices_set1.txt)`
 - `Ttrans != Tvib =Trot (indices_set2.txt)`
 - `Ttrans=Tvib != Trot (indices_set3.txt)` 
 - `Ttrans != Tvib != Trot (indices_set4.txt)` 
 
 you want to analyse.

Then run the code with the following command to obtain a file containing the desired accuracy measures, as well as a PDF with the corresponding plots for three example temperatures. 


*Don't remove (external_plotting directory).* 

```
python3 evaluate.py
```
The evaluations are compared with the available QCT data in `QCT_Data/Temp_Data`.

The complete list of temperatures <img src="https://render.githubusercontent.com/render/math?math=T_{rot}"> and <img src="https://render.githubusercontent.com/render/math?math=T_{vib}"> can be read from the file `tinput.dat` in `data_preprocessing/TEMP/tinput.dat` .


## Cite as: 
 
 Julian Arnold, Debasish Koner, Juan Carlos San Vicente, Narendra Singh, Raymond J. Bemish, and Markus Meuwly,
 ```diff
 !*Complete name of paper or do you want to cite the repository? Also, add an email or responsable*
 ```
