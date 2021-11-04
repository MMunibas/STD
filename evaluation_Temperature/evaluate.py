import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from scipy import interpolate

#path to Temperature distributions

path='../QCT_Data/Temp_Data/'

# function definition

def shiftsoftplus(x):
    return np.log(np.exp(x)+1)-np.log(2)

def softplus(x):
    return np.log(np.exp(x)+1)

# whether to plot the data and the corresponding predictions
plotting = True
if plotting:
    pdf = matplotlib.backends.backend_pdf.PdfPages("predictions.pdf")
    plt.rcParams.update({'figure.figsize': [5, 3.5]})
    plt.rcParams.update({'legend.frameon': False})


# If external plotting, extract he NN temperature evaluation
external_plotting = True

# whether an accuracy evaluation should be done, i.e., if RMSD and R2 values should be calculated
calculate_accuracy_measures = True
if calculate_accuracy_measures:
    RMSD_list_Etrans_prod = []
    RMSD_list_v_prod = []
    RMSD_list_j_prod = []
    R2_list_Etrans_prod = []
    R2_list_v_prod = []
    R2_list_j_prod = []



# accuracy_type should only be QCT
accuracy_type = 'QCT'


# define function which finds returns the index of the element in an array closest to a specific value


def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

# define activation function for the hidden layers


def shiftedsoftplus(x):
    return np.log(np.exp(x)+1)-np.log(2)


# define functions for the RKHS method


# gaussian reproducing kernel
def gauss_ker(x, xi, sigma):
    return np.exp(-abs(x-xi)**2/(2*sigma**2))


# define the grids for the product state distributions
DTDgrid_Etrans_prod = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5,
                             5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
DTDgrid_v_prod = np.array([0, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 47])
DTDgrid_j_prod = np.array([0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 220, 240])

# define the grids for the produc state distributions STD

Egridout = []
i = 0.0
while (i+0.1) < 1.0:
    Egridout.append(i)
    i = i + 0.1
while (i+0.2) < 6.0:
    Egridout.append(i)
    i = i + 0.2
while (i+0.1) < 6.4:
    Egridout.append(i)
    i = i + 0.1
while (i+0.2) < 10.0:
    Egridout.append(i)
    i = i + 0.2
Egridout.append(10.0)
Egridout = np.array(Egridout)

vgridout = []
i = 0
while (i + 1) < 47:
    vgridout.append(i)
    i = i + 1
vgridout.append(47)
vgridout = np.array(vgridout)

jgridout = []
i = 0
while (i + 6) < 240:
    jgridout.append(i)
    i = i + 6
jgridout.append(240)
jgridout = np.array(jgridout)




# calculate the number of features and outputs based on these grids
num_features = 11
num_outputs = len(Egridout)+len(vgridout)+len(jgridout)

print('Number of features: ' + str(num_features))
print('Number of outputs: ' + str(num_outputs))



# Import optimized NN weights and biases

h0W = np.loadtxt('../training/Coeff_h0W.dat', delimiter=',', unpack=True)
h0b = np.loadtxt('../training/Coeff_h0b.dat', delimiter=',', unpack=True)
h1W = np.loadtxt('../training/Coeff_h1W.dat', delimiter=',', unpack=True)
h1b = np.loadtxt('../training/Coeff_h1b.dat', delimiter=',', unpack=True)
h2W = np.loadtxt('../training/Coeff_h2W.dat', delimiter=',', unpack=True)
h2b = np.loadtxt('../training/Coeff_h2b.dat', delimiter=',', unpack=True)
h3W = np.loadtxt('../training/Coeff_h3W.dat', delimiter=',', unpack=True)
h3b = np.loadtxt('../training/Coeff_h3b.dat', delimiter=',', unpack=True)
h4W = np.loadtxt('../training/Coeff_h4W.dat', delimiter=',', unpack=True)
h4b = np.loadtxt('../training/Coeff_h4b.dat', delimiter=',', unpack=True)
h5W = np.loadtxt('../training/Coeff_h5W.dat', delimiter=',', unpack=True)
h5b = np.loadtxt('../training/Coeff_h5b.dat', delimiter=',', unpack=True)
h6W = np.loadtxt('../training/Coeff_h6W.dat', delimiter=',', unpack=True)
h6b = np.loadtxt('../training/Coeff_h6b.dat', delimiter=',', unpack=True)
h7W = np.loadtxt('../training/Coeff_h7W.dat', delimiter=',', unpack=True)
h7b = np.loadtxt('../training/Coeff_h7b.dat', delimiter=',', unpack=True)
h8W = np.loadtxt('../training/Coeff_h8W.dat', delimiter=',', unpack=True)
h8b = np.loadtxt('../training/Coeff_h8b.dat', delimiter=',', unpack=True)
h9W = np.loadtxt('../training/Coeff_h9W.dat', delimiter=',', unpack=True)
h9b = np.loadtxt('../training/Coeff_h9b.dat', delimiter=',', unpack=True)
h10W = np.loadtxt('../training/Coeff_h10W.dat', delimiter=',', unpack=True)
h10b = np.loadtxt('../training/Coeff_h10b.dat', delimiter=',', unpack=True)
h11W = np.loadtxt('../training/Coeff_h11W.dat', delimiter=',', unpack=True)
h11b = np.loadtxt('../training/Coeff_h11b.dat', delimiter=',', unpack=True)
h12W = np.loadtxt('../training/Coeff_h12W.dat', delimiter=',', unpack=True)
h12b = np.loadtxt('../training/Coeff_h12b.dat', delimiter=',', unpack=True)
h13W = np.loadtxt('../training/Coeff_h13W.dat', delimiter=',', unpack=True)
h13b = np.loadtxt('../training/Coeff_h13b.dat', delimiter=',', unpack=True)
h14W = np.loadtxt('../training/Coeff_h14W.dat', delimiter=',', unpack=True)
h14b = np.loadtxt('../training/Coeff_h14b.dat', delimiter=',', unpack=True)
h15W = np.loadtxt('../training/Coeff_h15W.dat', delimiter=',', unpack=True)
h15b = np.loadtxt('../training/Coeff_h15b.dat', delimiter=',', unpack=True)
h16W = np.loadtxt('../training/Coeff_h16W.dat', delimiter=',', unpack=True)
h16b = np.loadtxt('../training/Coeff_h16b.dat', delimiter=',', unpack=True)
h17W = np.loadtxt('../training/Coeff_h17W.dat', delimiter=',', unpack=True)
h17b = np.loadtxt('../training/Coeff_h17b.dat', delimiter=',', unpack=True)
# h18W = np.loadtxt('Coeff_h18W.dat', delimiter=',', unpack=True)
# h18b = np.loadtxt('Coeff_h18b.dat', delimiter=',', unpack=True)
# h19W = np.loadtxt('Coeff_h19W.dat', delimiter=',', unpack=True)
# h19b = np.loadtxt('Coeff_h19b.dat', delimiter=',', unpack=True)
# h20W = np.loadtxt('Coeff_h20W.dat', delimiter=',', unpack=True)
# h20b = np.loadtxt('Coeff_h20b.dat', delimiter=',', unpack=True)
outW = np.loadtxt('../training/Coeff_outW.dat', delimiter=',', unpack=True)
outb = np.loadtxt('../training/Coeff_outb.dat', delimiter=',', unpack=True)

# import mean values and standard deviations for standardization

mval_input = np.loadtxt('../training/Coeff_mval_input.txt', delimiter=',', unpack=True)
stdv_input = np.loadtxt('../training/Coeff_stdv_input.txt', delimiter=',', unpack=True)
mval_output = np.loadtxt('../training/Coeff_mval_output.txt', delimiter=',', unpack=True)
stdv_output = np.loadtxt('../training/Coeff_stdv_output.txt', delimiter=',', unpack=True)



# Import mean values and stdvs for normalization
features = np.zeros(num_features)
non_stand_features = np.zeros(num_features)

# Calculation for tempearture set
temp = []
temp = pd.read_csv("../data_preprocessing/Temp/tinput.dat",delimiter=' ',header=None)
temp.columns = ['Trans', 'Tvib', 'Trot']


# Change the desire case

#Ttrans=Trot=Tvib (indices_set1.txt); Ttrans !=Trot=Tvib (indices_set2.txt)
#Ttrans=Trot!=Tvib (indices_set3.txt); Ttrans !=Trot!=Tvib (indices_set4.txt)
test_indices = np.genfromtxt('../data_preprocessing/Temp/indices_set1.txt')

test_indices = test_indices.astype(int)

# Define number of temperatures to evaluate


# loop through all temperature 
output = np.zeros(num_outputs)
aout = np.zeros(num_outputs)
type = ''

###############################################

#loop through all temperature data set

for ii in test_indices:
    E_pred=[]
    E_pred=np.zeros(58)
    v_pred=[]
    v_pred=np.zeros(47)
    j_pred=[]
    j_pred=np.zeros(40)
    temp_E=[]
    temp_E=np.zeros(58)
    temp_v=[]
    temp_v=np.zeros(47)
    temp_j=[]
    temp_j=np.zeros(40)

    # Transform to integer and print current temperature set
    ttrans = np.int16(temp['Trans'][ii])
    tvib = np.int16(temp['Tvib'][ii])
    trot = np.int16(temp['Trot'][ii])

    print('Importing :'  + "T" + str(ttrans)+ "-T" + str(tvib) + "-T" + str(trot) +  ".txt")

    # Import STD data for a particular temperature 
    input_data = np.genfromtxt("../data_preprocessing/Temp/INPUT/" + "T" + str(ttrans)+ "-T" + str(tvib) + "-T" + str(trot) +  ".txt", delimiter=',')

    data = np.zeros((input_data.shape[0], input_data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = input_data[i, j]

    # Evaluate for 10000 STD inital conditions

    lenghindi=10000

    for i in range(lenghindi):
        # Calculate the STD predictions for a given set of initial conditions
        
        # Obtain normalized features and output from the dataset
        for j in range(num_features):
            non_stand_features[j] = data[i][j]
            features[j] = (data[i][j] - mval_input[j]) / stdv_input[j]

        if non_stand_features[0] != 20.0:

            a0 = shiftsoftplus(np.matmul(h0W, features) + h0b)
            a1 = shiftsoftplus(np.matmul(h1W, a0) + h1b)
            a2 = shiftsoftplus((np.matmul(h2W, a1) + h2b)+a0)

            a3 = shiftsoftplus(np.matmul(h3W, a2) + h3b)
            a4 = shiftsoftplus(np.matmul(h4W, a3) + h4b)
            a5 = shiftsoftplus((np.matmul(h5W, a4) + h5b)+a3)

            a6 = shiftsoftplus(np.matmul(h6W, a5) + h6b)
            a7 = shiftsoftplus(np.matmul(h7W, a6) + h7b)
            a8 = shiftsoftplus((np.matmul(h8W, a7) + h8b)+a6)

            a9 = shiftsoftplus(np.matmul(h9W, a8) + h9b)
            a10 = shiftsoftplus(np.matmul(h10W, a9) + h10b)
            a11 = shiftsoftplus((np.matmul(h11W, a10) + h11b)+a9)

            a12 = shiftsoftplus(np.matmul(h12W, a11) + h12b)
            a13 = shiftsoftplus(np.matmul(h13W, a12) + h13b)
            a14 = shiftsoftplus((np.matmul(h14W, a13) + h14b)+a12)

            a15 = shiftsoftplus(np.matmul(h15W, a14) + h15b)
            a16 = shiftsoftplus(np.matmul(h16W, a15) + h16b)
            a17 = shiftsoftplus((np.matmul(h17W, a16) + h17b)+a15)

            # a18 = shiftsoftplus(np.matmul(h18W, a17) + h18b)
            # a19 = shiftsoftplus(np.matmul(h19W, a18) + h19b)
            # a20 = shiftsoftplus((np.matmul(h20W, a19) + h20b)+a18)

            # calculate the predicted amplitudes

            aout_unscaled = softplus(np.matmul(outW, a17) + outb)
            aout = aout_unscaled * stdv_output

            p_pred_E = aout[:len(Egridout)]
            p_pred_v = aout[len(Egridout):len(vgridout)+len(Egridout)]
            p_pred_j = aout[len(vgridout)+len(Egridout):num_outputs]
            
            ###################################################################################

            # Sum the N initial conditions for a given temperature. 
            # To compare with QCT, STD_DTD_factor must be applied bmax_STD=12 bmax_DTD=10; (12^2/10^2)=1.44

            E_pred=E_pred + p_pred_E # p_pred_E has the 58 predictions, and we are adding over the 10000 initial conditions.
            #------------------------------------------------
            
            # similarly for the product vibrational state distributions
            
            v_pred=v_pred + p_pred_v

            #------------------------------------------------

            # similarly for the product rotational state distributions

            j_pred=j_pred + p_pred_j
            
    ####################################################################
    # Total sum is averaged by dividing by the number of initial conditions.
    # To compare with QCT, STD_DTD_factor must be applied bmax_STD=12 bmax_DTD=10; (12^2/10^2)=1.44

    temp_E=(E_pred/lenghindi)*1.44
    temp_v=(v_pred/lenghindi)*1.44
    temp_j=(j_pred/lenghindi)*1.44

    #####################################################################
            
    # load the product relative translational energy distribution data
    x, p = np.loadtxt(path + 'pevj' + str(ii) + '.dat', unpack=True)
    
    if calculate_accuracy_measures:
        # 1D-interpolation from predicted amplitudes.
        f = interpolate.interp1d(Egridout, temp_E, fill_value="extrapolate")#,fill_value="extrapolate")

        #Evaluate at the points where QCT data is available
        p_pred_acc=f(x)
        p_pred_acc=p_pred_acc/np.trapz(p, x) 


    # if RMSD_QCT and R2_QCT should be calculated
        if accuracy_type == 'QCT':
        
            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                else:
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))

            p_ref_acc = p_ref_acc/np.trapz(p, x)


        # if RMSD_DTD R2_DTD and Xi2_DTD should be calculated
#        elif accuracy_type == 'DTD':  Need to add the DTD grid and evaluate DTD.

        # calculate the corresponding RMSD value
        sum = 0
        for j in range(len(p_pred_acc)):
            if p_pred_acc[j] < 0:
                p_pred_acc[j] = 0.0
            if p_pred_acc[j] != p_ref_acc[j]:
                sum = sum + (p_pred_acc[j]-p_ref_acc[j])**2

        RMSD = float(np.sqrt(sum/len(p_pred_acc)))
        RMSD_list_Etrans_prod.append(RMSD)

        if calculate_accuracy_measures:
            with open("./RMSD_E_{}.txt".format(accuracy_type), "a") as txt_file:
                txt_file.write(str(ii) + ' ' + 'RMSD_E: ' + str(RMSD) + '\n')

        # calculate the corresponding R2 value
        RSS = sum
        SStot = np.sum((p_ref_acc-np.mean(p_ref_acc))**2)
        R2 = float(1 - RSS/SStot)
        R2_list_Etrans_prod.append(R2)

        if calculate_accuracy_measures:
            with open("./R2_E_{}.txt".format(accuracy_type), "a") as txt_file:
                txt_file.write(str(ii) + ' ' + 'R2_E: ' + str(R2) + '\n')



        
        #--------------------------------------------
        
        if plotting:
            plt.figure()
            plt.plot(x, p/np.trapz(p,x), '-k', label='QCT')
            plt.plot(x, p_pred_acc, '-r', label='STD')
            plt.xlabel(r"$E_{\mathrm{int}}'$" + ' [eV]')
            plt.ylabel('Probability')
            plt.title(r"$P(E_{\mathrm{int}}')$")
            plt.figtext(0.65,0.90, 'Ttrans:' + str(ttrans) + 'K' + ' Tvib:' + str(tvib) + 'K' + ' Trot:' + str(trot) + 'K', fontsize='xx-small', color='blue')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        #----------------------------------------------------
        # Export Individual temperatures if needed

        if external_plotting:
            with open("./external_plotting/STD_E" + "_T" + str(ttrans)+ "_T" + str(tvib) + "_T" + str(trot) + ".txt", "w") as txt_file:
                for j in range(len(x)):
                    txt_file.write(str(x[j]) + ' ' + str(p_pred_acc[j]) + '\n')

            with open("./external_plotting/QCT_E" + "_T" + str(ttrans)+ "_T" + str(tvib) + "_T" + str(trot) + ".txt", "w") as txt_file:
                for j in range(len(x)):
                    txt_file.write(str(x[j]) + ' ' + str(p[j]/np.trapz(p,x)) + '\n')


    ###########################################################################
    # similarly for the product vibrational state distributions
    x, p = np.loadtxt(path + 'pv' + str(ii) + '.txt', unpack=True)
        
    if calculate_accuracy_measures:
        # 1D-interpolation from predicted amplitudes.
        f = interpolate.interp1d(vgridout, temp_v,fill_value="extrapolate")

        #Evaluate at the points where QCT data is available
        p_pred_acc=f(x)
        p_pred_acc=p_pred_acc/np.sum(p)


    # if RMSD_QCT R2_QCT and Xi2_QCT should be calculated
        if accuracy_type == 'QCT':

            # n_max = 2
            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                elif j-2 < 0 or j+2 > (len(p)-2):
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))
                else:
                    p_ref_acc.append(np.mean(p[j-2:j+2+1]))

            p_ref_acc = p_ref_acc/np.sum(p)


        # if RMSD_DTD R2_DTD and Xi2_DTD should be calculated
#        elif accuracy_type == 'DTD':  Need to add the DTD grid and evaluate DTD.

        # calculate the corresponding RMSD and R2 value
        sum = 0
        for j in range(len(p_pred_acc)):
            if p_pred_acc[j] < 0:
                p_pred_acc[j] = 0.0
            if p_pred_acc[j] != p_ref_acc[j]:
                sum = sum + (p_pred_acc[j]-p_ref_acc[j])**2

        RMSD = float(np.sqrt(sum/len(p_pred_acc)))
        RMSD_list_v_prod.append(RMSD)


        # Individual accuracy measures
        if calculate_accuracy_measures:
            with open("./RMSD_v_{}.txt".format(accuracy_type), "a") as txt_file:
                txt_file.write(str(ii) + ' ' + 'RMSD_v: ' + str(RMSD) + '\n')

        RSS = sum
        SStot = np.sum((p_ref_acc-np.mean(p_ref_acc))**2)

        R2 = float(1 - RSS/SStot)
        R2_list_v_prod.append(R2)

        # Individual accuracy measures
        if calculate_accuracy_measures:
            with open("./R2_v_{}.txt".format(accuracy_type), "a") as txt_file:
                txt_file.write(str(ii) + ' ' + 'R2_v: ' + str(R2) + '\n')

        if plotting:
            plt.figure()
            plt.plot(x, p/np.sum(p), '-k', label='QCT')
            plt.plot(x, p_pred_acc, '-r', label='STD')
#            plt.plot(ker_grid, p_ker_ref, '-g')
            plt.xlabel(r"$v'$" + ' [eV]')
            plt.ylabel('Probability')
            plt.title(r"$P(v')$")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        #----------------------------------------------------
        # Export Individual temperatures if needed

        if external_plotting:
            with open("./external_plotting/STD_v" + "_T" + str(ttrans)+ "_T" + str(tvib) + "_T" + str(trot) + ".txt", "w") as txt_file:
                for j in range(len(x)):
                    txt_file.write(str(x[j]) + ' ' + str(p_pred_acc[j]) + '\n')

            with open("./external_plotting/QCT_v" + "_T" + str(ttrans)+ "_T" + str(tvib) + "_T" + str(trot) + ".txt", "w") as txt_file:
                for j in range(len(x)):
                    txt_file.write(str(x[j]) + ' ' + str(p[j]/np.sum(p)) + '\n')



    ###########################################################################
    # similarly for the product rotational state distributions
    x, p = np.loadtxt(path + 'pj' + str(ii) + '.txt', unpack=True)

    if calculate_accuracy_measures:
        # 1D-interpolation from predicted amplitudes.
        f = interpolate.interp1d(jgridout, temp_j,fill_value="extrapolate")

        #Evaluate at the points where QCT data is available
        p_pred_acc=f(x)
        p_pred_acc=p_pred_acc/np.sum(p)


        if accuracy_type == 'QCT':
            
            # n_max = 10
            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                elif j-2 < 0 or j+2 > (len(p)-2):
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))
                elif j-3 < 0 or j+3 > (len(p)-3):
                    p_ref_acc.append(np.mean(p[j-2:j+2+1]))
                elif j-4 < 0 or j+4 > (len(p)-4):
                    p_ref_acc.append(np.mean(p[j-3:j+3+1]))
                elif j-5 < 0 or j+5 > (len(p)-5):
                    p_ref_acc.append(np.mean(p[j-4:j+4+1]))
                elif j-6 < 0 or j+6 > (len(p)-6):
                    p_ref_acc.append(np.mean(p[j-5:j+5+1]))
                elif j-7 < 0 or j+7 > (len(p)-7):
                    p_ref_acc.append(np.mean(p[j-6:j+6+1]))
                elif j-8 < 0 or j+8 > (len(p)-8):
                    p_ref_acc.append(np.mean(p[j-7:j+7+1]))
                elif j-9 < 0 or j+9 > (len(p)-9):
                    p_ref_acc.append(np.mean(p[j-8:j+8+1]))
                elif j-10 < 0 or j+10 > (len(p)-10):
                    p_ref_acc.append(np.mean(p[j-9:j+9+1]))
                else:
                    p_ref_acc.append(np.mean(p[j-10:j+10+1]))

            p_ref_acc = p_ref_acc/np.sum(p)
                    


        # calculate the corresponding RMSD and R2 value

        sum = 0
        for j in range(len(p_pred_acc)):
            if p_pred_acc[j] < 0:
                p_pred_acc[j] = 0.0
            if p_pred_acc[j] != p_ref_acc[j]:
                sum = sum + (p_pred_acc[j]-p_ref_acc[j])**2


        RMSD = float(np.sqrt(sum/len(p_pred_acc)))
        RMSD_list_j_prod.append(RMSD)


        
        # Individual accuracy measures
        if calculate_accuracy_measures:
            with open("./RMSD_j_{}.txt".format(accuracy_type), "a") as txt_file:
                txt_file.write(str(ii) + ' ' + 'RMSD_j: ' + str(RMSD) + '\n')

        RSS = sum
        SStot = np.sum((p_ref_acc-np.mean(p_ref_acc))**2)
        R2 = float(1 - RSS/SStot)
        R2_list_j_prod.append(R2)

        if calculate_accuracy_measures:
            with open("./R2_j_{}.txt".format(accuracy_type), "a") as txt_file:
                txt_file.write(str(ii) + ' ' + 'R2_j: ' + str(R2) + '\n')


        if plotting:
            plt.figure()
            plt.plot(x, p/np.sum(p), '-k', label='QCT')
            plt.plot(x, p_pred_acc, '-r', label='STD')
#            plt.plot(ker_grid, p_ker_ref, '-g')
            plt.xlabel(r"$j'$" + ' [eV]')
            plt.ylabel('Probability')
            plt.title(r"$P(j')$")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        #----------------------------------------------------
        # Export Individual temperatures if needed

        if external_plotting:
            with open("./external_plotting/STD_j" + "_T" + str(ttrans)+ "_T" + str(tvib) + "_T" + str(trot) + ".txt", "w") as txt_file:
                for j in range(len(x)):
                    txt_file.write(str(x[j]) + ' ' + str(p_pred_acc[j]) + '\n')

            with open("./external_plotting/QCT_j" + "_T" + str(ttrans)+ "_T" + str(tvib) + "_T" + str(trot) + ".txt", "w") as txt_file:
                for j in range(len(x)):
                    txt_file.write(str(x[j]) + ' ' + str(p[j]/np.sum(p)) + '\n')

###########################################################################################

# close all pdf files if data and predictions were plotted
if plotting:
    pdf.close()
    plt.close('all')

# calculate the overall accuracy measures
if calculate_accuracy_measures:

    # calculate the overall RMSD value through averaging over all distributions of all data sets
    RMSD_Etrans_prod = np.mean(RMSD_list_Etrans_prod)
    RMSD_v_prod = np.mean(RMSD_list_v_prod)
    RMSD_j_prod = np.mean(RMSD_list_j_prod)
    RMSD_overall = 1/3*(RMSD_Etrans_prod + RMSD_v_prod + RMSD_j_prod)

    # calculate the overall R2 value through averaging over all distributions of all data sets
    R2_Etrans_prod = np.mean(R2_list_Etrans_prod)
    R2_v_prod = np.mean(R2_list_v_prod)
    R2_j_prod = np.mean(R2_list_j_prod)
    R2_overall = 1/3*(R2_Etrans_prod + R2_v_prod + R2_j_prod)




    # save the results in a txt file
    with open("./overall_prediction_accuracy_{}.txt".format(accuracy_type), "w") as txt_file:

        txt_file.write('RMSD_overall: ' + str(RMSD_overall) + ', RMSD_E: ' +
                    str(RMSD_Etrans_prod) + ', RMSD_v: ' + str(RMSD_v_prod) + ', RMSD_j: ' + str(RMSD_j_prod) + '\n')
    
        txt_file.write(', R2_overall: ' + str(R2_overall) + ', R2_E: ' + str(R2_Etrans_prod) + ', R2_v: ' +
                    str(R2_v_prod) + ', R2_j: ' + str(R2_j_prod) + '\n')

        txt_file.write('RMSD_overall: ' + str(RMSD_overall) + ', R2_overall: ' + str(R2_overall) )
