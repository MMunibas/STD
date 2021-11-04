import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import math as ma
from scipy import interpolate

plotting = True
external_plotting = True
path = '../QCT_Data/Initial_Condition_Data/'
performance_evaluation = True  # Must be always true
rate_calculation = True

# specify what type of accuracy should be evaluated (NN or QCT)
accuracy_type = 'NN'



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def shiftsoftplus(x):
    return np.log(np.exp(x)+1)-np.log(2)


def shifttanh(x):
    return np.tanh(x)+1.0


def softplus(x):
    return np.log(np.exp(x)+1)


def relu(x):
    return np.maximum(0, x)


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

###########

num_features = 11
num_outputs = len(Egridout)+len(vgridout)+len(jgridout)

print('Number of features: ' + str(num_features))
print('Number of outputs: ' + str(num_outputs))


# Import data and indices of validation and test set
input_data = np.genfromtxt('../data_preprocessing/Init_Cond/input_for_neural_new.txt', delimiter=',')

data = np.zeros((input_data.shape[0], input_data.shape[1])) # change input_data.shape[1]-1
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i, j] = input_data[i, j]

test_indices = np.genfromtxt('../data_preprocessing/Init_Cond/test_indices.txt')
test_indices = test_indices.astype(int)
#print(str(test_indices)) # print the indices of your data
indices_for_neural = np.genfromtxt('../data_preprocessing/Init_Cond/indices_for_neural.txt').astype(int)

mval_input = np.loadtxt('../training/Coeff_mval_input.txt', delimiter=',', unpack=True)
stdv_input = np.loadtxt('../training/Coeff_stdv_input.txt', delimiter=',', unpack=True)
mval_output = np.loadtxt('../training/Coeff_mval_output.txt', delimiter=',', unpack=True)
stdv_output = np.loadtxt('../training/Coeff_stdv_output.txt', delimiter=',', unpack=True)
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

# Import mean values and stdvs for normalization
# Setup

features = np.zeros(num_features)
non_stand_features = np.zeros(num_features)
output = np.zeros(num_outputs)
aout = np.zeros(num_outputs)
type = ''

if plotting:
    pdf = matplotlib.backends.backend_pdf.PdfPages("predictions_STD.pdf")

if performance_evaluation:
    RMSD_tot_E = []
    RMSD_tot_v = []
    RMSD_tot_j = []
    R2_tot_E = []
    R2_tot_v = []
    R2_tot_j = []
    rate_tot_E = []
    rate2_tot_E = []
    rate_tot_v = []
    rate2_tot_v = []
    rate_tot_j = []
    rate2_tot_j = []



for i in test_indices:

    # Obtain normalized features and output from the dataset
    for j in range(num_features):
        non_stand_features[j] = data[i][j]
        features[j] = (data[i][j] - mval_input[j]) / stdv_input[j]
    for k in range(num_features, num_features + num_outputs):
        output[k-num_features] = data[i][k]

    print('Current data set: ' + str(i) + '\n')
    print('E_trans:' + str(non_stand_features[0]) + ', v_in:' + str(non_stand_features[1]) + ', j_in:' + str(non_stand_features[2]) + ', Evj_in:' + str(non_stand_features[3]) + '\n')


    if non_stand_features[0] != 20.0:

        p_true_E = output[:len(Egridout)]
        p_true_v = output[len(Egridout):len(vgridout)+len(Egridout)]
        p_true_j = output[len(vgridout)+len(Egridout):num_outputs]

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

        aout_unscaled = softplus(np.matmul(outW, a17) + outb)
        aout = aout_unscaled * stdv_output

        p_pred_E = aout[:len(Egridout)]
        p_pred_v = aout[len(Egridout):len(vgridout)+len(Egridout)]
        p_pred_j = aout[len(vgridout)+len(Egridout):num_outputs]

        ###########################################################################################

        # Evaluation:
        p_pred = p_pred_E
        p_true = p_true_E


        E, p = np.loadtxt(path + 'pevj' + str(indices_for_neural[i]+1) + '.dat', unpack=True)
        

    if performance_evaluation:

        if accuracy_type == 'QCT':

            grid = E

            # interpolate STD grid prediction to points where QCT data is available

            f = interpolate.interp1d(Egridout, p_pred,fill_value="extrapolate")#,fill_value="extrapolate")

            #Evaluate at the points where QCT data is available
            p_pred=f(E)

            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                elif j-2 < 0 or j+2 > (len(p)-2):
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))
                elif j-3 < 0 or j+3 > (len(p)-3):
                    p_ref_acc.append(np.mean(p[j-2:j+2+1]))
                else:
                    p_ref_acc.append(np.mean(p[j-3:j+3+1]))

            # divide by the normalization factor obtained by numerical integration of the distributions obtained by QCT simulations
            p_true_perf = p_ref_acc/np.trapz(p, E)
            p_pred_perf = p_pred/np.trapz(p, E)

        elif accuracy_type == 'NN':

            grid = Egridout

            p_true_perf=p_true/np.trapz(p,E)
            p_pred_perf=p_pred/np.trapz(p,E)

            # evaluate compare with the reference amplitudes "NN" and extrapolated STD prediction, for the R2 and RMSD.

        sum = 0
        for j in range(len(grid)):
            if p_pred_perf[j] < 0:
                p_pred_perf[j] = 0.0
            if p_pred_perf[j] != p_true_perf[j] :
                sum = sum + (p_pred_perf[j]-p_true_perf[j])**2

        
        RMSD = float(np.sqrt(sum/len(grid)))
        RMSDE=RMSD
        RMSD_tot_E.append(RMSD)


        RSS = sum
        SStot = np.sum((p_true_perf-np.mean(p_true_perf))**2)
        R2 = float(RSS/SStot)
        R2E=1-R2
        R2_tot_E.append(R2)

        if rate_calculation:
            sigmaNN=np.trapz(p_pred_perf, grid)
            sigmaQCT=np.trapz(p_true_perf, grid)
            bmax=12.0
            bmax=bmax
            bmax=bmax*0.5291772E-8
            kb=1.38064852E-16
            m1=14.003074
            m2=15.99491462
            m3=m2
            mu = m1*(m2+m3)/(m1+m2+m3)*1.66054E-24
            ge=1.0/3.0  # Degeneracy factor
            rateNN=ge*np.sqrt((8.0*kb*300.0)/(ma.pi*mu))*ma.pi*bmax**2*sigmaNN
            rateQCT=ge*np.sqrt((8.0*kb*300.0)/(ma.pi*mu))*ma.pi*bmax**2*sigmaQCT
            rate2NN=ma.pi*bmax**2*sigmaNN
            rate2QCT=ma.pi*bmax**2*sigmaQCT

            ratediff=rateNN-rateQCT
            ratediff2=rate2NN-rate2QCT

            if ratediff <= 0.0:
                ratediff= ratediff*-1.0
            else:
                ratediff=ratediff

            if ratediff2 <= 0.0:
                ratediff2= ratediff2*-1.0
            else:
                ratediff2=ratediff2

            rate_tot_E.append(ratediff)
            rate2_tot_E.append(ratediff2)

        if plotting:
            plt.figure()
            plt.plot(E, p/np.trapz(p,E), '-k', label='QCT')
            plt.plot(grid, p_pred_perf, '-r', label='NN')
            plt.plot(grid, p_true_perf, color='g', marker='o', linestyle='', label='Grid')
            plt.xlabel('$\it{E^{\prime}_{int}}$ [eV]')
            plt.ylabel('Probability')
            plt.title('Internal energy distribution of product')

            plt.figtext(0.80, 0.65, 'RMSD: ' + str("{:.5f}".format(RMSD)), fontsize='small')
            plt.figtext(0.80, 0.60, 'R2: ' + str("{:.5f}".format(1-R2)), fontsize='small')

            plt.figtext(0.55, 0.85, '$\it{E_{trans}}$: ' + str(non_stand_features[0]) + ' eV' + ', $\it{v}$: ' + str(int(non_stand_features[1])) + ', $\it{j}$: ' + str(int(non_stand_features[2])), horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="blue")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()

            plt.close()

        if external_plotting:

            with open("./external_plotting/NN_pe" + str(i) + "_NN.txt", "w") as txt_file:
                for j in range(len(grid)):
                    txt_file.write(str(grid[j]) + ' ' + str(p_pred_perf[j]) + '\n')

            with open("./external_plotting/NN_pe" + str(i) + "_InputPoints.txt", "w") as txt_file:
                for j in range(len(grid)):
                    txt_file.write(str(grid[j]) + ' ' + str(p_true_perf[j]) + '\n')



    ##########################################################################################################
        p_pred = p_pred_v
        p_true = p_true_v

        E, p = np.loadtxt(path + 'pv' + str(indices_for_neural[i]+1) + '.dat', unpack=True)

    if performance_evaluation:

        if accuracy_type == 'QCT':

            grid = E

            # interpolate STD grid prediction to points where QCT data is available

            f = interpolate.interp1d(vgridout, p_pred,fill_value="extrapolate")#,fill_value="extrapolate")

            #Evaluate at the points where QCT data is available
            p_pred=f(E)

            p_ref_acc = []
            for j in range(len(p)):
                if j-1 < 0 or j+1 > (len(p)-1):
                    p_ref_acc.append(p[j])
                else:
                    p_ref_acc.append(np.mean(p[j-1:j+1+1]))

            # divide by the normalization factor obtained by numerical integration of the distributions obtained by QCT simulations
            p_true_perf = p_ref_acc/np.sum(p)
            p_pred_perf = p_pred/np.sum(p)

        elif accuracy_type == 'NN':

            grid = vgridout

            p_true_perf=p_true/np.sum(p)
            p_pred_perf=p_pred/np.sum(p)

            # evaluate compare with the reference amplitudes "NN" and extrapolated STD prediction, for the R2 and RMSD.

        sum = 0
        for j in range(len(grid)):
            if p_pred_perf[j] < 0:
                p_pred_perf[j] = 0.0
            if p_pred_perf[j] != p_true_perf[j] :
                sum = sum + (p_pred_perf[j]-p_true_perf[j])**2

        RMSD = float(np.sqrt(sum/len(grid)))
        RMSDv=RMSD
        RMSD_tot_v.append(RMSD)

        RSS = sum
        SStot = np.sum((p_true_perf-np.mean(p_true_perf))**2)
        R2 = float(RSS/SStot)
        R2v=1-R2
        R2_tot_v.append(R2)


        if rate_calculation:
            sigmaNN=np.sum(p_pred_perf)
            sigmaQCT=np.sum(p_true_perf)
            bmax=12.0
            bmax=bmax
            bmax=bmax*0.5291772E-8
            kb=1.38064852E-16
            m1=14.003074
            m2=15.99491462
            m3=m2
            mu = m1*(m2+m3)/(m1+m2+m3)*1.66054E-24
            ge=1.0/3.0  # Degeneracy factor
            rateNN=ge*np.sqrt((8.0*kb*300.0)/(ma.pi*mu))*ma.pi*bmax**2*sigmaNN
            rateQCT=ge*np.sqrt((8.0*kb*300.0)/(ma.pi*mu))*ma.pi*bmax**2*sigmaQCT
            rate2NN=ma.pi*bmax**2*sigmaNN
            rate2QCT=ma.pi*bmax**2*sigmaQCT

            ratediff=rateNN-rateQCT
            ratediff2=rate2NN-rate2QCT

            if ratediff <= 0.0:
                ratediff= ratediff*-1.0
            else:
                ratediff=ratediff

            if ratediff2 <= 0.0:
                ratediff2= ratediff2*-1.0
            else:
                ratediff2=ratediff2

            rate_tot_v.append(ratediff)
            rate2_tot_v.append(ratediff2)


        if plotting:
            plt.figure()
            plt.plot(E, p/np.sum(p), '-k', label='QCT')
            plt.plot(grid, p_pred_perf, '-r', label='NN')
            plt.plot(grid, p_true_perf, color='g', marker='o', linestyle='', label='Grid')
            plt.xlabel('$\it{v^{\prime}}$')
            plt.ylabel('Probability')
            plt.title('Vibrational distribution of product')
            plt.figtext(0.80, 0.65, 'RMSD: ' + str("{:.5f}".format(RMSD)), fontsize='small')
            plt.figtext(0.80, 0.60, 'R2: ' + str("{:.5f}".format(1-R2)), fontsize='small')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()

            plt.close()

        if external_plotting:


            with open("./external_plotting/NN_pv" + str(i) + "_NN.txt", "w") as txt_file:
                for j in range(len(grid)):
                    txt_file.write(str(grid[j]) + ' ' + str(p_pred_perf[j]) + '\n')

            with open("./external_plotting/NN_pv" + str(i) + "_InputPoints.txt", "w") as txt_file:
                for j in range(len(grid)):
                    txt_file.write(str(grid[j]) + ' ' + str(p_true_perf[j]) + '\n')


    #######################################################################################################

        p_pred = p_pred_j
        p_true = p_true_j

        E, p = np.loadtxt(path + 'pj' + str(indices_for_neural[i]+1) + '.dat', unpack=True)



    if performance_evaluation:

        if accuracy_type == 'QCT':

            grid = E

            # interpolate STD grid prediction to points where QCT data is available

            f = interpolate.interp1d(jgridout, p_pred, fill_value="extrapolate")#,fill_value="extrapolate")

            #Evaluate at the points where QCT data is available
            p_pred=f(E)

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
                else:
                    p_ref_acc.append(np.mean(p[j-7:j+7+1]))


            # divide by the normalization factor obtained by numerical integration of the distributions obtained by QCT simulations
            p_true_perf = p_ref_acc/np.sum(p)
            p_pred_perf = p_pred/np.sum(p)

        elif accuracy_type == 'NN':

            grid = jgridout
            p_true_perf=p_true/np.sum(p)
            p_pred_perf=p_pred/np.sum(p)

            # evaluate compare with the reference amplitudes "NN" and extrapolated STD prediction, for the R2 and RMSD.

        sum = 0
        for j in range(len(grid)):
            if p_pred_perf[j] < 0:
                p_pred_perf[j] = 0.0
            if p_pred_perf[j] != p_true_perf[j] and p_true_perf[j] != 0.0:
                sum = sum + (p_pred_perf[j]-p_true_perf[j])**2

        RMSD = float(np.sqrt(sum/len(grid)))
        RMSDj=RMSD
        RMSD_tot_j.append(RMSD)

        RSS = sum
        SStot = np.sum((p_true_perf-np.mean(p_true_perf))**2)
        R2 = float(RSS/SStot)
        R2j=1-R2
        R2_tot_j.append(R2)


        if rate_calculation:
            sigmaNN=np.sum(p_pred_perf)
            sigmaQCT=np.sum(p_true_perf)
            bmax=12.0
            bmax=bmax
            bmax=bmax*0.5291772E-8
            kb=1.38064852E-16
            m1=14.003074
            m2=15.99491462
            m3=m2
            mu = m1*(m2+m3)/(m1+m2+m3)*1.66054E-24
            ge=1.0/3.0  # Degeneracy factor
            rateNN=ge*np.sqrt((8.0*kb*300.0)/(ma.pi*mu))*ma.pi*bmax**2*sigmaNN
            rateQCT=ge*np.sqrt((8.0*kb*300.0)/(ma.pi*mu))*ma.pi*bmax**2*sigmaQCT
            rate2NN=ma.pi*bmax**2*sigmaNN
            rate2QCT=ma.pi*bmax**2*sigmaQCT

            ratediff=rateNN-rateQCT
            ratediff2=rate2NN-rate2QCT

            if ratediff <= 0.0:
                ratediff= ratediff*-1.0
            else:
                ratediff=ratediff

            if ratediff2 <= 0.0:
                ratediff2= ratediff2*-1.0
            else:
                ratediff2=ratediff2

            rate_tot_j.append(ratediff)
            rate2_tot_j.append(ratediff2)


        if plotting:
            plt.figure()
            plt.plot(E, p/np.sum(p), '-k', label='QCT')
            plt.plot(grid, p_pred_perf, '-r', label='NN')
            plt.plot(grid, p_true_perf, color='g', marker='o', linestyle='', label='Grid')
            plt.xlabel('$\it{j^{\prime}}$')
            plt.ylabel('Probability')
            plt.title('Rotational distribution of product')
            plt.figtext(0.80, 0.65, 'RMSD: ' + str("{:.5f}".format(RMSD)), fontsize='small')
            plt.figtext(0.80, 0.60, 'R2: ' + str("{:.5f}".format(1-R2)), fontsize='small')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()

            plt.close()

        if external_plotting:


            with open("./external_plotting/NN_pj" + str(i) + "_NN.txt", "w") as txt_file:
                for j in range(len(grid)):
                    txt_file.write(str(grid[j]) + ' ' + str(p_pred_perf[j]) + '\n')

            with open("./external_plotting/NN_pj" + str(i) + "_InputPoints.txt", "w") as txt_file:
                for j in range(len(grid)):
                    txt_file.write(str(grid[j]) + ' ' + str(p_true_perf[j]) + '\n')


        ####################################################################################################################
        # Analisis

        if external_plotting:
            R2over=1/3*(R2E+R2v+R2j)
            RMSDover=1/3*(RMSDE+RMSDv+RMSDj)
            with open("./external_plotting/NN" + str(i) + "_general.txt", "w") as txt_file:
                txt_file.write(str(i) + ' ' + 'E_trans:' + str(non_stand_features[0]) + ', v_in:' + str(
                    non_stand_features[1]) + ', j_in:' + str(non_stand_features[2]) + ', R2:' + str(R2over) + ', RMSD:' + str(RMSDover) +  '\n')


if plotting:
    pdf.close()
    plt.close('all')

if performance_evaluation:
    with open("./performance_RMSD.txt", "w") as txt_file:
        RMSD_unsorted = 1/3*np.add(np.add(RMSD_tot_E, RMSD_tot_v), RMSD_tot_j)
        RMSD_each = np.sort(RMSD_unsorted)

        print('Best: ')
        x = np.where(RMSD_unsorted == RMSD_each[0])
        txt_file.write('Best Dataset: ' + str(x)+'\n')
        txt_file.write('Best: ' + str(RMSD_each[0])+'\n')

        print('Worst: ')
        x = np.where(RMSD_unsorted == RMSD_each[len(RMSD_each)-1])
        txt_file.write('Worst Dataset: ' + str(x)+'\n')
        txt_file.write('Worst: ' + str(RMSD_each[len(RMSD_each)-1])+'\n')

        print('Middle: ')
        mean = np.mean(RMSD_each)
        x = find_nearest(RMSD_unsorted, mean)
        txt_file.write('Representative Dataset: ' + str(x)+'\n')
        txt_file.write('Representative: ' + str(RMSD_unsorted[x])+'\n')

    with open("./performance_R2.txt", "w") as txt_file:
        R2_unsorted = 1/3*np.add(np.add(R2_tot_E, R2_tot_v), R2_tot_j)
        R2_each = np.sort(R2_unsorted)

        print('Best: ')
        x = np.where(R2_unsorted == R2_each[0])
        print(x)
        txt_file.write('Best Dataset: ' + str(x)+'\n')
        txt_file.write('Best: ' + str(R2_each[0])+'\n')

        print('Worst: ')
        x = np.where(R2_unsorted == R2_each[len(R2_each)-1])
        print(x)
        txt_file.write('Worst Dataset: ' + str(x)+'\n')
        txt_file.write('Worst: ' + str(R2_each[len(R2_each)-1])+'\n')

        print('Middle: ')
        mean = np.mean(R2_each)
        x = find_nearest(R2_unsorted, mean)
        txt_file.write('Representative Dataset : ' + str(x) + '\n')
        txt_file.write('Representative: ' + str(R2_unsorted[x])+'\n')



    RMSD_E = np.mean(RMSD_tot_E)
    RMSD_v = np.mean(RMSD_tot_v)
    RMSD_j = np.mean(RMSD_tot_j)
    RMSD_overall = 1/3*(RMSD_E + RMSD_v + RMSD_j)

    R2_E = 1-np.mean(R2_tot_E)
    R2_v = 1-np.mean(R2_tot_v)
    R2_j = 1-np.mean(R2_tot_j)
    R2_overall = 1/3*(R2_E + R2_v + R2_j)


if rate_calculation:
    RATE_E = np.mean(rate_tot_E)
    RATE_v = np.mean(rate_tot_v)
    RATE_j = np.mean(rate_tot_j)
    RATE_overall = 1/3*(RATE_E + RATE_v + RATE_j)

    RATE2_E = np.mean(rate2_tot_E)
    RATE2_v = np.mean(rate2_tot_v)
    RATE2_j = np.mean(rate2_tot_j)
    RATE2_overall = 1/3*(RATE2_E + RATE2_v + RATE2_j)

if performance_evaluation:
    with open("./performance" + "_" + str(accuracy_type) + ".txt" , "w") as txt_file:
        txt_file.write('RMSD_overall: ' + str(RMSD_overall) + ', RMSD_E: ' +
                       str(RMSD_E) + ', RMSD_v: ' + str(RMSD_v) + ', RMSD_j: ' + str(RMSD_j) + '\n')

        txt_file.write('R2_overall: ' + str(R2_overall) + ', R2_E: ' + str(R2_E) + ', R2_v: ' +
                       str(R2_v) + ', R2_j: ' + str(R2_j) + '\n')

if rate_calculation:
    with open("./performance" + "_" + "rates" + "_" + str(accuracy_type) + ".txt" , "w") as txt_file:
    
        txt_file.write('RATE_overall: ' + str(RATE_overall) + ', RATE_E: ' + str(RATE_E) + ', RATE_v: ' +
                       str(RATE_v) + ', RATE_j: ' + str(RATE_j) + '\n' )

        txt_file.write('RATE2_overall: ' + str(RATE2_overall) + ', RATE2_E: ' + str(RATE2_E) + ', RATE2_v: ' +
                       str(RATE2_v) + ', RATE2_j: ' + str(RATE2_j) )


