import numpy as np
import pandas as pd
import os
from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models, get_prob_mat
from kde_ebm.plotting import mixture_model_grid, mcmc_uncert_mat, mcmc_trace, stage_histogram
from kde_ebm.mcmc import mcmc, parallel_bootstrap, bootstrap_ebm, bootstrap_ebm_fixedMM, bootstrap_ebm_return_mixtures
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import seaborn as sn
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import mannwhitneyu as mwu
import itertools
from datetime import datetime



def preliminaries(fname_save,wd='/Users/noxtoby/Documents/Research/EuroPOND/Projects/WP9-Modelling',csv_file='data/le.csv',events_set='data-driven'):
    """
    EBM prep.
    Returns a cleaned pandas DataFrame.
    Author: Neil P Oxtoby, UCL, September 2018
    """
    event_markers, event_markers_labels, edict = default_events(events_set)
    if os.path.isfile(fname_save):
        print('   ...Save file detected ({0}). Prep work done. Good on ya.'.format(fname_save))
        df = pd.read_csv(fname_save,low_memory=False)
        return df, event_markers, event_markers_labels
    else:
        print('   ...Executing preliminaries() function.')
    
    #* Load data
    df = pd.read_csv(os.path.join(wd,csv_file),low_memory=False)
    
    #* Make patient flag numeric
    df.loc[:,'DX'] = pd.to_numeric(df['id'])
    
    #df.to_csv(fname_save,index=False)
    
    return df, event_markers, event_markers_labels


#* Detrend (remove gradients)
def detrend(df, outcome_label, 
            normals_bool_array, # = df['MC']==0, 
            covariates_labels, #  = ['HAD_anx','HAD_dep','Education']
            verbose_flag=False):
    '''detrend(df, outcome_label, normals_bool_array, covariates_labels, mri_flag):
    Removes normal trends due to covariates, using a particular GLM
    '''
    
    glm_formula = 'y ~ ' + ' + '.join(covariates_labels)
    
    print('detrend(): {0}'.format(glm_formula))
    
    covariates_labels = list(np.unique(covariates_labels))
    #* lambda functions for centering
    mean_centred_categorical = lambda x: ( x - np.nanmean(np.unique(x)) ) / ( (np.nanmax(x) - np.nanmin(x))/len(np.unique(x)) )
    mean_centred_continuous  = lambda x: ( x - np.nanmean(x) ) 
    #* Centre the covariates
    df_centred_covariates = df[covariates_labels].copy()
    for col in covariates_labels:
        x = df_centred_covariates[col].copy()
        if len(np.unique(x)) > 5:
            df_centred_covariates[col] = mean_centred_continuous(x)
        else:
            df_centred_covariates[col] = mean_centred_categorical(x)
    #* Fit a GLM using statsmodels
    df_centred_covariates['y'] = df[outcome_label]
    mod = smf.ols(formula=glm_formula, data=df_centred_covariates.loc[normals_bool_array])
    res = mod.fit()
    #* Detrending (leave intercept unchanged):
    correction = res.predict(df_centred_covariates) - res.params[0]
    y_detrended = df[outcome_label] - correction
    
    return y_detrended, df_centred_covariates, correction, glm_formula, mod, res


def detrend_data(df,bl,fname_save=None,verbose=False,short=False,detrend_bool=True,covariates_labels=['Education'],events_set='data-driven'):
    """
    Detrend disease data using GLM
    
    "Regress-out" trends in normals using a GLM with no interactions:
      `y ~ covariate1 + covariate2`
    
    1. Longlist of candidate events
    2. Detrend (adjust for confounding covariates)
      - HAD_anx, HAD_dep, Education
    
    Also known as Covariate Adjustment / Regressing Out / et al.
    """
    event_markers, event_markers_labels, edict = default_events(events_set=events_set)
    
    if os.path.isfile(fname_save):
        print('   ...Save file detected ({0}). Prep work done. Good on ya.'.format(fname_save))
        df_ = pd.read_csv(fname_save,low_memory=False)
        markers_all_detrended = [e+'-detrended' for e in event_markers]
        return df_, markers_all_detrended
    else:
        df_ = df.copy()
        print('   ...Executing detrend_data() function.')

    #*** Loop through and detrend ***
    markers_all = event_markers
    markers_all_detrended = [e+'-detrended' for e in markers_all] # labels for covariate adjustment
    markers_all_detrended_model_fit = []
    #* Use baseline data to detrend longitudinal data
    n_bl = df_['normals_EBM'] & bl
    for k in range(len(markers_all)):
        #* Labels
        e = markers_all[k]
        e_detrended = markers_all_detrended[k]
        if verbose:
            print('Detrending {0} for covariates'.format(e))
        
        #* Save
        if detrend_bool:
            #*** detrend()
            y_detrended, df_centred_covariates, correction, glm_formula, y_model, y_model_fit = detrend(
                df_,
                outcome_label=e,
                normals_bool_array=n_bl,
                verbose_flag=verbose,
                covariates_labels=covariates_labels)
            df_[e_detrended] = y_detrended
            markers_all_detrended_model_fit.append(y_model_fit)
        else:
            df_[e_detrended] = df_[e]

    #df_.to_csv(fname_save,index=False)

    return df_, markers_all_detrended


def default_events(events_set='data-driven'):
    """
    EBM prep.
    Returns a list of strings containing biomarker event names.
    Author: Neil P Oxtoby, UCL, September 2018
    """
    if events_set=='updated_LE':
        #* Events
        event_markers_dict = {
            'deg_0':   'Degree 0',
            'cc_0':    'Clustering 0',
            'ne_0':    'Efficiency 0',
            'deg_1':   'Degree 1',
            'cc_1':    'Clustering 1',
            'ne_1':    'Efficiency 1',
            'deg_2':   'Degree 2',
            'cc_2':    'Clustering 2',
            'ne_2':    'Efficiency 2',
            'deg_3':   'Degree 3',
            'cc_3':    'Clustering 3',
            'ne_3':    'Efficiency 3',
            'deg_4':   'Degree 4',
            'cc_4':    'Clustering 4',
            'ne_4':    'Efficiency 4',
            'deg_5':   'Degree 5',
            'cc_5':    'Clustering 5',
            'ne_5':    'Efficiency 5'   }
            # 'deg_6':   'Degree 6',
            # 'cc_6':    'Clustering 6',
            # 'ne_6':    'Efficiency 6',
            # 'deg_7':   'Degree 7',
            # 'cc_7':    'Clustering 7',
            # 'ne_7':    'Efficiency 7'
        
        event_markers = ['deg_0','cc_0','ne_0',
                         'deg_1','cc_1','ne_1',
                         'deg_2','cc_2','ne_2',
                         'deg_3','cc_3','ne_3',
                         'deg_4','cc_4','ne_4',
                         'deg_5','cc_5','ne_5'] 
                         # 'deg_6','cc_6','ne_6'
                         # 'deg_7','cc_7','ne_7'
    elif events_set=='data-driven':
        #* Events
        event_markers_dict = {
            'deg1':   'Degree 1',
            'cc1':    'Clustering 1',
            'ne1':    'Efficiency 1',
            'deg2':   'Degree 2',
            'cc2':    'Clustering 2',
            'ne2':    'Efficiency 2',
            'deg3':   'Degree 3',
            'cc3':    'Clustering 3',
            'ne3':    'Efficiency 3',
            'deg4':   'Degree 4',
            'cc4':    'Clustering 4',
            'ne4':    'Efficiency 4',
            'deg5':   'Degree 5',
            'cc5':    'Clustering 5',
            'ne5':    'Efficiency 5',
            'deg6':   'Degree 6',
            'cc6':    'Clustering 6',
            'ne6':    'Efficiency 6'
        }
        event_markers = ['deg1','cc1','ne1',
                         'deg2','cc2','ne2',
                         'deg3','cc3','ne3',
                         'deg4','cc4','ne4',
                         'deg5','cc5','ne5',
                         'deg6','cc6','ne6']
    elif events_set=='lobes':
        event_markers_dict = {
            'deg_1_lob':  'Degree I',
            'cc_1_lob':   'Clustering I',
            'ne_1_lob':   'Efficiency I',
            'deg_2_lob':  'Degree II',
            'cc_2_lob':   'Clustering II',
            'ne_2_lob':   'Efficiency II',
            'deg_3_lob':  'Degree III',
            'cc_3_lob':   'Clustering III',
            'ne_3_lob':   'Efficiency III',
            'deg_4_lob':  'Degree IV',
            'cc_4_lob':   'Clustering IV',
            'ne_4_lob':   'Efficiency IV',
            'deg_5_lob':  'Degree V',
            'cc_5_lob':   'Clustering V',
            'ne_5_lob':   'Efficiency V',
            'deg_6_lob':  'Degree VI',
            'cc_6_lob':   'Clustering VI',
            'ne_6_lob':   'Efficiency VI',
            'deg_7_lob':  'Degree VII',
            'cc_7_lob':   'Clustering VII',
            'ne_7_lob':   'Efficiency VII',
            'deg_8_lob':  'Degree VIII',
            'cc_8_lob':   'Clustering VIII',
            'ne_8_lob':   'Efficiency VIII',
            'deg_9_lob':  'Degree IX',
            'cc_9_lob':   'Clustering IX',
            'ne_9_lob':   'Efficiency IX',
            'deg_10_lob': 'Degree X',
            'cc_10_lob':  'Clustering X',
            'ne_10_lob':  'Efficiency X',
            'deg_11_lob': 'Degree XI',
            'cc_11_lob':  'Clustering XI',
            'ne_11_lob':  'Efficiency XI',
            'deg_12_lob': 'Degree XII',
            'cc_12_lob':  'Clustering XII',
            'ne_12_lob':  'Efficiency XII'
        }
        event_markers = [
            'deg_1_lob','cc_1_lob','ne_1_lob',
            'deg_2_lob','cc_2_lob','ne_2_lob',
            'deg_3_lob','cc_3_lob','ne_3_lob',
            'deg_4_lob','cc_4_lob','ne_4_lob',
            'deg_5_lob','cc_5_lob','ne_5_lob',
            'deg_6_lob','cc_6_lob','ne_6_lob',
            'deg_7_lob','cc_7_lob','ne_7_lob',
            'deg_8_lob','cc_8_lob','ne_8_lob',
            'deg_9_lob','cc_9_lob','ne_9_lob',
            'deg_10_lob','cc_10_lob','ne_10_lob',
            'deg_11_lob','cc_11_lob','ne_11_lob',
            'deg_12_lob','cc_12_lob','ne_12_lob'
        ]
    elif events_set=='updated_FG':
        #* Events
        event_markers_dict = {
            'deg_0':   'Degree 0',
            'cc_0':    'Clustering 0',
            'ne_0':    'Efficiency 0',
            'deg_1':   'Degree 1',
            'cc_1':    'Clustering 1',
            'ne_1':    'Efficiency 1',
            'deg_2':   'Degree 2',
            'cc_2':    'Clustering 2',
            'ne_2':    'Efficiency 2',
            'deg_3':   'Degree 3',
            'cc_3':    'Clustering 3',
            'ne_3':    'Efficiency 3',
            'deg_4':   'Degree 4',
            'cc_4':    'Clustering 4',
            'ne_4':    'Efficiency 4',
            'deg_5':   'Degree 5',
            'cc_5':    'Clustering 5',
            'ne_5':    'Efficiency 5',
            'deg_6':   'Degree 6',
            'cc_6':    'Clustering 6',
            'ne_6':    'Efficiency 6',
            'deg_7':   'Degree 7',
            'cc_7':    'Clustering 7',
            'ne_7':    'Efficiency 7'
        }
        event_markers = ['deg_0','cc_0','ne_0',
                         'deg_1','cc_1','ne_1',
                         'deg_2','cc_2','ne_2',
                         'deg_3','cc_3','ne_3',
                         'deg_4','cc_4','ne_4',
                         'deg_5','cc_5','ne_5', 
                         'deg_6','cc_6','ne_6',
                         'deg_7','cc_7','ne_7']

    elif events_set=='updated_LE_b':
        #* Events
        event_markers_dict = {
            'deg_0':   'Degree 0',
            'ne_0':    'Efficiency 0',
            'deg_1':   'Degree 1',
            'ne_1':    'Efficiency 1',
            'deg_2':   'Degree 2',
            'ne_2':    'Efficiency 2',
            'deg_3':   'Degree 3',
            'ne_3':    'Efficiency 3',
            'deg_4':   'Degree 4',
            'ne_4':    'Efficiency 4'
        }
        event_markers = ['deg_0','ne_0',
                         'deg_1','ne_1',
                         'deg_2','ne_2',
                         'deg_3','ne_3',
                         'deg_4','ne_4']

    elif events_set=='updated_FG_b':
        #* Events
        event_markers_dict = {
            'deg_0':   'Degree 0',
            'ne_0':    'Efficiency 0',
            'deg_1':   'Degree 1',
            'ne_1':    'Efficiency 1',
            'deg_3':   'Degree 3',
            'ne_3':    'Efficiency 3',
            'deg_4':   'Degree 5',
            'ne_4':    'Efficiency 5'
        }
        event_markers = ['deg_0','ne_0',
                         'deg_1','ne_1',
                         'deg_3','ne_3',
                         'deg_4','ne_4']

    elif events_set=='global':
        #* Events
        event_markers_dict = {
            'Deg':   'Degree',
            'Cc':    'Clustering Coef',
            'Ne':    'Nodal Efficiency' }
        event_markers = ['Deg','Cc','Ne']
    
    elif events_set=='lobes_2020':
        event_markers_dict = {
            'deg_1_lob':  'Degree I',
            'cc_1_lob':   'Clustering I',
            'ne_1_lob':   'Efficiency I',
            'deg_2_lob':  'Degree II',
            'cc_2_lob':   'Clustering II',
            'ne_2_lob':   'Efficiency II',
            'deg_3_lob':  'Degree III',
            'cc_3_lob':   'Clustering III',
            'ne_3_lob':   'Efficiency III',
            'deg_4_lob':  'Degree IV',
            'cc_4_lob':   'Clustering IV',
            'ne_4_lob':   'Efficiency IV',
            'deg_5_lob':  'Degree V',
            'cc_5_lob':   'Clustering V',
            'ne_5_lob':   'Efficiency V',
            'deg_7_lob':  'Degree VII',
            'cc_7_lob':   'Clustering VII',
            'ne_7_lob':   'Efficiency VII',
            'deg_8_lob':  'Degree VIII',
            'cc_8_lob':   'Clustering VIII',
            'ne_8_lob':   'Efficiency VIII',
            'deg_10_lob': 'Degree X',
            'cc_10_lob':  'Clustering X',
            'ne_10_lob':  'Efficiency X',
            'deg_11_lob': 'Degree XI',
            'cc_11_lob':  'Clustering XI',
            'ne_11_lob':  'Efficiency XI',
            'deg_12_lob': 'Degree XII',
            'cc_12_lob':  'Clustering XII',
            'ne_12_lob':  'Efficiency XII',
            'deg_13_lob': 'Degree XIII',
            'cc_13_lob':  'Clustering XIII',
            'ne_13_lob':  'Efficiency XIII',
            'deg_14_lob': 'Degree XIV',
            'cc_14_lob':  'Clustering XIV',
            'ne_14_lob':  'Efficiency XIV',
        }
        event_markers = [
            'deg_1_lob','cc_1_lob','ne_1_lob',
            'deg_2_lob','cc_2_lob','ne_2_lob',
            'deg_3_lob','cc_3_lob','ne_3_lob',
            'deg_4_lob','cc_4_lob','ne_4_lob',
            'deg_5_lob','cc_5_lob','ne_5_lob',
            'deg_7_lob','cc_7_lob','ne_7_lob',
            'deg_8_lob','cc_8_lob','ne_8_lob',
            'deg_10_lob','cc_10_lob','ne_10_lob',
            'deg_11_lob','cc_11_lob','ne_11_lob',
            'deg_12_lob','cc_12_lob','ne_12_lob',
            'deg_13_lob','cc_13_lob','ne_13_lob',
            'deg_14_lob','cc_14_lob','ne_14_lob',
        ]
    elif events_set=='lobes_2020_b':
        event_markers_dict = {
            'deg_1_lob':  'Deg L_Frontal_lobe',
            'ne_1_lob':   'Ne L_Frontal_lobe',

            'deg_2_lob':  'Deg L_Parietal_lobe',
            'ne_2_lob':   'Ne L_Parietal_lobe',

            'deg_3_lob':  'Deg L_Occipital_lobe',
            'ne_3_lob':   'Ne L_Occipital_lobe',

            'deg_4_lob':  'Deg L_Temporal_lobe',
            'ne_4_lob':   'Ne L_Temporal_lobe',

            'deg_5_lob':  'Deg L_Cingulate_lobe',
            'ne_5_lob':   'Ne L_Cingulate_lobe',

            'deg_7_lob':  'Deg L_Subcortical',
            'ne_7_lob':   'Ne L_Subcortical',

            'deg_8_lob':  'Deg R_Subcortical',
            'ne_8_lob':   'Ne R_Subcortical',

            'deg_10_lob': 'Deg R_Cingulate_lobe',
            'ne_10_lob':  'Ne R_Cingulate_lobe',

            'deg_11_lob': 'Deg R_Temporal_lobe',
            'ne_11_lob':  'Ne R_Temporal_lobe',

            'deg_12_lob': 'Deg R_Occipital_lobe',
            'ne_12_lob':  'Ne R_Occipital_lobe',

            'deg_13_lob': 'Deg R_Parietal_lobe',
            'ne_13_lob':  'Ne R_Parietal_lobe',

            'deg_14_lob': 'Deg R_Frontal_lobe',
            'ne_14_lob':  'Ne R_Frontal_lobe',
        }
        event_markers = [
            'deg_1_lob','ne_1_lob',
            'deg_2_lob','ne_2_lob',
            'deg_3_lob','ne_3_lob',
            'deg_4_lob','ne_4_lob',
            'deg_5_lob','ne_5_lob',
            'deg_7_lob','ne_7_lob',
            'deg_8_lob','ne_8_lob',
            'deg_10_lob','ne_10_lob',
            'deg_11_lob','ne_11_lob',
            'deg_12_lob','ne_12_lob',
            'deg_13_lob','ne_13_lob',
            'deg_14_lob','ne_14_lob',
        ]
    elif events_set=='lobes_2020_6yo_b': # 12 LOBES (based on anatomy) in total. Insula was merged with Cingulate
                                       # and R/L cerebellum was excluded as these lobes inlcuded only 1 regions 
                                       # thus, not so meaningful for network-based analysis.
        event_markers_dict = {
            'deg_1_lob':  'Deg L_Frontal_lobe',
            'ne_1_lob':   'Ne L_Frontal_lobe',

            'deg_2_lob':  'Deg L_Parietal_lobe',
            'ne_2_lob':   'Ne L_Parietal_lobe',

            'deg_3_lob':  'Deg L_Occipital_lobe',
            'ne_3_lob':   'Ne L_Occipital_lobe',

            'deg_4_lob':  'Deg L_Temporal_lobe',
            'ne_4_lob':   'Ne L_Temporal_lobe',

            'deg_5_lob':  'Deg L_Cingulate_lobe',
            'ne_5_lob':   'Ne L_Cingulate_lobe',

            'deg_7_lob':  'Deg L_Subcortical',
            'ne_7_lob':   'Ne L_Subcortical',

            'deg_8_lob':  'Deg R_Subcortical',
            'ne_8_lob':   'Ne R_Subcortical',

            'deg_10_lob': 'Deg R_Cingulate_lobe',
            'ne_10_lob':  'Ne R_Cingulate_lobe',

            'deg_11_lob': 'Deg R_Temporal_lobe',
            'ne_11_lob':  'Ne R_Temporal_lobe',

            'deg_12_lob': 'Deg R_Occipital_lobe',
            'ne_12_lob':  'Ne R_Occipital_lobe',

            'deg_13_lob': 'Deg R_Parietal_lobe',
            'ne_13_lob':  'Ne R_Parietal_lobe',

            'deg_14_lob': 'Deg R_Frontal_lobe',
            'ne_14_lob':  'Ne R_Frontal_lobe',
        }
        event_markers = [
            'deg_1_lob','ne_1_lob',
            'deg_2_lob','ne_2_lob',
            'deg_3_lob','ne_3_lob',
            'deg_4_lob','ne_4_lob',
            'deg_5_lob','ne_5_lob',
            'deg_7_lob','ne_7_lob',
            'deg_8_lob','ne_8_lob',
            'deg_10_lob','ne_10_lob',
            'deg_11_lob','ne_11_lob',
            'deg_12_lob','ne_12_lob',
            'deg_13_lob','ne_13_lob',
            'deg_14_lob','ne_14_lob',
        ]
    elif events_set=='lobes_2020_6yo_c': # 12 LOBES (based on anatomy) in total. Insula was merged with Cingulate
                                       # and R/L cerebellum was excluded as these lobes inlcuded only 1 regions 
                                       # thus, not so meaningful for network-based analysis.
        event_markers_dict = {
            'deg_1_lob':  'Deg L_Frontal_lobe',
            'deg_2_lob':  'Deg L_Parietal_lobe',
            'deg_3_lob':  'Deg L_Occipital_lobe',
            'deg_4_lob':  'Deg L_Temporal_lobe',
            'deg_5_lob':  'Deg L_Cingulate_lobe',
            'deg_7_lob':  'Deg L_Subcortical',
            'deg_8_lob':  'Deg R_Subcortical',
            'deg_10_lob': 'Deg R_Cingulate_lobe',
            'deg_11_lob': 'Deg R_Temporal_lobe',
            'deg_12_lob': 'Deg R_Occipital_lobe',
            'deg_13_lob': 'Deg R_Parietal_lobe',
            'deg_14_lob': 'Deg R_Frontal_lobe',
        }
        event_markers = [
            'deg_1_lob',
            'deg_2_lob',
            'deg_3_lob',
            'deg_4_lob',
            'deg_5_lob',
            'deg_7_lob',
            'deg_8_lob',
            'deg_10_lob',
            'deg_11_lob',
            'deg_12_lob',
            'deg_13_lob',
            'deg_14_lob',
        ]
    elif events_set=='lobes_2020_TEA_b' or events_set=='lobes_2020_TEA_e2': # 12 LOBES (based on anatomy) in total. Insula was merged with Cingulate
                                       # and R/L cerebellum was excluded as these lobes inlcuded only 1 regions 
                                       # thus, not so meaningful for network-based analysis.
        event_markers_dict = {
            'deg_1_lob':  'Deg L_Frontal_lobe',
            'ne_1_lob':   'Ne L_Frontal_lobe',

            'deg_2_lob':  'Deg L_Parietal_lobe',
            'ne_2_lob':   'Ne L_Parietal_lobe',

            'deg_3_lob':  'Deg L_Occipital_lobe',
            'ne_3_lob':   'Ne L_Occipital_lobe',

            'deg_4_lob':  'Deg L_Temporal_lobe',
            'ne_4_lob':   'Ne L_Temporal_lobe',

            'deg_5_lob':  'Deg L_Cingulate_lobe',
            'ne_5_lob':   'Ne L_Cingulate_lobe',

            'deg_7_lob':  'Deg L_Subcortical',
            'ne_7_lob':   'Ne L_Subcortical',

            'deg_8_lob':  'Deg R_Subcortical',
            'ne_8_lob':   'Ne R_Subcortical',

            'deg_10_lob': 'Deg R_Cingulate_lobe',
            'ne_10_lob':  'Ne R_Cingulate_lobe',

            'deg_11_lob': 'Deg R_Temporal_lobe',
            'ne_11_lob':  'Ne R_Temporal_lobe',

            'deg_12_lob': 'Deg R_Occipital_lobe',
            'ne_12_lob':  'Ne R_Occipital_lobe',

            'deg_13_lob': 'Deg R_Parietal_lobe',
            'ne_13_lob':  'Ne R_Parietal_lobe',

            'deg_14_lob': 'Deg R_Frontal_lobe',
            'ne_14_lob':  'Ne R_Frontal_lobe',
        }
        event_markers = [
            'deg_1_lob','ne_1_lob',
            'deg_2_lob','ne_2_lob',
            'deg_3_lob','ne_3_lob',
            'deg_4_lob','ne_4_lob',
            'deg_5_lob','ne_5_lob',
            'deg_7_lob','ne_7_lob',
            'deg_8_lob','ne_8_lob',
            'deg_10_lob','ne_10_lob',
            'deg_11_lob','ne_11_lob',
            'deg_12_lob','ne_12_lob',
            'deg_13_lob','ne_13_lob',
            'deg_14_lob','ne_14_lob',
        ]
    elif events_set=='lobes_2020_TEA_c': # 12 LOBES (based on anatomy) in total. Insula was merged with Cingulate
                                       # and R/L cerebellum was excluded as these lobes inlcuded only 1 regions 
                                       # thus, not so meaningful for network-based analysis.
        event_markers_dict = {
            'deg_1_lob':  'Deg L_Frontal_lobe',
            'deg_2_lob':  'Deg L_Parietal_lobe',
            'deg_3_lob':  'Deg L_Occipital_lobe',
            'deg_4_lob':  'Deg L_Temporal_lobe',
            'deg_5_lob':  'Deg L_Cingulate_lobe',
            'deg_7_lob':  'Deg L_Subcortical',
            'deg_8_lob':  'Deg R_Subcortical',
            'deg_10_lob': 'Deg R_Cingulate_lobe',
            'deg_11_lob': 'Deg R_Temporal_lobe',
            'deg_12_lob': 'Deg R_Occipital_lobe',
            'deg_13_lob': 'Deg R_Parietal_lobe',
            'deg_14_lob': 'Deg R_Frontal_lobe',
        }
        event_markers = [
            'deg_1_lob',
            'deg_2_lob',
            'deg_3_lob',
            'deg_4_lob',
            'deg_5_lob',
            'deg_7_lob',
            'deg_8_lob',
            'deg_10_lob',
            'deg_11_lob',
            'deg_12_lob',
            'deg_13_lob',
            'deg_14_lob',
        ]
    elif events_set=='lobes_2020_TEA_d' or events_set=='lobes_2020_TEA_e' : # 12 LOBES (based on anatomy) in total. Insula was merged with Cingulate
                                       # and R/L cerebellum was excluded as these lobes inlcuded only 1 regions 
                                       # thus, not so meaningful for network-based analysis.
        event_markers_dict = {
            'deg_1_lob':  'Deg L_Frontal_lobe',
            'deg_2_lob':  'Deg L_Parietal_lobe',
            'deg_3_lob':  'Deg L_Occipital_lobe',
            'deg_4_lob':  'Deg L_Temporal_lobe',
            'deg_5_lob':  'Deg L_Cingulate_lobe',
            'deg_7_lob':  'Deg L_Subcortical',
            'deg_8_lob':  'Deg R_Subcortical',
            'deg_10_lob': 'Deg R_Cingulate_lobe',
            'deg_11_lob': 'Deg R_Temporal_lobe',
            'deg_12_lob': 'Deg R_Occipital_lobe',
            'deg_13_lob': 'Deg R_Parietal_lobe',
            'deg_14_lob': 'Deg R_Frontal_lobe',
        }
        event_markers = [
            'deg_1_lob',
            'deg_2_lob',
            'deg_3_lob',
            'deg_4_lob',
            'deg_5_lob',
            'deg_7_lob',
            'deg_8_lob',
            'deg_10_lob',
            'deg_11_lob',
            'deg_12_lob',
            'deg_13_lob',
            'deg_14_lob',
        ]
    elif events_set=='lobes_2020_TEA_e3' : # 12 LOBES (based on anatomy) in total. Insula was merged with Cingulate
                                       # and R/L cerebellum was excluded as these lobes inlcuded only 1 regions 
                                       # thus, not so meaningful for network-based analysis.
                                       # L/R averaged !!!
        event_markers_dict = {
            'deg_1_14_lob':  'Deg Frontal_lobe',
            'deg_2_13_lob':  'Deg Parietal_lobe',
            'deg_3_12_lob':  'Deg Occipital_lobe',
            'deg_4_11_lob':  'Deg Temporal_lobe',
            'deg_5_10_lob':  'Deg Cingulate_Insula_lobe',
            'deg_7_8_lob':  'Deg Subcortical',
        }
        event_markers = [
            'deg_1_14_lob',
            'deg_2_13_lob',
            'deg_3_12_lob',
            'deg_4_11_lob',
            'deg_5_10_lob',
            'deg_7_8_lob',
        ]
    elif events_set=='volumes' or events_set=='volumes_ep_ctrl':
        event_markers_dict = {
            'Cortical gray matter total':  'Cortical GM',
            'Unmyelinated white matter total':  'Unmyel WM',
            'Subcortical gray matter total':  'Subcortical GM',
            'CSF':  'CSF ',
            'Cerebellum':  'Cerebellum ',
            'Brainstem':  'Brainstem ',
            'ICC':'Intracranial cavity',
        }
        event_markers = [
            'Cortical gray matter total',
            'Unmyelinated white matter total',
            'Subcortical gray matter total',
            'CSF',
            'Cerebellum',
            'Brainstem',
            'ICC',
        ]
    elif events_set=='volumes_norm' or events_set=='volumes_ep_ctrl_norm' or events_set=='volumes_clean':
        event_markers_dict = {
            'Cortical gray matter total':  'Cortical GM',
            'Unmyelinated white matter total':  'Unmyel WM',
            'Subcortical gray matter total':  'Subcortical GM',
            'CSF':  'CSF ',
            'Cerebellum':  'Cerebellum ',
            'Brainstem':  'Brainstem '
        }
        event_markers = [
            'Cortical gray matter total',
            'Unmyelinated white matter total',
            'Subcortical gray matter total',
            'CSF',
            'Cerebellum',
            'Brainstem'
        ]
    elif events_set=='volumes_clean_6_14':
        event_markers_dict = {
        	'SGM':  'Subcortical GM',
        	'WM':  'WM',
            'CGM':  'Cortical GM',
            'Brainstem':  'Brainstem',
            'Cerebellum':  'Cerebellum',
            'CSF': 'CSF'

        }
        event_markers = [
            'SGM',
            'WM',
            'CGM',
            'Brainstem',
            'Cerebellum',
            'CSF'
        ]
    elif events_set=='volumes_clean_6':
        event_markers_dict = {
        	'SGM':  'Subcortical GM',
        	'WM':  'WM',
            'CGM':  'Cortical GM',
            'Brainstem':  'Brainstem',
            'Cerebellum':  'Cerebellum',
            'CSF': 'CSF'

        }
        event_markers = [
            'SGM',
            'WM',
            'CGM',
            'Brainstem',
            'Cerebellum',
            'CSF'
        ]
    else:
        print('* * * * * * default_events(): ERROR')
        print('* * * * * * invalid events_set. Must be "data-driven" or "lobes" or "updated"')
    event_markers_labels = [event_markers_dict.get(e) for e in event_markers]
    
    return event_markers, event_markers_labels, event_markers_dict



def ebm_3_staging(x,mixtures,samples):
    """
    Given a trained EBM (mixture_models,mcmc_samples), and correctly-formatted data, stage the data
    NOTE: To use CV-EBMs, you'll need to call this for each fold, then combine.
    Author: Neil P Oxtoby, UCL, September 2018
    """
    if type(mixtures[0]) is list:
        #* List of mixture models from cross-validation
        n_cv = len(mixtures)
        prob_mat = []
        stages = []
        stage_likelihoods = []
        for k in range(n_cv):
            #* Stage the data
            prob_mat.append(get_prob_mat(x, mixtures[k]))
            stages_k, stage_likelihoods_k = samples[k][0].stage_data(prob_mat[k])
            stages.append(stages_k)
            stage_likelihoods.append(stage_likelihoods_k)
    else:
        #* Stage the data
        prob_mat = get_prob_mat(x, mixtures)
        stages, stage_likelihoods = samples[0].stage_data(prob_mat)
    return prob_mat, stages, stage_likelihoods


def ebm_3_staging_plot(stages_array,labels, export_plot_flag=True, normed=True, sub='EP', n_events=None):
    """
    WIP

    Author: Neil P Oxtoby, UCL, September 2018
    """
    if n_events is None:
        n_events = np.nanmax(np.concatenate(stages_array)) + 1
    binz = np.linspace(0,n_events,n_events+1,dtype=int)

    if len(labels)!=len(stages_array):
        labelz = [str(k) for k in range(len(stages_array))]
    else:
        labelz = labels

    fs_ax = 32
    # fig, ax = plt.subplots(2,1,sharex=True)
    # colorz = sn.color_palette('colorblind', len(labelz))
    # ax[0].hist(stages_array,
    #         label=labelz,
    #         normed=False,
    #         color=colorz,
    #         stacked=False,
    #         bins=binz)
    # ax[0].legend(loc=0,fontsize=fs_ax-2)
    # ax[0].set_yscale('log')
    # # ax[0].set_xlabel('EBM stage')
    # ax[0].set_ylabel('Count')
    # ax[0].tick_params(axis='both', which='major', labelsize=fs_ax)
    # ax[0].yaxis.label.set_size(fs_ax)
    # # ax[0].xaxis.label.set_size(42)
    # ax[1].hist(stages_array,
    #         label=labelz,
    #         normed=True,
    #         color=colorz,
    #         stacked=False,
    #         bins=binz)
    # ax[1].legend(loc=0,fontsize=fs_ax-2)
    # ax[1].set_yscale('linear')
    # ax[1].set_xlabel('EBM stage')
    # ax[1].set_ylabel('Proportion')
    # ax[1].tick_params(axis='both', which='major', labelsize=fs_ax)
    # ax[1].yaxis.label.set_size(fs_ax+6)
    # ax[1].xaxis.label.set_size(fs_ax+6)
    # # ax.set_xticklabels([str(x) for x in binz])
    # fig.set_figwidth(12)
    # fig.set_figheight(6)
    # plt.tight_layout()
    # fig.show()
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    colorz = sn.color_palette('colorblind', len(labelz))
    ax.hist(stages_array,
         label=labelz,
         normed=normed,
         color=colorz,
         stacked=False,
         bins=binz)
    ax.legend(loc=0,fontsize=fs_ax-2)
    ax.set_yscale('linear')
    ax.set_xlabel('Severity Score') #ax.set_xlabel('EBM stage')
    ax.set_ylabel(normed*'Proportion' + (~normed)*'Count')
    ax.tick_params(axis='both', which='major', labelsize=fs_ax)
    ax.yaxis.label.set_size(fs_ax+6)
    ax.xaxis.label.set_size(fs_ax+6)
    # ax.set_xticklabels([str(x) for x in binz])
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.tight_layout()
    fig.show()
    
    if export_plot_flag:
        d = str(datetime.now().date()).replace('-','')
        f_name = 'wp9_ebm_{0}-staging-{1}-hist.png'.format(sub,d)
        fig.savefig(f_name,dpi=300)

    #*** KDE fits to staging histogram for a cleaner visualisation
    x_s = np.linspace(0,n_events,100).reshape(-1,1)
    x_b = np.linspace(0,n_events,n_events+1)
    from sklearn.neighbors import KernelDensity
    p_staging = []
    for s in stages_array:
        p = KernelDensity(kernel='gaussian', bandwidth=2/3).fit(s.reshape(-1,1))
        p = np.exp(p.score_samples(x_s)).reshape(-1,1)
        p_staging.append(p)

    fig,ax = plt.subplots(1,1,figsize=(12,6))
    for k in range(len(p_staging)):
        # ax.fill_between(x_s, 0, p_staging_HC)
        ax.plot(x_s,p_staging[k],label=labelz[k])
    ax.legend(fontsize=12)
    ax.set_title('Baseline staging: {0}'.format(sub))
    ax.set_xlabel('EBM Stage')
    ax.set_ylabel('PDF')
    fig.show()
    if export_plot_flag:
        d = str(datetime.now().date()).replace('-','')
        f_name = 'wp9_ebm_{0}-staging-{1}-kde.png'.format(sub,d)
        fig.savefig(f_name,dpi=300)

    return fig, ax

    # if plot_flag:
        # n_events = x.shape[1]
        # fig, ax = plt.subplots(2,1,sharex=True)
        # # binz = np.linspace(0,n_events,int((n_events+1)/4)).astype(int)
        # binz = np.linspace(0,n_events,n_events+1).astype(int)
        # stages_array = [stages[y == 0],
        #                 stages[y == 1]]
        # labelz = ['HC',
        #           'PD']
        # colorz = sn.color_palette('colorblind', len(labelz))
        # ax[0].hist(stages_array,
        #         label=labelz,
        #         normed=False,
        #         color=colorz,
        #         stacked=False,
        #         bins=binz)
        # ax[0].legend(loc=0,fontsize=24)
        # ax[0].set_yscale('log')
        # # ax[0].set_xlabel('EBM stage')
        # ax[0].set_ylabel('Count (log scale)')
        # ax[0].tick_params(axis='both', which='major', labelsize=24)
        # ax[0].yaxis.label.set_size(36)
        # # ax[0].xaxis.label.set_size(36)
        # ax[1].hist(stages_array,
        #         label=labelz,
        #         normed=True,
        #         color=colorz,
        #         stacked=False,
        #         bins=binz)
        # ax[1].legend(loc=0,fontsize=24)
        # ax[1].set_yscale('linear')
        # ax[1].set_xlabel('EBM stage')
        # ax[1].set_ylabel('Proportion')
        # ax[1].tick_params(axis='both', which='major', labelsize=24)
        # ax[1].yaxis.label.set_size(36)
        # ax[1].xaxis.label.set_size(36)
        # # ax.set_xticklabels([str(x) for x in binz])
        # fig.set_figwidth(12)
        # fig.set_figheight(12)
        # plt.tight_layout()
        # if export_plot_flag:
        #     f_name = 'PPMI-EBM-Staging-HC_PD.png' # FIXME: add timestamp
        #     fig.savefig(f_name,dpi=300)



def ebm_2_bs(x,y,events,
    fixed_mixture_models_flag=False,
    kde_flag=True,
    mixtures=None,
    n_bs=10,n_mcmc=10000,
    return_mixtures=True,
    ):
    """
    Run bootstrapping to get a more conservative (over-)estimate of uncertainty
    Author: Neil P Oxtoby, UCL, September 2018
    """

    if fixed_mixture_models_flag:
        if mixtures is None:
            #* Fit the mixture models
            if kde_flag:
                mixtures_bs = fit_all_kde_models(x, y)
            else:
                mixtures_bs = fit_all_gmm_models(x, y)
        else:
            mixtures_bs = mixtures
        #* Run the bootstrapping with fixed MM
        mcmc_samples_bs = bootstrap_ebm_fixedMM(x, y,
                                                n_bootstrap=n_bs,
                                                n_mcmc_iter=n_mcmc,
                                                score_names=events,
                                                mix_mod=mixtures_bs)
    else:
        #* Run the bootstrapping, refitting the MM for each sample
        mixtures_bs, mcmc_samples_bs = bootstrap_ebm(x, y, 
                                                     n_bootstrap=n_bs,
                                                     n_mcmc_iter=n_mcmc,
                                                     kde_flag=kde_flag,
                                                     return_mixtures=return_mixtures)

    ml_orders_bs = [bs[0].ordering for bs in mcmc_samples_bs]

    #pvd_bs, seq_bs = extract_pvd(ml_order=ml_orders_bs,samples=mcmc_samples_bs)

    return mixtures_bs, mcmc_samples_bs, ml_orders_bs



# def ebm_2_cv(x,y,events,
#              cv_folds=StratifiedKFold(n_splits=10, shuffle=False, random_state=None),
#              kde_flag=True,plot_each_fold=False
#              ):
#     """
#     Run 10-fold cross-validation
#         FIXME: consider using the test set for something?
#         FIXME: consider nested cross-validation for selecting markers
#     Author: Neil P Oxtoby, UCL, September 2018
#     """
#     mixtures_cv = []
#     mcmc_samples_cv = []
#     ml_orders_cv = []
#
#     f = 0
#     for train_index, test_index in cv_folds.split(x, y):
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         #* Fit
#         mixtures_k, mcmc_samples_k, ml_order_k = ebm_2_run(x_train,y_train,events,kde_flag=kde_flag,plot_flag=plot_each_fold)
#         #* Save
#         mixtures_cv.append(mixtures_k)
#         mcmc_samples_cv.append(mcmc_samples_k)
#         ml_orders_cv.append(ml_order_k)
#         f+=1
#         print('CV fold {0} of {1}'.format(f,cv_folds.n_splits))
#     return mixtures_cv, mcmc_samples_cv, ml_orders_cv


def ebm_2_cv(x,y,events,
             cv_folds=StratifiedKFold(n_splits=10, shuffle=False, random_state=None),
             kde_flag=True,plot_each_fold=False,
             implement_fixed_controls=True,
             patholog_dirn_array=None
             ):
    """
    Run k-fold cross-validation
        FIXME: consider using the test set for something?
        FIXME: consider nested cross-validation for selecting markers
    Author: Neil P Oxtoby, UCL, September 2018
    """
    mixtures_cv = []
    mcmc_samples_cv = []
    ml_orders_cv = []

    f = 0
    for train_index, test_index in cv_folds.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #* Fit
        mixtures_k, mcmc_samples_k, ml_order_k = ebm_2_run(x_train,y_train,events,kde_flag=kde_flag,plot_flag=plot_each_fold,
                                                           implement_fixed_controls=implement_fixed_controls,
                                                           patholog_dirn_array=patholog_dirn_array)
        #* Save
        mixtures_cv.append(mixtures_k)
        mcmc_samples_cv.append(mcmc_samples_k)
        ml_orders_cv.append(ml_order_k)
        f+=1
        print('CV fold {0} of {1}'.format(f,cv_folds.n_splits))
    return mixtures_cv, mcmc_samples_cv, ml_orders_cv



def cv_similarity(mcmc_samples_cv,seq):
    n_folds = len(mcmc_samples_cv)
    pvd_cv = []
    for k in range(n_folds):
        pvd, seq = extract_pvd(ml_order=seq,samples=mcmc_samples_cv[k])
        pvd_normalised = pvd/np.tile(np.sum(pvd,axis=1).reshape(-1,1),(1,pvd.shape[1]))
        pvd_cv.append(pvd_normalised)
        #print(pvd_normalised)
    
    #* Hellinger distance between rows
    # => average HD between PVDs
    #   => k^2-k HDs across k folds (exclude self-distances, which are zero)
    n_events = pvd_cv[0].shape[0]
    hd = np.zeros(shape=(n_folds,n_folds))
    hd_full = np.zeros(shape=(n_folds,n_folds,n_events))

    for f in range(len(pvd_cv)):
        for g in range(len(pvd_cv)):
            for e in range(pvd_cv[f].shape[0]):
                hd[f,g] += hellinger_distance(pvd_cv[f][e],pvd_cv[g][e]) / pvd_cv[f].shape[0]
                hd_full[f,g,e] = hellinger_distance(pvd_cv[f][e],pvd_cv[g][e])
    cvs = 1 - np.mean( hd[np.triu_indices(hd.shape[0],k=1)]**2 )

    return cvs, hd_full



def extract_pvd(ml_order,samples):
    if type(ml_order) is list:
        #* List of PVDs from cross-validation/bootstrapping
        n_ = len(ml_order[0])
        pvd = np.zeros((n_,n_))
        #all_orders = np.array(ml_order)
        if type(samples[0]) is list:
            #* 10-fold CV returns MCMC samples for each fold separately in a list - concatenate them here
            all_samples = list(itertools.chain.from_iterable(samples))
        else:
            #* Bootstrapping returns MCMC samples pre-concatenated
            all_samples = samples
        all_orders = np.array([x.ordering for x in all_samples])
        for i in range(n_):
            pvd[i, :] = np.sum(all_orders == ml_order[0][i], axis=0)
        #pvd_cv, cv_rank = reorder_PVD_average_ranking(PVD=pvd)
        pvd, rank = reorder_PVD(pvd)
        seq = [ml_order[0][i] for i in rank]
    else:
        #* Single PVD (ML results)
        n_ = len(ml_order)
        pvd = np.zeros((n_,n_))
        samples_ = np.array([x.ordering for x in samples])
        seq = ml_order
        for i in range(n_):
            pvd[i, :] = np.sum(samples_ == seq[i], axis=0)
    return pvd, seq



def reorder_PVD(PVD,mean_bool=False,edf_threshold=0.5):
    """
    Reorders a PVD by scoring the frequencies in each row, then ranking in increasing order.

    Score: integral of complementary empirical distribution (1-EDF) up to a threshold.
    Rationale: the sooner the EDF gets to the threshold, the earlier it should be in the ranking.

    """

    if mean_bool:
        n_ = PVD.shape[0]
        ranking = np.linspace(1,n_,n_) # weights
        weights = PVD
        mean_rank = []
        for i in range(n_):
            mean_rank.append( sum( weights[i,:] * ranking ) / sum(weights[i,:]) )
        new_order = np.argsort(mean_rank)
    else:
        #* Find where the empirical distribution first exceeds the threshold
        edf = np.cumsum(PVD,axis=1)
        edf = edf / np.tile(np.max(edf,axis=1).reshape(-1,1),(1,edf.shape[1]))
        edf_above_threshold = []
        for k in range(edf.shape[0]):
            edf_above_threshold.append(np.where(edf[k,:]>=edf_threshold)[0][0])
        #* Ties implicitly split by original ordering in the PVD (likely the ML ordering)
        edf_rank = np.argsort(edf_above_threshold)
        new_order = edf_rank

    PVD_new = PVD[new_order,:]
    # PVD_new = np.zeros((n_,n_))
    # for i in range(n_):
    #     PVD_new[i, :] = PVD[new_order[i],:]

    return PVD_new, new_order



def ebm_plot(seq,labels,samples):
    """
    Author: Neil P Oxtoby, UCL, September 2018
    """
    pvd, seq_ = extract_pvd(seq,samples)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pvd, interpolation='nearest', cmap='Oranges')
    labels_ = [labels[i].replace('TOTAL','').replace('TOT','').replace('-detrended','') for i in seq_]

    n_biomarkers = pvd.shape[0]
    stp = 1
    fs = 18
    if n_biomarkers>8:
        stp = 2
        fs = 8
    tick_marks_x = np.arange(0,n_biomarkers,stp)
    x_labs = range(1, n_biomarkers+1,stp)
    ax.set_xticks(tick_marks_x)
    ax.set_xticklabels(x_labs, rotation=0,fontsize=12)
    tick_marks_y = np.arange(n_biomarkers)
    ax.set_yticks(tick_marks_y+0.2)
    labels_trimmed = [x[2:].replace('_', ' ') if x.startswith('p_') else x.replace('_', ' ') for x in labels_]
    ax.set_yticklabels(labels_trimmed,#,np.array(labels_trimmed, dtype='object')[seq_],
                       rotation=0, ha='right',
                       rotation_mode='anchor',
                       fontsize=fs)
    ax.set_ylabel('Biomarker Name', fontsize=32)
    ax.set_xlabel('Event Order', fontsize=32)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig
    return fig, ax


# def ebm_2_run(x,y,events,
#               kde_flag=True,
#               verbose_flag = False,
#               plot_flag = False,
#               export_plot_flag = True
#               ):
#     """
#     Build a KDE EBM from the data in df_EBM[events]
#     Author: Neil P. Oxtoby, UCL, September 2018
#     """
#     #* Fit the mixture models
#     if kde_flag:
#         mixtures = fit_all_kde_models(x, y)
#     else:
#         mixtures = fit_all_gmm_models(x, y)
#     #* MCMC sequencing
#     mcmc_samples = mcmc(x, mixtures)
#     ml_order = mcmc_samples[0].ordering # max-like order
#     #* Print out the biomarkers in ML order
#     if verbose_flag:
#         for k in range(len(ml_order)):
#             print('ML order: {0}'.format(events[ml_order[k]]))
#     if plot_flag:
#         #* Plot the positional variance diagram
#         events_labels = [l.replace('-detrended','') for l in events]
#         fig, ax = mcmc_uncert_mat(mcmc_samples, score_names=events_labels)
#         fig.tight_layout()
#         ax.tick_params(axis='both', which='major', labelsize=14)
#         ax.yaxis.label.set_size(24)
#         ax.xaxis.label.set_size(24)
#         fig.set_figwidth(14)
#         fig.set_figheight(14)
#         #* Export figure
#         if export_plot_flag:
#             f_name = 'PPMI-EBM-PVD.png' # FIXME: add timestamp
#             fig.savefig(f_name,dpi=300)
#
#     return mixtures, mcmc_samples, ml_order


def ebm_2_run(x,y,events,
              kde_flag=True,
              verbose_flag = False,
              plot_flag = False,
              export_plot_flag = True,
              implement_fixed_controls=True,
              patholog_dirn_array=None
              ):
    """
    Build a KDE EBM from the data in df_EBM[events]
    Author: Neil P. Oxtoby, UCL, September 2018
    """
    #* Fit the mixture models
    if kde_flag:
        mixtures = fit_all_kde_models(x, y, implement_fixed_controls=implement_fixed_controls, patholog_dirn_array=patholog_dirn_array)
    else:
        mixtures = fit_all_gmm_models(x, y, implement_fixed_controls=implement_fixed_controls)
    
    #* MCMC sequencing
    mcmc_samples = mcmc(x, mixtures) # X is the whole dataset
    #print(x.shape)
    ml_order = mcmc_samples[0].ordering # max-like order
    
    #* Print out the biomarkers in ML order
    if verbose_flag:
        for k in range(len(ml_order)):
            print('ML order: {0}'.format(events[ml_order[k]]))
    if plot_flag:
        #* Plot the positional variance diagram
        events_labels = [l.replace('-detrended','') for l in events]
        fig, ax = mcmc_uncert_mat(mcmc_samples, score_names=events_labels)
        fig.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.yaxis.label.set_size(24)
        ax.xaxis.label.set_size(24)
        fig.set_figwidth(14)
        fig.set_figheight(14)
        #* Export figure
        if export_plot_flag:
            f_name = 'PPMI-EBM-PVD.png' # FIXME: add timestamp
            fig.savefig(f_name,dpi=300)

    return mixtures, mcmc_samples, ml_order


def check_missing(x,y,events=None):
    """
    Quantifies missingness of event data in matrix X, per diagnosis class in {0,1}
    Author: Neil P Oxtoby, UCL, September 2018
    """
    #n0 = sum(y==0)
    #n1 = sum(y==1)
    ne = x.shape[1]
    ni = x.shape[0]
    x_missing = np.isnan(x)
    f_missing_per_event = []
    f_missing_per_individual = []
    bins = np.linspace(0,1,25)
    for dx in range(0,2):
        x_missing_dx = x_missing[y==dx,:]
        f_missing_per_event.append(np.sum(x_missing_dx,axis=0)/x_missing_dx.shape[0])
        f_missing_per_individual.append(np.sum(x_missing_dx,axis=1)/x_missing_dx.shape[1])
    
    # fig, ax = plt.subplots(1,1)
    # for dx in range(0,2):
    #     ax.hist(f_missing_per_event[dx], bins, normed=True, alpha=0.5, label='Per event (dx = {0})'.format(dx))
    #     ax.hist(f_missing_per_individual[dx], bins, normed=True, alpha=0.5, label='Per individual (dx = {0})'.format(dx))
    # ax.set_ylabel('Probability Mass')
    # ax.set_xlabel('Fraction missing')
    # ax.legend()
    
    fig, ax = plt.subplots(1,2,sharey=True)
    for dx in range(0,2):
        ax[dx].plot(np.arange(0,len(f_missing_per_event[dx])),np.transpose(f_missing_per_event[dx]))
        ax[dx].set_xticks(np.where(f_missing_per_event[dx]>0.1)[0])
        if events is not None:
            ax[dx].set_xticklabels([events[k][0:12] for k in range(len(events)) if f_missing_per_event[dx][k]>0.1],rotation=70)
    ax[0].set_title('Controls')
    ax[0].set_ylabel('Fraction missing data')
    ax[0].set_xlabel('Events')
    ax[1].set_title('Patients')
    ax[1].set_xlabel('Events')
    fig.show()
    
    return fig, ax, f_missing_per_event, f_missing_per_individual



def ebm_1_extract_data(df,bl,markers,dx_column='DX',sub='EP',events_set='data-driven'):
    """
    EBM prep.
    This does two things to the input DataFrame:
      1. Remove mixture model failures (via manual intervention)
      2. Returns baseline data only
      3. Missing data: removes events where both diagnoses have more missing data than not missing,
         i.e, keeps x if both x_controls and x_cases have data for at least 50% of individuals within each group

    I manually checked KDE mixture models (using manual_check_kdemm()) that I fit to the putative events,
    and postselected accordingly:

    - Remove failures:
      - Non-monotonic event probabilities
        When mixture components have multiple peaks (future work: regularise this in the KDEMM),
        often due to outliers or heavy tails (e.g., when NC are completely contained within the extremes of MC)

    Author: Neil P Oxtoby, UCL, September 2018
    """
    #*** Manual postselection: see also assistor function manual_check_kdemm()
    if events_set=='updated_LE':
        # this is the LE dataset
        if sub=='EP':
            markers_to_remove_postMM = ['deg_1','ne_1','deg_3','ne_2','ne_3', 'cc_0',
                                        'cc_1','cc_2','cc_3','cc_4','cc_5'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_0','ne_0','deg_1','ne_1','deg_3','ne_3', 
                                        'cc_0','cc_1','cc_2','cc_3','cc_4','cc_5'] #* Increase with abnormality: cc3, cc4, ne6
            markers_to_remove_postCV = []
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = []
            markers_to_remove_postCV = []
    
    elif events_set=='updated_FG':
        # Fast greedy dataset, CODE RAISES SINGULAR MATRIX ERROR FOR THIS CASE
        if sub=='EP':
            markers_to_remove_postMM = ['deg_1','ne_1', 'cc_0','cc_1','cc_2','cc_3',
                                        'cc_4','cc_5', 'cc_6', 'cc_7'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_0', 'ne_1', 'cc_0','cc_1','cc_2','cc_3',
                                        'cc_4','cc_5', 'cc_6', 'cc_7'] #* Increase with abnormality: cc3, cc4, ne6
            markers_to_remove_postCV = []
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = []
            markers_to_remove_postCV = []
    
    elif events_set=='data-driven':
        if sub=='EP':
            markers_to_remove_postMM = ['deg2','cc2','cc4','cc6'] # Increase with abnormality: ne2,deg6,ne6
            markers_to_remove_postCV = ['deg3','cc5']
        elif sub=='IUGR':
            markers_to_remove_postMM = ['cc2','cc5','cc6'] #* Increase with abnormality: cc3, cc4, ne6
            markers_to_remove_postCV = ['ne6','cc3','deg3','deg6','deg4','ne4']
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = []
            markers_to_remove_postCV = []
    
    elif events_set=='lobes':
        if sub=='EP':
            markers_to_remove_postMM = ['deg_1_lob','cc_7_lob','cc_11_lob'] # cc_9_lob (bit dodgy), ne_11_lob (p_event skewed right)
            markers_to_remove_postCV = ['deg_3_lob','ne_3_lob','cc_7_lob','cc_5_lob',
                                        'ne_7_lob','cc_11_lob','cc_2_lob','cc_12_lob'] 
            # deg4, deg7, cc10, deg11, deg8, ne10, deg10, cc9
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_4_lob','ne_4_lob','cc_5_lob','cc_10_lob'] #cc_4_lob
            markers_to_remove_postCV = []
    
    elif events_set=='global':
        if sub=='EP':
            markers_to_remove_postMM = [] 
            markers_to_remove_postCV = [] 
        elif sub=='IUGR':
            markers_to_remove_postMM = [] # should be ['Ne'] but breaks plotting function
            markers_to_remove_postCV = []
    
    elif events_set=='lobes_2020':
        if sub=='EP':
            markers_to_remove_postMM = [] 
            markers_to_remove_postCV = [] 
        elif sub=='IUGR':
            markers_to_remove_postMM = []
            markers_to_remove_postCV = []

    elif events_set=='lobes_2020_b': # all CC were removed
        if sub=='EP':
            markers_to_remove_postMM = ['deg_1_lob', 'ne_1_lob', 'deg_5_lob', 'ne_5_lob', 'deg_7_lob', 
                                        'ne_7_lob' ,'deg_10_lob', 'ne_10_lob','ne_11_lob']
            markers_to_remove_postCV = [] 
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_2_lob', 'ne_2_lob', 'deg_4_lob','ne_4_lob','deg_5_lob', 
                                        'ne_5_lob','deg_7_lob','ne_7_lob','deg_8_lob','ne_8_lob',  
                                        'deg_10_lob','ne_10_lob', 'deg_13_lob', 'ne_13_lob']
            markers_to_remove_postCV = []

    elif events_set=='lobes_2020_6yo_b': # all CC were removed in advance
    ###* Anatomical-based decomposition of the 84 free surfer regions into 12 lobes.
        
        if sub=='EP':
            markers_to_remove_postMM = ['deg_5_lob', 'ne_5_lob', 'deg_7_lob', 
                                        'ne_7_lob' ,'deg_10_lob', 'ne_10_lob','deg_14_lob']
            markers_to_remove_postCV = [] 
        
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_4_lob','ne_4_lob','deg_5_lob', 'ne_5_lob',
                                        'deg_7_lob', 'ne_7_lob','deg_8_lob','ne_8_lob',  
                                        'deg_10_lob','ne_10_lob', 'deg_11_lob', 'ne_11_lob',
                                        'deg_13_lob']
            markers_to_remove_postCV = []
        
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = ['deg_5_lob', 'ne_5_lob',
                                        'deg_7_lob', 'ne_7_lob', 
                                        'deg_10_lob','ne_10_lob']
            markers_to_remove_postCV = []

    elif events_set=='lobes_2020_6yo_c': # all CC were removed in advance
    ###* Anatomical-based decomposition of the 84 free surfer regions into 12 lobes.
        
        if sub=='EP':
            markers_to_remove_postMM = ['deg_5_lob','deg_7_lob','deg_10_lob','deg_14_lob']
            markers_to_remove_postCV = [] 
        
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_4_lob','deg_5_lob','deg_7_lob','deg_8_lob','deg_10_lob','deg_13_lob']
            markers_to_remove_postCV = []
        
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = ['deg_5_lob','deg_7_lob','deg_10_lob']
            markers_to_remove_postCV = []

    elif events_set=='updated_FG_b':
        # all CCs were removed, community 3 was removed (has a lot of 0s)
        if sub=='EP':
            markers_to_remove_postMM = ['deg_0','deg_5','ne_5', 'ne_6', 'ne_7'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
        elif sub=='IUGR':
            markers_to_remove_postMM = ['deg_0','ne_0','deg_1','ne_1', 'deg_5',
                                        'ne_5', 'deg_6','ne_6'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = []
            markers_to_remove_postCV = []

    elif events_set=='updated_LE_b':
        # all CCs were removed
        if sub=='EP':
            markers_to_remove_postMM = [] # Non-monotonic P(event)
            markers_to_remove_postCV = []
        elif sub=='IUGR':
            markers_to_remove_postMM = [] # Non-monotonic P(event)
            markers_to_remove_postCV = []
        elif sub=='EP+IUGR':
            markers_to_remove_postMM = []
            markers_to_remove_postCV = []
    
    elif events_set=='lobes_2020_TEA_b':
        # all CCs were removed
        if sub=='PT':
            markers_to_remove_postMM = ['deg_3_lob','ne_5_lob', 'deg_7_lob', 
                                        'ne_7_lob','deg_8_lob', 'ne_8_lob'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
    
    elif events_set=='lobes_2020_TEA_c':
        # all CCs were removed
        if sub=='PT':
            markers_to_remove_postMM = ['deg_3_lob','deg_7_lob','deg_8_lob'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
    
    elif events_set=='lobes_2020_TEA_d':
        # all CCs were removed
        if sub=='PT':
            markers_to_remove_postMM = ['deg_10_lob','deg_12_lob','deg_13_lob'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
    
    elif events_set=='lobes_2020_TEA_e2':
        # all CCs were removed
        if sub=='PT':
            markers_to_remove_postMM = ['ne_5_lob', 'deg_8_lob', 'ne_8_lob'] # Non-monotonic P(event) 
            markers_to_remove_postCV = []

    elif events_set=='lobes_2020_TEA_e':
        # all CCs were removed
        if sub=='PT':
            markers_to_remove_postMM = ['deg_7_lob', 'deg_8_lob' ] # Non-monotonic P(event)  2/3/4/5/11 (
            markers_to_remove_postCV = []

    elif events_set=='lobes_2020_TEA_e3':
        # all CCs were removed
        if sub=='PT':
            markers_to_remove_postMM = ['deg_7_8_lob'] # Non-monotonic P(event) 
            markers_to_remove_postCV = []
    
    elif events_set=='volumes' or events_set=='volumes_ep_ctrl':
        if sub=='PT':
            markers_to_remove_postMM = [] # Non-monotonic P(event)
            markers_to_remove_postCV = []


    elif events_set=='volumes_clean':
        if sub=='PT':
            markers_to_remove_postMM = [] # Non-monotonic P(event)
            markers_to_remove_postCV = []
    elif events_set=='volumes_clean_6_14':
        if sub=='PT':
            markers_to_remove_postMM = [] # Non-monotonic P(event)
            markers_to_remove_postCV = []

    elif events_set=='volumes_clean_6':
        if sub=='PT':
            markers_to_remove_postMM = [] # Non-monotonic P(event)
            markers_to_remove_postCV = []

    elif events_set=='volumes_norm':
        if sub=='PT':
            markers_to_remove_postMM = ['Cortical gray matter total', 'Subcortical gray matter total', 'Brainstem'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
    
    elif events_set=='volumes_ep_ctrl_norm':
        if sub=='PT':
            markers_to_remove_postMM = ['Cortical gray matter total', 'Subcortical gray matter total', 'Brainstem'] # Non-monotonic P(event)
            markers_to_remove_postCV = []
    else:
        print('* * * * * * ebm_1_extract_data() ERROR: invalid input "events_set"')
    # #*** Manual postselection: see also assistor function manual_check_kdemm()
    # if sub=='EP':
    #     markers_to_remove_postMM = ['deg2','cc2','cc4','cc6']
    #     markers_to_remove_postCV = ['deg3','cc5','cc3'] # 'ne2', 'deg1', 'ne1', 'ne4', 'deg4'
    # elif sub=='IUGR':
    #     markers_to_remove_postMM = ['cc2','cc5','cc6'] # 'cc3','cc4'
    #     markers_to_remove_postCV = ['deg2','ne6','deg6','cc3','ne4','deg4']
    # elif sub=='EP+IUGR':
    #     markers_to_remove_postMM = ['cc2','cc3','cc4','cc6']
    #     markers_to_remove_postCV = ['ne4','ne1','deg3','cc5'] #'deg4'
    markers_to_remove_post = markers_to_remove_postMM + markers_to_remove_postCV
    
    #* Add suffix: '-detrended'
    markers_to_remove_detrended  = [m+'-detrended' for m in markers_to_remove_post]
    markers_all_detrended_postselectedManually = [m for m in markers if not(m in markers_to_remove_detrended)]
    print('\n Manually postselected markers (e.g. removing KDE-MM failures): {0}\n'.format(markers_all_detrended_postselectedManually))
    
    #* Missing Data
    if type(bl) is pd.Series:
        bl = bl.values
    x = df.loc[bl,markers_all_detrended_postselectedManually].values
    y = df.loc[bl,dx_column].values
    fig, ax, f_missing_per_event, f_missing_per_individual = check_missing(x=x,y=y,events=markers_all_detrended_postselectedManually)
    missing_data_threshold = 0.5
    events_with_too_much_missing_data = [(f>missing_data_threshold) or (g>missing_data_threshold) for f,g in zip(f_missing_per_event[0],f_missing_per_event[1])]
    if len(events_with_too_much_missing_data)>0:
        print('   ...ebm_1_extract_data(): removed events having too much missing data: {0}'.format(', '.join([markers_all_detrended_postselectedManually[k] for k in np.where(events_with_too_much_missing_data)[0]])))
    
    e_keep_bool = [~e for e in events_with_too_much_missing_data]
    markers_all_detrended_postselectedManually = [markers_all_detrended_postselectedManually[k] for k in np.where(e_keep_bool)[0]]
    
    #* Mann-Whitney U test
    #  NH: a sample from one distribution is equally likely to be either greater/less than a sample from the other distribution
    mwu_stat = {}
    mwu_p = {}
    mwu_rejects = []
    n_mwu = len(markers_all_detrended_postselectedManually) # Multiple Comparisons
    for k in range(len(markers_all_detrended_postselectedManually)):
        m = markers_all_detrended_postselectedManually[k]
        x_controls = df.loc[bl & (df[dx_column]==0),m].values
        x_patients = df.loc[bl & (df[dx_column]==1),m].values
        mwu_stat[m],mwu_p[m] = mwu(x_controls,x_patients)
        if mwu_p[m]>(0.05/n_mwu):
            mwu_rejects.append(m)
    if len(mwu_rejects)>0:
        print('The following markers showed no significant statistical difference between controls and cases (Mann-Whitney U test; p > 0.05/N, N = number of postselected events):')
        for k in range(len(mwu_rejects)):
            print('  - {0}'.format(mwu_rejects[k]))
    
    #* Get readable labels: FIXME - move this to a comprehensive function events_dict() or similar
    e, el, event_markers_dict = default_events(events_set=events_set)
    markers_post_labels = [m.replace('-detrended','') for m in markers_all_detrended_postselectedManually]
    markers_post_labels = [event_markers_dict.get(m) for m in markers_post_labels]
    
    return df.loc[bl], markers_all_detrended_postselectedManually, markers_post_labels


# def fit_mixtures_x(df, marker='deg1', plot_bool=True, dx_column='DX', hist_bool=False, sub='EP'):
#     """
#     Subroutine designed for manual_check_kdemm()
#     """
#     class_names = ['Controls',sub]
#
#     #* Extract data
#     X = df[marker].values
#     y = df[dx_column].values
#
#     #* Include only dx=0/1 (means you can only stage data within these limits - any dx!=0/1 need to fall within these groups: not always the case)
#     #rowz = (y==1) | (y==0)
#     #X = X[rowz]
#     #y = y[rowz]
#
#     if np.alltrue(np.isnan(X)):
#         print('Data is all NaN!')
#         return None, None
#
#     #* Remove NaN before fitting KDEs
#     y = y[~np.isnan(X)]
#     X = X[~np.isnan(X),].reshape(-1,1)
#     p_q_kde_mixture = fit_all_kde_models(X, y)
#
#     hist_c = ['g','r']
#     if plot_bool:
#         fig,ax = plt.subplots(figsize=(12,6))
#         #* Range of the data
#         n = 500
#         x_lower = np.nanmin(X)
#         x_upper = np.nanmax(X)
#         x_ = np.linspace(x_lower,x_upper,n).reshape(-1,1)
#         if hist_bool:
#             hist_dat = [X[y == 0],
#                         X[y == 1]]
#             leg1 = ax.hist(hist_dat,
#                            label=class_names,
#                            normed=True,
#                            color=hist_c,
#                            alpha=0.7,
#                            stacked=False)
#         controls_score, patholog_score = p_q_kde_mixture[0].pdf(x_)
#         probability = 1 - p_q_kde_mixture[0].probability(x_)
#         probability *= np.max((patholog_score, controls_score))
#         ax.plot(x_, controls_score, color=hist_c[0], label='Normal')
#         ax.plot(x_, patholog_score, color=hist_c[1], label='Abnormal')
#         leg2 = ax.plot(x_, probability, color='k', label='P(event)')
#         ax.set_title(marker)
#         ax.axes.get_yaxis().set_visible(False)
#         if hist_bool:
#             fig.legend(leg1[2]+leg2, list(class_names) + ['P(event)'],
#                        loc='upper left', fontsize=15)
#         else:
#             ax.legend(loc='upper left', fontsize=15)
#
#     return fig, ax


def fit_mixtures_x(df, marker, plot_bool=True, dx_column='DX', hist_bool=False,
                   implement_fixed_controls=True, patholog_dirn=None, sub='EP'):
    """
    Subroutine designed for manual_check_kdemm()
    """
    class_names = ['Control (IUGR)',sub]
    
    #* Extract data
    X = df[marker].values
    y = df[dx_column].values
    
    if np.alltrue(np.isnan(X)):
        print('Data is all NaN!')
        return None, None

    #* Remove NaN before fitting KDEs
    y = y[~np.isnan(X)]
    X = X[~np.isnan(X),].reshape(-1,1)
    #print(X[y == 0].shape)
    #print(X[y == 1].shape)
    p_q_kde_mixture = fit_all_kde_models(X, y, implement_fixed_controls=implement_fixed_controls, patholog_dirn_array=[patholog_dirn])

    hist_c = ['g','r']
    if plot_bool:
        fig,ax = plt.subplots(figsize=(12,6))
        #* Range of the data
        n = 500
        x_lower = np.nanmin(X)
        x_upper = np.nanmax(X)
        x_ = np.linspace(x_lower,x_upper,n).reshape(-1,1)
        if hist_bool:
            #hist_dat = [X[y == 0],
            #            X[y == 1]]
            # the reshape was added by Serafeim to make this work
            hist_dat = [X[y == 0].reshape(-1,),
                        X[y == 1].reshape(-1,)]
            #print(len(hist_dat))
            leg1 = ax.hist(hist_dat,
                           label=class_names,
                           density=True, # 1/(N*bin_width)see https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html
                           color=hist_c,
                           alpha=0.7,
                           stacked=False)
        controls_score, patholog_score = p_q_kde_mixture[0].pdf(x_) # normalized: see https://github.com/ncfirth/kde_ebm_open/blob/master/kdeebm/mixture_model/kde.py#L83
        probability = 1 - p_q_kde_mixture[0].probability(x_) # see https://github.com/ncfirth/kde_ebm_open/blob/master/kdeebm/mixture_model/kde.py#L89
        probability *= np.max((patholog_score, controls_score)) # why *= np.max((patholog_score, controls_score)) ???
        #print(patholog_score.shape )
        ax.plot(x_, controls_score, color=hist_c[0], label='Normal')
        ax.plot(x_, patholog_score, color=hist_c[1], label='Abnormal')
        leg2 = ax.plot(x_, probability, color='k', label='P(event)')
        ax.set_title(marker)
        #ax.axes.get_yaxis().set_visible(False)
        if hist_bool:
            fig.legend(leg1[2]+leg2, list(class_names) + ['P(event)'],
                       loc='upper left', fontsize=15)
        else:
            ax.legend(loc='upper left', fontsize=15)
            
    return fig, ax

# def manual_check_kdemm(df,markers,privacy=False,sub='EP'):
#     """Simple loop over detrended biomarker data, plotting histograms and mixture models as we go - for manual inspection"""
#     for m in markers:
#         fig, ax = fit_mixtures_x(df, marker=m, hist_bool=~privacy, sub=sub)
#         if privacy:
#             ax.set_xticks([])
#         fig.show()
#         input('\nPress enter to continue...')
#     print('\nI hope you kept note of the biomarker events you want to keep')


def manual_check_kdemm(df,markers,hist_bool=False, implement_fixed_controls=True, patholog_dirn_array=None, sub='EP'):
    """Simple loop over detrended biomarker data, plotting histograms and mixture models as we go - for manual inspection"""
    for m,d in zip(markers,patholog_dirn_array):
        fig, ax = fit_mixtures_x(df, marker=m, hist_bool=hist_bool, implement_fixed_controls=implement_fixed_controls, patholog_dirn=d, sub=sub)
        #ax.set_xticks([])
        fig.show()
        #input('\nPress enter to continue...')
    print('\nI hope you kept note of the biomarker events you want to keep')


# def log_transform(df,markers):
#     """Log-transform CSF/blood markers: add small positive number to avoid log(zero)"""
#     markers_to_log_transform = ['Abeta 42','ABeta 1-42',
#                                 'CSF Alpha-synuclein',
#                                 'CSF Hemoglobin',
#                                 'Total tau', 'tTau', 'p-Tau181P', 'pTau',
#                                 'Triglycerides', 'EGF ELISA', 'HDL', 'IgG', 'IgG3', 'IgG3/IgG', 'LDL']
#
#     markers_to_log_transform = [m for m in markers_to_log_transform if m in markers]
#
#     df_ = df.copy()
#     markers_logged = markers.copy()
#
#     for m in markers_to_log_transform:
#         if m in df_.columns:
#             eps = np.nanmin(df_[m])/2 + np.finfo(float).eps
#             y_logged = np.log10( df_[m] + eps ).values
#             #* Save
#             df_[m+'-log10'] = y_logged
#             markers_logged[ np.where([e==m for e in markers])[0][0] ] = m+'-log10'
#
#
#     return df_, markers_logged
#
#
# def select_cohort(df,cohort='denovo'):
#
#     if cohort=='denovo':
#         #* denovo: restrict APPRDX to only HC, PD, SWEDD, Prodromal (RBD/hyposmia)
#         APPRDX_denovoEBM = [1,2,3,4]
#         df_ = df.loc[df.APPRDX_enrol.isin(APPRDX_denovoEBM)].copy()
#         bl = df_['EVENT_ID'].apply(lambda x: x=='BL')
#         APPRDX_map = {1: 1, 2: 0, 3:3, 4:4 } # 5:3, 6:3, 7:3, 8:3
#         PATNO_abnormals  = df_.PATNO[df_['PD_denovo_amplified'] & bl]
#     elif cohort=='genetic':
#         #* genetic:
#         #    - restrict APPRDX to only HC, carriers (all from Genetic Cohort; carriers only from Genetic Registry)
#         APPRDX_geneticEBM = [5,6,7,8]
#         carriers = (df['Genetic'] & (df.APPRDX_enrol.isin(APPRDX_geneticEBM)))
#         selection_genetic = (df.APPRDX_enrol==2) | carriers
#         df_ = df.loc[selection_genetic].copy()
#         bl = df_['EVENT_ID'].apply(lambda x: x=='BL')
#         #* genetic at-risk: noncarriers in Genetic Registry who have a carrier as a 1st-degree relative
#         #GeneticRegistry_carriers =    ( (df.APPRDX_enrol.isin([7,8])) & (df.MUTRSLT==1) ).values
#         #GeneticRegistry_noncarriers = ( (df.APPRDX_enrol.isin([7,8])) & (df.MUTRSLT==0) ).values
#         #df_genetic_atrisk = df.loc[GeneticRegistry_noncarriers].copy()
#         #bl_genetic_atrisk = df_genetic_atrisk['EVENT_ID'].apply(lambda x: x=='BL')
#         APPRDX_map = { 2:0 , 5:1 , 7:1 , 6:3 , 8:4} # 1:3,3:3,4:3
#         PATNO_abnormals = df_.PATNO[df_['PD_genetic_amplified'] & bl]
#     else:
#         raise ValueError('WTF are you doing? cohort must be denovo or genetic')
#
#     #* Prep for cleaning / covariate adjustment
#     PATNO_normals = df_.PATNO[df_['HC_amplified'] & bl]
#     df_['normals_EBM']   = df_.PATNO.isin(PATNO_normals)
#     df_['abnormals_EBM'] = df_.PATNO.isin(PATNO_abnormals)
#     df_.loc[:,'DX_coded'] = df_['APPRDX_enrol'].map(APPRDX_map)
#
#     #
#     # df_EBM_denovo.loc[:,'DX_coded'] = df_EBM_denovo['APPRDX_enrol'].map(APPRDX_map_denovo)
#
#
#     #* Longitudinal
#     # n = df.PATNO.isin(PATNO_normals)
#     # a = df.PATNO.isin(PATNO_abnormals_denovo)
#     # a_g = df.PATNO.isin(PATNO_abnormals_genetic)
#
#     return df_, bl
#


def check_for_save_file(file_name,function):
    if os.path.isfile(file_name):
        print('check_for_save_file(): File detected ({0}) - you can load data.'.format(file_name))
        #ebm_save = sio.loadmat(file_name)
        return 1
    else:
        if function is None:
            print('You should call your function')
        else:
            print('You should call your function {0}'.format(function.__name__))
        return 0


def hellinger_distance(p,q):
    #hd = np.linalg.norm(np.sqrt(p)-np.sqrt(q),ord=2)/np.sqrt(2)
    #hd = (1/np.sqrt(2)) * np.sqrt( np.sum( [(np.sqrt(pi) - np.sqrt(qi))**2 for pi,qi in zip(p,q)] ) )
    hd = np.sqrt( np.sum( (np.sqrt(p) - np.sqrt(q))**2 ) / 2 )
    return hd


# def print_demographics(demographic,demographic_label,groups,labels):
#     """
#     PPMI EBM prep.
#     Prints demographics summary. Useful for the paper.
#     Author: Neil P Oxtoby, UCL, September 2018
#     """
#     print('{0}: mean (std)'.format(demographic_label))
#     lj = [len(l) for l in labels]
#     lj = max(lj)
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups[k]
#         print('  {0} - {1} ({2})'.format(l.ljust(lj+1),
#                                          round(np.nanmean(demographic[g]),1),
#                                          round(np.nanstd(demographic[g]),1)))
#     return 0
#
# def report_numbers(df):
#     """
#     PPMI EBM prep.
#     Prints some numbers for included PPMI participants. Useful for the paper.
#     Author: Neil P Oxtoby, UCL, September 2018
#     """
#     #*** Report some numbers ***#
#     print('PPMI Merge data loaded:\n {0} rows, \n  {1} columns, \n {2} individuals'.format(df.shape[0],df.shape[1],len(np.unique(df.PATNO))))
#
#     bl = df['EVENT_ID'].apply(lambda x: x=='BL')
#
#     #* Report demographics by sex
#     Males = df.GENDER==2
#     Females_of_child_bearing_potential = df.GENDER==0
#     Females = df.GENDER<2
#
#     #* Genetic: separate preclinical and clinical
#     GeneticCohort_affected   = (df.APPRDX_enrol==5).values
#     GeneticCohort_unaffected = (df.APPRDX_enrol==6).values
#     GeneticRegistry_affected = (df.APPRDX_enrol==7).values
#     GeneticRegistry_unaffected = (df.APPRDX_enrol==8).values
#
#     #* Print the numbers
#     print('Participant numbers')
#     print('  Total       - {0}, {1} Females ({2} of child-bearing potential)'.format(
#         sum(bl),
#         sum(Females & bl),
#         sum(Females_of_child_bearing_potential & bl)))
#     print('  HC          -  {0},  {1} Females'.format( sum(df['HC'] & bl),       sum(df['HC'] & bl & Females)))
#     print('  PD          -  {0}, {1} Females'.format(  sum(df['PD'] & bl),       sum(df['PD'] & bl & Females)))
#     print('  SWEDD       -   {0},  {1} Females'.format(sum(df['SWEDD'] & bl),    sum(df['SWEDD'] & bl & Females)))
#     print('  Prodromal   -   {0},  {1} Females'.format(sum(df['Prodromal'] & bl),sum(df['Prodromal'] & bl & Females)))
#     print('  Genetic Cohort -  {0}   affected, {1} Females '.format(sum(GeneticCohort_affected & bl),  sum(GeneticCohort_affected & bl & Females)))
#     print('                 -  {0} unaffected, {1} Females '.format(sum(GeneticCohort_unaffected & bl),sum(GeneticCohort_unaffected & bl & Females)))
#     print('  Genetic Registry -  {0}   affected,  {1} Females '.format(sum(GeneticRegistry_affected & bl),  sum(GeneticRegistry_affected & bl & Females)))
#     print('                   -  {0} unaffected, {1} Females '.format( sum(GeneticRegistry_unaffected & bl),sum(GeneticRegistry_unaffected & bl & Females)))
#     #* Carriers combined
#     print('  Genetic PD (carriers)')
#     print('       clinical (affected)   -  {0},  {1} Females '.format(sum(df['Genetic_affected'] & bl),  sum(df['Genetic_affected'] & bl & Females)))
#     print('    preclinical (unaffected) -  {0},  {1} Females '.format(sum(df['Genetic_unaffected'] & bl),  sum(df['Genetic_unaffected'] & bl & Females)))
#
#     print('\n"Amplified" groups for analysis (DaTSCAN visual interpretations):')
#     print('  HC         (-)          -  {0},  {1} Females'.format(sum(df['HC_amplified'] & bl),       sum(df['HC_amplified'] & bl & Females)))
#     print('  de novo PD (+)          -  {0}, {1} Females'.format(sum(df['PD_denovo_amplified'] & bl),sum(df['PD_denovo_amplified'] & bl & Females)))
#     print('  genetic PD (aff/un) (+/-) -  {0}/{1}, {2}/{3} Females <<<<<< carriers Genetic Cohort+Registry'.format(
#         sum(df['PD_genetic_amplified'] & bl),
#         sum(df['PD_genetic_amplified_preclinical'] & bl),
#         sum(df['PD_genetic_amplified'] & bl & Females),
#         sum(df['PD_genetic_amplified_preclinical'] & bl & Females)))
#     print('  SWEDD      (-)          -   {0},  {1} Females'.format(sum(df['SWEDD_amplified'] & bl),sum(df['SWEDD_amplified'] & bl & Females)))
#     print('  Prodromal  (-)          -   {0},   {1} Females'.format(sum(df['Prodromal_amplified'] & bl),sum(df['Prodromal_amplified'] & bl & Females)))
#
#     #*** Print the demographics
#     print('\n\nBaseline Demographics (amplified groups)')
#     Age = df.Age
#     Edu = df.EDUCYRS
#     YearsSinceDX = df.Years_since_DX_bl
#     Hand = df.HANDED
#
#     labels = ['HC','de novo PD', 'genetic PD (aff)', 'genetic PD (unaff)','SWEDD','Prodromal']
#     groups = [df['HC_amplified'] & bl,df['PD_denovo_amplified'] & bl,df['PD_genetic_amplified'] & bl,df['PD_genetic_amplified_preclinical'] & bl,df['SWEDD_amplified'] & bl,df['Prodromal_amplified'] & bl]
#     groups_bl = [g[bl] for g in groups]
#     lj = max([len(l) for l in labels])
#
#     print_demographics(Age,'Age',groups,labels)
#     print('  {0} - {1} ({2}) [Range: {3}-{4}]'.format('All'.ljust(lj+1),round(np.nanmean(Age[bl]),1),round(np.nanstd(Age[bl]),1),round(np.min(Age[bl]),1),round(np.max(Age[bl]),1)))
#     print_demographics(Edu,'Education',groups,labels)
#     print('  {0} - {1} ({2}) [Range: {3}-{4}]'.format('All'.ljust(lj+1),round(np.nanmean(Edu[bl]),1),round(np.nanstd(Edu[bl]),1),round(np.min(Edu[bl]),1),round(np.max(Edu[bl]),1)))
#     print_demographics(YearsSinceDX,'Years since PD DX',groups,labels)
#
#     print('Handedness: N right-handed, N left-handed, N other (Total)')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups[k]
#         print('  {0} - {1}, {2}, {3} ({4})'.format(l.ljust(lj+1),
#                                                    sum(Hand[g]==1),
#                                                    sum(Hand[g]==2),
#                                                    sum(Hand[g]==3),
#                                                    len(Hand[g])))
#     print('  {0} - {1}, {2}, {3} ({4})'.format('All'.ljust(lj+1),
#                                                sum(Hand[bl]==1),
#                                                sum(Hand[bl]==2),
#                                                sum(Hand[bl]==3),
#                                                len(Hand[bl])))
#
#     print('\n\nClinical scores')
#     MOCA = df.MCATOT
#     GDS  = df.GDS_TOT
#     RBDSQ = df.RBD_TOT
#     UPSIT = df.UPSIT_TOT
#     NP3 = df.NP3_TOT
#     SCOPA = df.SCOPAAUT_TOT
#     HY = df.NHY
#     print_demographics(MOCA,'MOCA',groups,labels)
#     print_demographics(GDS,'GDS',  groups,labels)
#     print_demographics(RBDSQ,'RBDSQ',groups,labels)
#     print_demographics(UPSIT,'UPSIT',groups,labels)
#     print_demographics(NP3,'NP3',groups,labels)
#     print_demographics(SCOPA,'SCOPA-AUT',groups,labels)
#     print_demographics(HY,'Hoehn & Yahr stage',groups,labels)
#
#     TD = (df.TD_PIGD_class=='TD')
#     PIGD = (df.TD_PIGD_class=='PIGD')
#     Indet = (df.TD_PIGD_class=='Indeterminate')
#     print('\nMotor subclass: Tremor or Posture/Gait')
#     print('TD - N')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(TD[g]==1)))
#     print('PIGD - N')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(PIGD[g]==1)))
#     print('Indeterminate - N')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(Indet[g]==1)))
#
#
#
#
#     #* FIXME: The following code might not work, after moving from a jupyter notebook (above has been checked)
#     #* Number of visits each
#     n_visits_unique = len(df.EVENT_ID.unique())
#     PATNO = df.PATNO.values
#     PATNO_u = np.unique(df.PATNO.values)
#
#     #* Raw numbers of visits
#     n_visits_PATNO = []
#     duration_first_to_last_visit_PATNO = []
#
#     clinical_markers = ['RBD_TOT','NP3_TOT','SCOPAAUT_TOT','GDS_TOT','MCATOT'] # 'UPSIT_TOT'
#     mri_markers = ['TIV']
#     csf_markers = ['CSF Alpha-synuclein']
#     plasma_markers = ['Triglycerides']
#     dat_markers = ['CAUDATE_L']
#     dti_markers = ['DTI_FA_ROI1_calc']
#
#     #* Visits per modality
#     n_visits_PATNO_MRI = []
#     n_visits_PATNO_CSF = []
#     n_visits_PATNO_plasma = []
#     n_visits_PATNO_DaTSCAN = []
#     n_visits_PATNO_DTI = []
#     n_visits_PATNO_clinical = []
#     n_visits_PATNO_allModalities = []
#
#     for k in range(len(PATNO_u)):
#         rowz = PATNO_u[k]==PATNO
#         n_visits_PATNO.append(sum(rowz))
#         duration_first_to_last_visit_PATNO.append(np.nanmax(df.Years_bl[rowz]))
#         #* Each modality
#         notNAN_MRI = np.all(~np.isnan(df[mri_markers][rowz]),axis=1)
#         notNAN_CSF = np.all(~np.isnan(df[csf_markers][rowz]),axis=1)
#         notNAN_plasma = np.all(~np.isnan(df[plasma_markers][rowz]),axis=1)
#         notNAN_DaTSCAN = np.all(~np.isnan(df[dat_markers][rowz]),axis=1)
#         notNAN_DTI = np.all(~np.isnan(df[dti_markers][rowz]),axis=1)
#         notNAN_clinical = np.all(~np.isnan(df[clinical_markers][rowz]),axis=1)
#         n_visits_PATNO_MRI.append(sum(notNAN_MRI))
#         n_visits_PATNO_CSF.append(sum(notNAN_CSF))
#         n_visits_PATNO_plasma.append(sum(notNAN_plasma))
#         n_visits_PATNO_DaTSCAN.append(sum(notNAN_DaTSCAN))
#         n_visits_PATNO_DTI.append(sum(notNAN_DTI))
#         n_visits_PATNO_clinical.append(sum(notNAN_clinical))
#         n_visits_PATNO_allModalities.append(sum(notNAN_MRI & notNAN_CSF & notNAN_plasma & notNAN_DaTSCAN & notNAN_DTI))
#
#     print('>>> Longitudinal data <<<')
#
#     print('   Visits per PATNO:\n     {0}-{1};\n     mean (std) = {2} ({3})\n'.format(
#         np.min(n_visits_PATNO_clinical),
#         np.max(n_visits_PATNO_clinical),
#         round(np.nanmean(n_visits_PATNO_clinical),1),
#         round(np.nanstd(n_visits_PATNO_clinical),1)))
#
#     print('=== Clinical visits ===')
#     x = np.array(n_visits_PATNO_clinical)
#     print_demographics(x,'Clinical',groups_bl,labels)
#     print('*N having >1 visit: ')
#     for k in range(len(groups_bl)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#
#     x = np.array(n_visits_PATNO_MRI)
#     print_demographics(x,'Number of visits (structural imaging)',groups_bl,labels)
#     print('*Number having longitudinal data (>1 visit): ')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#     x = np.array(n_visits_PATNO_CSF)
#     print_demographics(x,'Number of visits (CSF)',groups_bl,labels)
#     print('*Number having longitudinal data (>1 visit): ')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#     x = np.array(n_visits_PATNO_plasma)
#     print_demographics(x,'Number of visits (plasma)',groups_bl,labels)
#     print('*Number having longitudinal data (>1 visit): ')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#     x = np.array(n_visits_PATNO_DaTSCAN)
#     print_demographics(x,'Number of visits (DaTSCAN)',groups_bl,labels)
#     print('*Number having longitudinal data (>1 visit): ')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#     x = np.array(n_visits_PATNO_DTI)
#     print_demographics(x,'Number of visits (DTI)',groups_bl,labels)
#     print('*Number having longitudinal data (>1 visit): ')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#     x = np.array(n_visits_PATNO_allModalities)
#     print_demographics(x,'Number of visits (all modalities, complete data)',groups_bl,labels)
#     print('*Number having longitudinal data (>1 visit): ')
#     for k in range(len(groups)):
#         l = labels[k]
#         g = groups_bl[k]
#         print('  {0} - {1}'.format(l.ljust(lj+1),sum(x[g]>1)))
#
#     return 0
#



from scipy import special
def evaluate_GP_posterior(x_p,x_data,y_data,rho_sq,eta_sq,sigma_sq,
                          nSamplesFromGPPosterior = 1000,
                          plotGPPosterior = True,
                          CredibleIntervalLevel = 0.95):
    #* Observations - full kernel
    K = kernel_obs(np.sqrt(eta_sq),np.sqrt(rho_sq),np.sqrt(sigma_sq),x_data)
    #* Interpolation - signal only
    K_ss = kernel_pred(np.sqrt(eta_sq),np.sqrt(rho_sq),x_p,x_p)
    #* Covariance (observations & interpolation) - signal only
    K_s = kernel_pred(np.sqrt(eta_sq),np.sqrt(rho_sq),x_p,x_data)
    #* GP mean and covariance
    #* Covariance from fit
    y_post_mean = np.matmul(np.matmul(K_s,np.linalg.inv(K)),y_data)
    y_post_Sigma = (K_ss - np.matmul(np.matmul(K_s,np.linalg.inv(K)),K_s.transpose()))
    y_post_std = np.sqrt(np.diag(y_post_Sigma))
    #* Covariance from data - to calculate residuals
    K_data = K
    K_s_data = kernel_pred(np.sqrt(eta_sq),np.sqrt(rho_sq),x_data,x_data)
    y_post_mean_data = np.matmul(np.matmul(K_s_data,np.linalg.inv(K_data)),y_data)
    residuals = y_data - y_post_mean_data
    RMSE = np.sqrt(np.mean(residuals**2))
    # Numerical precision
    eps = np.finfo(float).eps
    ## 3. Sample from the posterior (multivariate Gaussian)
    stds = np.sqrt(2) * special.erfinv(CredibleIntervalLevel)
    #* Diagonalise the GP posterior covariance matrix
    Vals,Vecs = np.linalg.eig(y_post_Sigma)
    A = np.real(np.matmul(Vecs,np.diag(np.sqrt(Vals))))

    y_posterior_middle = y_post_mean
    y_posterior_upper = y_post_mean + stds*y_post_std
    y_posterior_lower = y_post_mean - stds*y_post_std

    #* Sample
    y_posterior_samples = np.tile(y_post_mean,(nSamplesFromGPPosterior,1)).transpose() + np.matmul(A,np.random.randn(len(y_post_mean),nSamplesFromGPPosterior))
    if np.abs(np.std(y_data)-1) < eps:
        y_posterior_samples = y_posterior_samples*np.std(y_data) + np.mean(y_data)

    return y_posterior_samples, y_posterior_middle, y_posterior_upper, y_posterior_lower, RMSE


#* Covariance matrices from kernels: @kernel_pred, @kernel_err, @kernel_obs
def kernel_pred(eta,rho,x_1,x_2):
    kp = eta**2*np.exp(-rho**2 * (np.tile(x_1,(len(x_2),1)).transpose() - np.tile(x_2,(len(x_1),1)))**2)
    return kp
def kernel_err(sigma,x_1):
    ke = sigma**2*np.eye(len(x_1))
    return ke
def kernel_obs(eta,rho,sigma,x_1):
    ko = kernel_pred(eta,rho,x_1,x_1) + kernel_err(sigma,x_1)
    return ko

