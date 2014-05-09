
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import collections as col
import statsmodels.api as sm


from statsmodels.formula.api import logit, probit, poisson
from scipy import stats
from scipy.stats import nbinom
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from mpl_toolkits.basemap import Basemap
from IPython.display import display
from IPython.display import HTML

# This code was written using Scipy 0.13.3, Numpy 1.6.1, Pandas 0.12.0, XLRD 0.9.2, Matplotlib Basemap 1.0.6, GEOS 3.3.3
# It may not be compatible with other module versions
# Code version 14 April 2014
# No known bugs

# Lines of code (underneath code comment and before assert statement) are taken from this project's main iPython Notebook

# Test that all conflicts starting prior to 1990 were removed
def test_1():
    #create sample dataset: start with all zeros and make 15 years pre-1990
    scad = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','location_precision'])
    scad.styr = 1995
    scad.styr[0:15] = 1987
    
    #code
    scad = scad[scad.styr>=1990]
    
    #test assert
    assert len(scad) == 7985

# Test that all conflicts that exceed one calendar year were removed
def test_2():
    #create sample dataset: start with all zeroes amd make 15 conflicts exceed one year
    scad = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','location_precision'])
    scad.styr = 1995
    scad.eyr = 1995
    scad.styr[0:15] = 1987
    
    #code
    scad = scad[scad.styr == scad.eyr]
    
    #test assert
    assert len(scad) == 7985  

# Test all duplicated rows were removed    
def test_3():
    #create sample dataset: start with zeroes and then remove all duplicates (all but first row are the same)
    scad = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','location_precision'])
    
    #code
    scad = scad.drop_duplicates(cols='eventid')
    
    #test assert
    assert len(scad) == 1

#Test code to determine the dominant religion
def test_4():
    #create sample dataset: all zeros with only one religion having any members for any given record
    wrp = pd.DataFrame(data=np.zeros((16, 18)), columns=['year', 'state', 'chrstgen', 'judgen', 'islmgen', 'budgen', 'zorogen', 'hindgen', 'sikhgen', 'shntgen', 'bahgen', 'taogen', 'jaingen', 'confgen', 'syncgen', 'anmgen', 'nonrelig', 'othrgen'])
    wrp.chrstgen[0]=100
    wrp.judgen[1]=100
    wrp.islmgen[2]=100
    wrp.budgen[3]=100
    wrp.zorogen[4]=100
    wrp.hindgen[5]=100
    wrp.sikhgen[6]=100
    wrp.shntgen[7]=100
    wrp.bahgen[8]=100
    wrp.taogen[9]=100
    wrp.jaingen[10]=100
    wrp.confgen[11]=100
    wrp.syncgen[12]=100
    wrp.anmgen[13]=100
    wrp.nonrelig[14]=100
    wrp.othrgen[15]=100
    wrp.year[15] = 700 #pass in a year that is larger than the largest religion (should be ignored)
    wrp.state[9]=19643 #pass in a state that is larger than the largest religion (should be ignored)
        
    #code
    yearState = wrp[['year', 'state']]
    wrp = wrp.ix[:,2:]
    wrp['domRel'] = wrp.idxmax(axis=1)
    wrp = pd.DataFrame(wrp.domRel)
    wrp=pd.merge(yearState, wrp, left_index=True, right_index=True)
    
    #test assert
    np.testing.assert_array_equal(wrp.domRel, np.array(['chrstgen', 'judgen', 'islmgen', 'budgen', 'zorogen', 'hindgen', 'sikhgen', 'shntgen', 'bahgen', 'taogen', 'jaingen', 'confgen', 'syncgen', 'anmgen', 'nonrelig', 'othrgen']) )

# Test setting values as NaN    
def test_5():
    #create sample dataset: start with zeroes, set 15 events as -99 (the code for missing)
    scad = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','location_precision'])
    scad.npart[0:15] = -99
    
    #code
    scad.npart = scad['npart'].astype(float)
    scad.npart[scad.npart == -99] = np.nan
    
    #test assert
    assert np.isnan(scad.npart).sum() == 15
    
    
# Test recoding of a string level variable as numeric codes (used for dominant religion)
def test_6():
    #create sample dataset: 
    scad = pd.DataFrame(data=np.repeat(('a','b','c','d'), 2000), columns=['domRel'])
    
    #code
    levels = dict([(val, i) for i, val in enumerate(set(scad.domRel))])
    scad['domRelNum'] = [levels[val] for val in scad.domRel]
    
    #test assert
    np.testing.assert_array_equal( scad['domRelNum'], np.repeat((0,2,1,3), 2000)) 
    
# Test coding of death/nodeath indicator variable
def test_7():
    #create sample dataset: start with zeroes, set 4752 events as 1 death
    #since death variable is 0/1, it should equal the coded death/nodeath indicator
    scad = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','location_precision'])
    scad.ndeath[1:4752] = 1
    
    #code
    scad['death01']=scad.ndeath
    scad.death01[scad.death01>0] = 1
    scad.death01[scad.ndeath == np.nan] = np.nan
    
    #test assert
    np.testing.assert_array_equal(scad.ndeath, scad.death01)

#test binning code in figure 2
def test_8():
    #create dummy data: all zeros but 1 in the other bins
    scad = pd.DataFrame(data=np.zeros((8000, 1)), columns=['ndeath'])
    for i in ((1,2,3,4,7,20,75, 150, 2000)):
        scad.ndeath[i] = i
    
    #code
    noNan = scad.ndeath[~np.isnan(scad.ndeath)]    
    counts = col.Counter(noNan)
    bins = np.arange(0, 10, 1)
    freqs = [0]*len(counts)
    j=0
    for i in np.unique(counts):
        freqs[j] = counts[i]
        j = j+1
    
    freqs = np.array(freqs)
    histFreqs = [0]*10
    c = np.unique(counts)
    histFreqs[0] = freqs[c==0]
    histFreqs[1] = freqs[c==1]
    histFreqs[2] = freqs[c==2]
    histFreqs[3] = freqs[c==3]
    histFreqs[4] = freqs[c==4]
    histFreqs[5] = np.sum(freqs[(c>=5) & (c<=10)])
    histFreqs[6] = np.sum(freqs[(c>=11) & (c<=50)])
    histFreqs[7] = np.sum(freqs[(c>=51) & (c<=100)])
    histFreqs[8] = np.sum(freqs[(c>=101) & (c<=200)])
    histFreqs[9] = np.sum(freqs[(c>=201) & (c<=5000)])
    
    #test assert
    np.testing.assert_array_equal(histFreqs, np.array((7991,1,1,1,1,1,1,1,1,1)) )    
    
#test latitude/longitude round and duplicate drop iused in n figure 6
def test_9():
    #create dummy data
    scad = pd.DataFrame(data=np.zeros((8000, 3)), columns=['latitude', 'longitude', 'death01'])
    scad.death01[0:4000] = 1
    
    #code
    noNan = scad[['longitude', 'latitude', 'death01']]
    noNan.longitude = np.round(noNan.longitude, decimals=0)
    noNan.latitude = np.round(noNan.latitude, decimals=0)
    noNan = noNan.drop_duplicates()
    
    #test assert
    assert len(noNan) == 2

#test cross validation doesn't pull the same rows
def test_10():
    #create dummy data: zeros with a 1:8000 counter in event id
    scad = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','location_precision'])
    scad.eventid = np.arange(1, 8001, 1)
        
    #code
    np.random.seed(13)
    crossVal = [False] * (len(scad))
    numTest = int(0.3*len(scad))
    crossVal[0:numTest] = [True] * numTest
    crossVal = np.random.permutation(crossVal)
    train = scad[~crossVal]
    test = scad[crossVal]
    
    #test assert
    assert len(np.intersect1d(train.eventid, test.eventid)) == 0
    
#test code to create individual dummy variables from categoricals
def test_11():
    #generate all zeroes and then three single instances of 1, 2, 3
    #dummies will drop the lowest case (here the 7997 zeroes) so we need to test for one '1' indicator in each of the 3 new variables for 1-3
    test = pd.DataFrame(data=np.zeros((8000, 42)), columns=['eventid','ccode','id','countryname','startdate','enddate','duration','stday','stmo','styr','eday','emo','eyr','etype','escalation','actor1','actor2','actor3','target1','target2','cgovtarget','rgovtarget','npart','ndeath','repress','elocal','ilocal','sublocal','locnum','gislocnum','issue1','issue2','issue3','issuenote','nsource','notes','coder','ACD_questionable','latitude','longitude','geo_comments','cinc'])
    test.etype[0:3] = np.arange(1,4,1)
    
    #code
    v = test[['ndeath', 'locnum', 'cgovtarget', 'etype', 'issue1', 'cinc']]
    v = v[~np.isnan(v).any(1)]
    dummies = pd.get_dummies(v.etype, prefix='etype')
    v = v.drop('etype', 1)
    v = v.join(dummies.ix[:, 1:])
    
    #test assert
    assert sum(v['etype_1.0'])==1
    assert sum(v['etype_2.0'])==1
    assert sum(v['etype_3.0'])==1
    
#test computing model prediction error
def test_12():
    #generate a model with a perfect fit (use OLS and pass a matrix of all 0).  prediction error should be 0
    #train/test split already tested; dummy generation already tested
    #only other untested things in this modeling process is the modeling itself, which is a packaged function and is assumed to have been tested by the developer
    v = pd.DataFrame(data=np.zeros((8000, 5)), columns=['ndeath', 'locnum', 'cgovtarget', 'etype', 'cinc'])
    v['intercept'] = 1.0
    modp2 = sm.OLS(v.ndeath, v[v.columns[1:]], missing='drop')
    modp2 = modp2.fit()
    
    #code
    ests = modp2.predict(v[v.columns[1:]])
    true = v[['ndeath']].values
    mspe2 = np.sum(np.square(true-ests))
    
    #test assert
    assert mspe2==0
#test creation of prediction accuracy chart (code for logit regression)
def test_13():
    #use the perfect fit model generated above
    v = pd.DataFrame(data=np.zeros((8000, 5)), columns=['ndeath', 'locnum', 'cgovtarget', 'etype', 'cinc'])
    v['intercept'] = 1.0
    modp2 = sm.OLS(v.ndeath, v[v.columns[1:]], missing='drop')
    modp2 = modp2.fit()
    
    #code
    ests = modp2.predict(v[v.columns[1:]])
    true = v[['ndeath']].values
    predAnalysis = [[0 for x in xrange(2)] for x in xrange(2)]
    for i in range(0,len(true),1):
        if true[[i]]==0 and ests[[i]]==0:
            predAnalysis[0][0] = predAnalysis[0][0] + 1
        elif true[[i]]==0 and ests[[i]]==1:
            predAnalysis[0][1] = predAnalysis[0][1] + 1
        elif true[[i]]==1 and ests[[i]]==0:
            predAnalysis[1][0] = predAnalysis[1][0] + 1
        elif true[[i]]==1 and ests[[i]]==1:
            predAnalysis[1][1] = predAnalysis[1][1] + 1

    #test assert (all are predicted correctly as zeros, so we know what the matrix should look like)
    np.testing.assert_array_equal( predAnalysis, np.array([[8000,0],[0,0]]) )
    