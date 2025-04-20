# -*- coding: utf-8 -*-

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''MANIFEST'''
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""

MPhEV
Metastatistical Physically-based Extreme Value

author: pietro devò
e-mail: pietro.devo@phd.unipd.it

            .==,_
           .===,_`\
         .====,_ ` \      .====,__
   ---     .==-,`~. \           `:`.__,
    ---      `~~=-.  \           /^^^
      ---       `~~=. \         /
                   `~. \       /
                     ~. \____./
                       `.=====)
                    ___.--~~~--.__
          ___\.--~~~              ~~~---.._|/
          ~~~"                             /

"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''LIBRARIES'''
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""layer 0"""

import os
import math
import pickle
import random
import functools
import itertools

""""layer 1"""

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sb
import pathlib as pl
import datetime as dt
import geopandas as gp
import matplotlib as mp

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''PARAMETERS'''
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

mp.pyplot.rc('text', usetex=True)
mp.pyplot.rc('font', family="Computer Modern")
mp.pyplot.rc('text.latex', preamble=r'\usepackage{xfrac}')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""FUNCTIONS"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def data_name(lst):
    """data name
    """

    if not isinstance(lst,list):
        lst = [lst]

    # removing nones
    lst = [l for l in lst if l is not None]

    for l in range(len(lst)):
        if not isinstance(lst[l],list):
            lst[l] = [lst[l]]

    # extract
    lst = [j for i in lst for j in i]

    # naming
    nam = '_'.join([str(l) for l in lst])

    return nam

def data_path(sub,fil,ext):
    """data path
    """

    # working
    cwd = os.getcwd()

    # folder
    fol = os.path.abspath(os.path.join(cwd,os.pardir))

    if sub is not None:
        fol = '%s/%s'%(fol,sub)
        pl.Path(fol).mkdir(parents=True,exist_ok=True)

    if ext is None:
        pth = '%s/%s'%(fol,fil)
    else:
        pth = '%s/%s.%s'%(fol,fil,ext)

    return pth

def data_remove(sub,fil):
    """"data removing
    """

    if sub is None:
        sub = 'data'

    # settings
    ext = 'pckl'

    # path
    pth = data_path(sub,fil,ext)

    # removing
    os.remove(pth)

def data_load(sub,fil):
    """data loading
    """

    if sub is None:
        sub = 'data'

    # settings
    ext = 'pckl'

    # path
    pth = data_path(sub,fil,ext)

    # locking
    dmp = open(pth,'rb')

    try:
        obj = pd.read_pickle(dmp)
    except:
        obj = pickle.load(dmp)

    return obj

def data_save(sub,fil,obj):
    """"data saving
    """

    if sub is None:
        sub = 'data'

    # settings
    ext = 'pckl'

    # path
    pth = data_path(sub,fil,ext)

    # locking
    dmp = open(pth,'wb')

    # saving
    pickle.dump(obj,dmp)

def config_list(sub,fil,ext):
    """"config list
    """

    # path
    pth = data_path(sub,fil,ext)

    # reading
    out = open(pth).read().split('\n')

    # deleting
    del out[-1]

    return out

def config_extract(dat,lbl):
    """config extracting
    """

    # initialize
    out = []

    for line in dat:

        if lbl in line:

            # object
            obj = list(line.split(" "))

            # indexing
            idx = obj.index(lbl)

            # output
            out.append(obj[idx+1])

    return out

def config_format(dat,typ):
    """config formatting
    """

    if isinstance(dat,list):
        out = [typ(i) for i in dat]

    else:
        out = typ(dat)

    return out

def config_function(sub,fil,ext,lbl,typ):
    """config function
    """

    if sub is None:
        sub = 'config'

    # list
    cfg = config_list(sub,fil,ext)

    if lbl is not None:
        cfg = config_extract(cfg,lbl)

    if typ is not None:
        cfg = config_format(cfg,typ)

    if len(cfg) == 1:
        cfg = cfg[0]

    return cfg

def dataframe_meta():
    """indexed meta dataframe
    """

    # settings
    sub = None
    ext = 'csv'
    idx = 1

    # name
    nam = 'watersheds'

    # path
    pth = data_path(sub,nam,ext)

    # table creation
    dtf = pd.read_csv(pth,index_col=idx)

    return dtf

def dataframe_obs():
    """indexed observations dataframe
    """

    # observations list
    idn = list_idn(None,None)

    # variables
    var = ['P','Q']

    # table creation
    dtf = table_single(None,'series')

    for i in idn:

        try:

            # temporary dataframe
            tmp              = table_single(i,'series')
            tmp['series'][i] = table_series(i,var)

            # concatenate
            dtf = pd.concat([dtf,tmp])

            for v in var:
                dtf.loc[i,r'%s_min'%(v)] = np.nanmin(dtf.loc[i,'series'][v].to_numpy())
                dtf.loc[i,r'%s_max'%(v)] = np.nanmax(dtf.loc[i,'series'][v].to_numpy())
                dtf.loc[i,r'%s_avg'%(v)] = np.nanmean(dtf.loc[i,'series'][v].to_numpy())

        except:

            # missing idn
            fun_print(None,'Missing:',i)

    return dtf

def geo_dataframe(crd):
    """geographical dataframe
    """

    # settings
    sub = 'geo'
    ext = 'json'

    if crd is None:
        crd = 'EPSG:3035'

    # names
    nam_rg = 'NUTS_RG_01M_2021_3035'
    nam_bn = 'NUTS_BN_01M_2021_3035'
    nam_lb = 'NUTS_LB_2021_3035'

    # paths
    pth_rg = data_path(sub,nam_rg,ext)
    pth_bn = data_path(sub,nam_bn,ext)
    pth_lb = data_path(sub,nam_lb,ext)

    # geodataframes
    gdf_rg = gp.read_file(pth_rg)
    gdf_bn = gp.read_file(pth_bn)
    gdf_lb = gp.read_file(pth_lb)

    # coordinates system
    gdf_rg.crs = crd
    gdf_bn.crs = crd
    gdf_lb.crs = crd

    return gdf_rg,gdf_bn,gdf_lb

def geo_country(gdf,cnt):
    """geographical country
    """

    # selecting country
    gdf = gdf[gdf.CNTR_CODE == cnt]

    return gdf

def cod_generate(num):
    """codification generation
    """

    # products
    pro = itertools.product('01',repeat=num)

    # initialize
    lst = []

    for p in pro:
        lst.append(''.join(p))

    return lst

def cod_string(cod):
    """codification string
    """

    # stringing
    stn = ''.join([str(int(i)) for i in cod])

    return stn

def cod_flag(cod):
    """codification flag
    """

    if not isinstance(cod,str):
        cod = str(cod)

    # flagging
    flg = [bool(i) for i in [int(j) for j in cod]]

    return flg

def dic_codification(nam):
    """codification dictionary
    """

    # initialize
    dic = {}

    # analyses
    dic['analyses'] = ['analyses']

    # model
    dic['model'] = cod_generate(4)

    # framework
    dic['framework'] = dic['analyses']+dic['model']

    # parameter
    dic['p1_0'] = ['0000','0001','0010','0100','0011','0101','0110','0111']
    dic['p1_1'] = ['1000','1001','1010','1100','1011','1101','1110','1111']
    dic['p2_0'] = ['0000','0001','0010','1000','0011','1001','1010','1011']
    dic['p2_1'] = ['0100','0101','0110','1100','0111','1101','1110','1111']
    dic['p3_0'] = ['0000','0001','0100','1000','0101','1001','1100','1101']
    dic['p3_1'] = ['0010','0011','0110','1010','0111','1011','1110','1111']
    dic['p4_0'] = ['0000','0010','0100','1000','0110','1010','1100','1110']
    dic['p4_1'] = ['0001','0011','0101','1001','0111','1011','1101','1111']

    # number
    dic['0p'] = ['0000']
    dic['1p'] = ['1000','0100','0010','0001']
    dic['2p'] = ['1100','1010','0110','0101','0011','1001']
    dic['3p'] = ['0111','1011','1101','1110']
    dic['4p'] = ['1111']

    # reference
    dic['1p_r'] = dic['0p']+['1000','0100','0010','0001']+dic['4p']
    dic['2p_r'] = dic['0p']+['1100','1010','0110','0101','0011','1001']+dic['4p']
    dic['3p_r'] = dic['0p']+['0111','1011','1101','1110']+dic['4p']

    # group
    dic['2p_0_sx'] = dic['0p']+['0010','0001','0011']+dic['4p']
    dic['2p_0_dx'] = dic['0p']+['1000','0100','1100']+dic['4p']
    dic['2p_1_sx'] = dic['0p']+['1100','1110','1101']+dic['4p']
    dic['2p_1_dx'] = dic['0p']+['0011','1011','0111']+dic['4p']

    # set
    dic['hybrid']      = dic['0p']+['0011','1100']+dic['4p']
    dic['estimation']  = [x for x in dic['model'] if x not in set(dic['4p'])]
    dic['calibration'] = [x for x in dic['model'] if x not in set(dic['0p'])]

    if nam is not None:
        dic = dic[nam]

    return dic

def dic_period(nam):
    """period dictionary
    """

    # initialize
    dic = {}

    # periods
    dic['spring'] = {'day':92,'month':[3,4,5]}
    dic['summer'] = {'day':92,'month':[6,7,8]}
    dic['autumn'] = {'day':91,'month':[9,10,11]}
    dic['winter'] = {'day':90,'month':[12,1,2]}
    dic['year']   = {'day':365,'month':[1,2,3,4,5,6,7,8,9,10,11,12]}

    if nam is not None:
        dic = dic[nam]

    return dic

def dic_parameters(nam):
    """parameters dictionary
    """

    # initialize
    dic = {}

    # parameters
    dic[0] = {'symbol':'alpha','reference':1,'delta':1,'min':0,'max':25}
    dic[1] = {'symbol':'labda','reference':0.5,'delta':0.5,'min':0,'max':1}
    dic[2] = {'symbol':'a','reference':2,'delta':2,'min':1,'max':10}
    dic[3] = {'symbol':'k','reference':0.1,'delta':10,'min':0,'max':250}

    if nam is not None:
        dic = dic[nam]

    return dic

def dic_solvers(nam):
    """solvers dictionary
    """

    # initialize
    dic = {}

    # parameters
    dic['Nelder-Mead']  = {'iterations':5e+3,'tolerance':1e-12}
    dic['L-BFGS-B']     = {'iterations':5e+3,'tolerance':1e-12}
    dic['TNC']          = {'iterations':5e+3,'tolerance':1e-12}
    dic['SLSQP']        = {'iterations':5e+3,'tolerance':1e-12}
    dic['Powell']       = {'iterations':5e+3,'tolerance':1e-12}
    dic['trust-constr'] = {'iterations':5e+3,'tolerance':1e-12}

    if nam is not None:
        dic = dic[nam]

    return dic

def dic_plot(nam):
    """plot dictionary
    """

    # initialize
    dic = {}

    # nuls
    dic['nul'] = {'type':None}

    # scatters
    dic['sct']     = {'type':'scatter','function':{'palette':None,'marker':None,'s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_vs']  = {'type':'scatter','function':{'palette':None,'marker':None,'s':10,'c':'silver','linewidth':None,'linestyle':'-','edgecolor':None},'format':{'aspect':'equal'},'line':{'k':'bisector'}}
    dic['sct_c1']  = {'type':'scatter','function':{'palette':'Set1','marker':None,'s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_c2']  = {'type':'scatter','function':{'palette':'Set2','marker':None,'s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_c3']  = {'type':'scatter','function':{'palette':'Set3','marker':None,'s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_s1']  = {'type':'scatter','function':{'palette':None,'marker':'o','s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_s2']  = {'type':'scatter','function':{'palette':None,'marker':'x','s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_s3']  = {'type':'scatter','function':{'palette':None,'marker':'w','s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None}}
    dic['sct_log'] = {'type':'scatter','function':{'palette':None,'marker':None,'s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None},'format':{'xscale':'log','yscale':'log'}}
    dic['sct_ffc'] = {'type':'scatter','function':{'palette':None,'marker':None,'s':10,'c':None,'linewidth':None,'linestyle':'-','edgecolor':None},'format':{'xscale':'log','yscale':'linear','xlim':[0.9,200],'ylim':[-1,None]}}

    # lines
    dic['lin']     = {'type':'line','function':{'palette':None,'linestyle':None,'linewidth':1}}
    dic['lin_c1']  = {'type':'line','function':{'palette':'Set1','linestyle':None,'linewidth':1}}
    dic['lin_c2']  = {'type':'line','function':{'palette':'Set2','linestyle':None,'linewidth':1}}
    dic['lin_c3']  = {'type':'line','function':{'palette':'Set3','linestyle':None,'linewidth':1}}
    dic['lin_s1']  = {'type':'line','function':{'palette':None,'linestyle':'solid','linewidth':1}}
    dic['lin_s2']  = {'type':'line','function':{'palette':None,'linestyle':'dashed','linewidth':1}}
    dic['lin_s3']  = {'type':'line','function':{'palette':None,'linestyle':'dotted','linewidth':1}}
    dic['lin_log'] = {'type':'line','function':{'palette':None,'linestyle':None,'linewidth':1},'format':{'xscale':'log','yscale':'log'}}
    dic['lin_ffc'] = {'type':'line','function':{'palette':None,'linestyle':None,'linewidth':1},'format':{'xlabel':'RP','xscale':'log','yscale':'linear','xlim':[0.9,200],'ylim':[-1,None]}}

    # boxes
    dic['box']       = {'type':'box','function':{'palette':None,'showfliers':False,'width':0.8,'dodge':True,'fliersize':0,'linewidth':0.5}}
    dic['box_log']   = {'type':'box','function':{'palette':None,'showfliers':False,'width':0.8,'dodge':True,'fliersize':0,'linewidth':0.5},'format':{'xscale':'log','yscale':'log'}}
    dic['box_err_v'] = {'type':'box','function':{'palette':None,'showfliers':False,'width':0.8,'dodge':True,'fliersize':0,'linewidth':0.5},'format':{'ylim':[-1,None],'orient':'v','axhliney':0,'axhlinels':'--','axhlinelw':0.5,'axhlinec':'black'}}
    dic['box_err_h'] = {'type':'box','function':{'palette':None,'showfliers':False,'width':0.8,'dodge':True,'fliersize':0,'linewidth':0.5},'format':{'xlim':[-1,None],'orient':'h','axvlinex':0,'axvlinels':'--','axvlinelw':0.5,'axvlinec':'black'}}

    # strips
    dic['str']     = {'type':'strip','function':{'palette':None,'dodge':True,'size':1,'linewidth':0.5,'alpha':0.25}}
    dic['str']     = {'type':'strip','function':{'palette':None,'dodge':True,'size':1,'linewidth':0.5,'alpha':0.25},'format':{'xscale':'log','yscale':'log'}}
    dic['str_err'] = {'type':'strip','function':{'palette':None,'dodge':True,'size':1,'linewidth':0.5,'alpha':0.25},'format':{'ylim':[-1,None],'orient':None,'axhliney':0,'axhlinels':'--','axhlinelw':0.5,'axhlinec':'black'}}

    # violins
    dic['vio']     = {'type':'violin','function':{'palette':None,'linewidth':0.5,'cut':2}}
    dic['vio_log'] = {'type':'violin','function':{'palette':None,'linewidth':0.5,'cut':2},'format':{'xscale':'log','yscale':'log','ylim':[-1,None],'axhliney':0,'axhlinels':'--','axhlinelw':0.5,'axhlinec':'black'}}
    dic['vio_err'] = {'type':'violin','function':{'palette':None,'linewidth':0.5,'cut':0},'format':{'ylim':[-1,None],'axhliney':0,'axhlinels':'--','axhlinelw':0.5,'axhlinec':'black'}}

    # histograms
    dic['hst']     = {'type':'hist','function':{'palette':None,'bins':'auto','binwidth':None,'discrete':None,'stat':'count'}}
    dic['hst_vs']  = {'type':'hist','function':{'palette':None,'bins':'auto','binwidth':None,'discrete':None,'stat':'count','log_scale':True},'format':{'aspect':'equal'},'line':{'k':1}}
    dic['hst_log'] = {'type':'hist','function':{'palette':None,'bins':'auto','binwidth':None,'discrete':None},'format':{'xscale':'log','yscale':'log'}}
    dic['hst_prb'] = {'type':'hist','function':{'palette':None,'bins':'auto','binwidth':None,'discrete':None,'stat':'probability'},'format':{'ylim':[0,1]}}
    dic['hst_ffc'] = {'type':'hist','function':{'palette':None,'bins':'auto','binwidth':None,'discrete':None},'format':{'xscale':'log','yscale':'linear','xlim':[0.9,200],'ylim':[-1,None]}}

    # kernel
    dic['kde']    = {'type':'kde','function':{'palette':None,'fill':True,'log_scale':False,'levels':5,'cut':0}}
    dic['kde_vs'] = {'type':'kde','function':{'palette':None,'fill':True,'log_scale':True,'levels':5,'cut':0},'format':{'aspect':'equal'},'line':{'k':1}}

    # joints
    dic['jnt'] = {'type':'joint','function':{'palette':None,'kind':'hist'}}

    # heats
    dic['hea'] = {'type':'heat','function':{'annot':None,'linewidhts':None,'linecolor':None,'cbar':True,'square':True,'mask':None},'format':{'xticksrotation':90}}

    # multiplot
    dic['mlt_mod'] = {'type':[dic['sct'],dic['sct']],'format':{'xlabel':'observations','ylabel':'model'}}
    dic['mlt_ser'] = {'type':[dic['sct'],dic['lin']],'format':{'xlabel':'date','ylabel':'streamflow'}}
    dic['mlt_ext'] = {'type':[dic['sct_c1'],dic['sct_c2'],dic['lin_s1'],dic['lin_s2']],'format':{'xlabel':'q','ylabel':'probability','xscale':'log','yscale':'log','ylim':[1e-3,2]}}
    dic['mlt_prb'] = {'type':[dic['sct'],dic['lin']],'format':{'xlabel':'q','ylabel':'probability','xscale':'log','yscale':'log','ylim':[1e-3,2]}}
    dic['mlt_ffc'] = {'type':[dic['sct'],dic['lin']],'format':{'xlabel':'RP [years]','ylabel':'q [-]','xscale':'log','yscale':'linear','xlim':[0.9,200],'ylim':[-1,None]}}
    dic['mlt_err'] = {'type':[dic['box'],dic['str']],'format':{'ylim':[-1,None],'orient':None,'axhliney':0,'axhlinels':'--','axhlinelw':0.5,'axhlinec':'black'}}

    if nam is not None:
        dic = dic[nam]

    return dic

def dic_geo(nam):
    """geo dictionary
    """

    # initialize
    dic = {}

    # select
    dic['select'] = {'label':'basins','marker':'o','s':25,'c':'lightgray'}

    # tag
    dic['lw'] = {'label':'underestimation','marker':'X','s':100,'c':'blue'}
    dic['up'] = {'label':'overestimation','marker':'X','s':100,'c':'red'}
    dic['ok'] = {'label':'ok','marker':'X','s':100,'c':'green'}
    dic['ko'] = {'label':'ko','marker':'X','s':100,'c':'black'}

    if nam is not None:
        dic = dic[nam]

    return dic

"""list of codifications
"""

def list_cod(nam):

    if nam is None:
        nam = 'framework'

    # codification
    cod = dic_codification(nam)

    return cod

def list_yea(lw,up):
    """list of years
    """

    if lw is None:
        lw = 1900

    if up is None:
        up = 2100

    # years
    yea = list(range(lw,up+1))

    return yea

def list_idn(lw,up):
    """list of idns
    """

    # settings
    sub = None
    ext = 'txt'
    idx = None

    # name
    nam = 'watersheds'

    # path
    pth = data_path(sub,nam,ext)

    # dataframe
    dtf = pd.read_csv(pth,index_col=idx)

    # array
    arr = list(dtf.loc[:,'ID'].to_numpy())

    if lw is None:
        lw = np.min(arr)

    if up is None:
        up = np.max(arr)

    # idn
    idn = [i for i in arr if i >= lw and i <= up]

    return idn

def list_prd(typ):
    """list of periods
    """

    # dictionary
    dic = dic_period(None)

    # index
    idx = dic.keys()

    if typ is None:
        prd = [i for i in idx]

    elif typ == 'year':
        prd = [i for i in idx if i == 'year']

    elif typ == 'season':
        prd = [i for i in idx if i == 'spring' or i == 'summer' or i == 'autumn' or i == 'winter']

    elif typ == 'spring' or typ == 'summer' or typ == 'autumn' or typ == 'winter' :
        prd = [typ]

    return prd

def select_year(dtf,yea):
    """reducing the table to selected year(s)"""

    if not isinstance(yea,list):
        yea = [yea]

    # work table
    dtf = dtf.loc[dtf['year'].isin(yea)]
    dtf.reset_index(drop=True,inplace=True)

    return dtf

def select_period(dtf,prd):
    """reducing the table to selected period(s)
    """

    if not isinstance(prd,list):
        prd = [prd]

    # dictionary
    dic = dic_period(None)

    # initialize
    sel = []

    for p in prd:
        mon = dic[p]['month']

        for m in mon:
            sel.append(m)

    # work table
    dtf = dtf.loc[dtf['month'].isin(sel)]
    dtf.reset_index(drop=True,inplace=True)

    return dtf

def select_range(dtf,ttl,rng):
    """reducing the table to values range
    """

    if not isinstance(rng,list):
        rng = [rng,rng]

    if ttl is None:
        ttl = dtf.select_dtypes(include=np.number).columns.tolist()

    if not isinstance(ttl,list):
        ttl = [ttl]

    for t in ttl:
        dtf = dtf.loc[(dtf.loc[:,t] >= rng[0]) & (dtf.loc[:,t] <= rng[1])]

    # cleaning
    dtf = table_clean(dtf,None)

    return dtf

def table_single(idn,ttl):
    """single-indexed table
    """

    if not isinstance(idn,list):
        if idn is None:
            idn = []
        else:
            idn = [idn]

    if not isinstance(ttl,list):
        if ttl is None:
            ttl = []
        else:
            ttl = [ttl]

    # table creation
    dtf = pd.DataFrame(index=idn,columns=ttl)

    # indexing
    dtf.index.name = 'ID'

    return dtf

def table_multi(idn,prd,cod,ttl):
    """multi-indexed table
    """

    if not isinstance(idn,list):
        if idn is None:
            idn = []
        else:
            idn = [idn]

    if not isinstance(prd,list):
        if prd is None:
            prd = []
        else:
            prd = [prd]

    if not isinstance(cod,list):
        if cod is None:
            cod = []
        else:
            cod = [cod]

    if not isinstance(ttl,list):
        if ttl is None:
            ttl = []
        else:
            ttl = [ttl]

    # indexing
    idx = pd.MultiIndex.from_product([idn,prd,cod],names=['ID','period','codification'])

    # table creation
    dtf = pd.DataFrame('-',idx,ttl)

    return dtf

def table_clean(dtf,col):
    """table clean
    """

    # replacing none
    dtf = dtf.replace(to_replace='None',value=np.nan)

    if col is None:
        dtf = dtf.dropna()

    else:
        dtf = dtf.dropna(how='all',subset=col)

    return dtf

def table_create(dtf,col):
    """table column create
    """

    if not isinstance(col,list):
        col = [col]

    for c in col:
        dtf.loc[:,c] = np.nan
        dtf.loc[:,c] = dtf[c].astype(object)

    return dtf

def table_remove(dtf,col):
    """table column remove
    """

    if not isinstance(col,list):
        col = [col]

    for c in col:
        dtf = dtf.drop(columns=c)

    return dtf

def table_rename(dtf,old,new):
    """table column rename
    """

    if not isinstance(old,list):
        old = [old]

    if not isinstance(new,list):
        new = [new]

    # size
    siz = min([len(old),len(new)])

    for i in range(siz):
        dtf = dtf.rename(columns={old[i]:new[i]})

    return dtf

def table_reduce(dtf,idx):
    """table reduction
    """

    if isinstance(idx,list):

        # size
        siz = len(idx)

        for i in range(siz):

            if idx[i] is None:
                idx[i] = dtf.index.get_level_values(i).drop_duplicates().tolist()

            if not isinstance(idx[i],list):
                idx[i] = [idx[i]]

        # index
        idx = pd.MultiIndex.from_product(idx).intersection(dtf.index)

    # dataframe
    dtf = dtf.loc[idx]

    return dtf

def table_object(dtf,ttl,obj):
    """table object
    """

    # column
    dtf = table_create(dtf,ttl)

    # idx
    idx = obj.index.get_level_values(0).drop_duplicates()

    for i in idx:

        try:
            dtf[ttl][i] = obj.loc[i]

        except:
            pass

    return dtf

def table_work(dtb,idn):
    """work table
    """

    # titles
    ttl = ['date','precipitation','streamflow','day','month','year','timestamp']

    # table creation
    wtb = pd.DataFrame(columns=ttl)

    # dates
    wtb.loc[:,ttl[0]] = dtb.loc[idn,'OBS'].loc['series'].index.values

    # precipitation/streamflow
    wtb.loc[:,ttl[1]] = dtb.loc[idn,'OBS'].loc['series']['P'].to_numpy()
    wtb.loc[:,ttl[2]] = dtb.loc[idn,'OBS'].loc['series']['Q'].to_numpy()

    # day/month/year
    wtb.loc[:,ttl[3]] = pd.DatetimeIndex(wtb.loc[:,'date']).day
    wtb.loc[:,ttl[4]] = pd.DatetimeIndex(wtb.loc[:,'date']).month
    wtb.loc[:,ttl[5]] = pd.DatetimeIndex(wtb.loc[:,'date']).year

    # timestamps
    wtb.loc[:,ttl[6]] = wtb.date.apply(pd.Timestamp.toordinal)

    return wtb

def table_adjust(wtb,ttl,adj):
    """adjusted table
    """

    if not isinstance(ttl,list):
        ttl [ttl]

    if not isinstance(adj,list):
        adj [adj]

    # length
    l = min(len(ttl),len(adj))

    for i in range(l):

        # unit adjustments
        wtb[ttl[i]] = wtb[ttl[i]]*adj[i]

    return wtb

def table_select(wtb,yea,prd):
    """selected table
    """

    # selecting the year
    wtb = select_year(wtb,yea)

    # selecting the period
    wtb = select_period(wtb,prd)

    if prd == 'winter':

        # december adjustments
        wtb['year'].loc[wtb['month'] == 12] = wtb['year'].loc[wtb['month'] == 12]+1

    return wtb

def table_series(idn,var):
    """series table
    """

    # settings
    sub = 'series'
    ext = 'txt'
    dlm = ' '
    idx = 0

    # generating dataframe
    dtf = pd.DataFrame()

    for v in var:

        # name
        nam = r'%s_%s'%(idn,v)

        # path
        pth = data_path(sub,nam,ext)

        # data
        dtf = dtf.join(pd.read_csv(pth,delimiter=dlm,index_col=idx,parse_dates=True,header=0,names=[v]),how='outer')

    # indexing
    dtf.index.name = 'date'

    return dtf

def extract_data(dtf,i,obj,col,fun,nam):
    """extracting data
    """

    # labelling
    lbl = list(range(len(obj)))

    # adding entry
    dtf.loc[i,lbl] = '-'

    try:
        dtf.loc[i,lbl] = np.asarray(obj,dtype=object)
    except:
        dtf.loc[i,lbl] = pd.Series(obj,index=lbl)

    return dtf

def extract_value(dtf,i,obj,col,fun,nam):
    """extracting value
    """

    if fun is None:
        dtf.at[i,'value'] = obj
    else:
        dtf.at[i,'value'] = fun_stats(obj,fun,0)

    return dtf

def extract_array(dtf,i,obj,col,fun,nam):
    """extracting array
    """

    if col is not None:
        obj = obj[col]

    if fun is not None:
        obj = fun_stats(obj,fun,0)

    try:
        lbl = list(range(obj.shape[0]))
    except:
        lbl = 'value'

    if fun is None:
        dtf.loc[i,lbl] = obj
    else:
        dtf.at[i,'value'] = fun_stats(obj,fun,0)

    return dtf

def extract_dataframe(dtf,i,obj,col,fun,nam):
    """extracting array
    """

    if col is not None:
        obj = obj.loc[:,col]

    if fun is not None:

        # labelling
        lbl = obj.columns.tolist()

        # object function
        obj = np.asarray(obj)
        obj = fun_stats(obj,fun,0)

        # adding entry
        dtf.loc[i,lbl] = obj

    else:

        # object generation
        obj.loc[:,nam] = i
        obj.set_index(nam,inplace=True)

        # adding entry
        dtf = pd.concat([dtf,obj],axis=0,ignore_index=False,join='outer')

    return dtf

def plot_figure(dic):
    """plot figure
    """

    if dic is None:
        dic = {}

    # dictionary
    dic = dic.copy()

    if 'nrows' not in dic:
        dic['nrows'] = 1
    if 'ncols' not in dic:
        dic['ncols'] = 1

    if 'figsize' not in dic:
        dic['figsize'] = None

    if dic['figsize'] is not None:

        # constant
        c = 2.54

        # scaling
        x = dic['figsize'][0]/c
        y = dic['figsize'][1]/c

        # updating
        dic['figsize'] = (x,y)

    if 'layout' not in dic:
        dic['layout'] = None

    if 'gridspec' not in dic:
        dic['gridspec'] = None

    if 'sharex' not in dic:
        dic['sharex'] = None
    if 'sharey' not in dic:
        dic['sharey'] = None

    # figure
    fig,axs = mp.pyplot.subplots(nrows=dic['nrows'],ncols=dic['ncols'],figsize=dic['figsize'],layout=dic['layout'],sharex=dic['sharex'],sharey=dic['sharey'],gridspec_kw=dic['gridspec'])

    if 'title' in dic:
        if dic['title'] is not None:
            if 'titlex' not in dic:
                dic['titlex'] = 0.500
            if 'titley' not in dic:
                dic['titley'] = 0.995
            if 'titlesize' not in dic:
                dic['titlesize'] = 10
            if 'titleweight' not in dic:
                dic['titleweight'] =  None
            fig.suptitle(t=dic['title'],x=dic['titlex'],y=dic['titley'],fontsize=dic['titlesize'],weight=dic['titleweight'])

    if 'text' in dic:
        if isinstance(dic['text'],dict):
            for k,v in dic['text'].items():
                fig.text(**v)

    return fig,axs

def plot_function(dic,axs,dtf,lbl,hue):
    """plot function
    """

    # dictionary
    dic = dic.copy()

    if isinstance(hue,list):
        hue = dtf[hue.copy()].apply(tuple,axis=1)

    if dic['type'] == 'scatter':
        sb.scatterplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'line':
        sb.lineplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'box':
        sb.boxplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'strip':
        sb.stripplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'violin':
        sb.violinplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'hist':
        sb.histplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'kde':
        sb.kdeplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'joint':
        sb.jointplot(ax=axs,data=dtf,x=lbl[0],y=lbl[1],hue=hue,**dic['function'])

    if dic['type'] == 'heat':
        sb.heatmap(ax=axs,data=dtf,**dic['function'])

def plot_line(dic,axs,dtf,lbl):
    """plot line
    """

    if 'ls' not in dic:
        dic['ls'] = '-'
    if 'lw' not in dic:
        dic['lw'] = 1.0
    if 'c' not in dic:
        dic['c'] = 'k'

    if dic['k'] == 'bisector':
        axs.axline((0, 0), slope=1, ls=dic['ls'],lw=dic['lw'],c=dic['c'],zorder=1)

    elif isinstance(dic['k'],(int,float)):
        # data
        arr = dtf.loc[:,lbl[0]]

        # values
        x_min = 10**(math.floor(math.log(np.nanmin(arr),10))-10)
        x_max = 10**(math.floor(math.log(np.nanmax(arr),10))+10)

        # function
        fun = lambda x,k: x*k

        # values
        x = np.array([x_min,x_max])
        y = fun(x,dic['k'])

        # plot
        axs.plot(x,y,ls=dic['ls'],lw=dic['lw'],c=dic['c'],zorder=1)

def plot_distribution(dic,axs,dtf,lbl):
    """plot distribution
    """

    # data
    arr = dtf.loc[:,lbl[0]]

    # values
    x_min = np.nanmin(arr)
    x_max = np.nanmax(arr)
    x_avg = np.nanmean(arr)
    x_dlt = abs(x_min-x_max)

    if dic['type'] =='normal':
        x = np.arange(x_min-x_dlt,x_max+x_dlt,1e-1)
        y = sp.stats.norm.pdf(x,loc=0,scale=1)

    if dic['type'] =='poisson':
        x = np.arange(0,x_max+x_dlt,1e-1)
        y = sp.stats.poisson.pmf(x,x_avg,loc=0)

    if 'ls' not in dic:
        dic['ls'] = '-'
    if 'lw' not in dic:
        dic['lw'] = 0.5
    if 'c' not in dic:
        dic['c'] = 'k'

    # plot
    axs.plot(x,y,ls=dic['ls'],lw=dic['lw'],c=dic['c'])

def plot_format(dic,axs):
    """plot format
    """

    if 'title' in dic:
        if dic['title'] is not None:
            if 'titlesize' not in dic:
                dic['titlesize'] = 10
            if 'titleweight' not in dic:
                dic['titleweight'] = None
            axs.set_title(dic['title'],fontsize=dic['titlesize'],weight=dic['titleweight'])

    if 'xlabel' in dic or 'xlabelfontsize' in dic or 'xlabelrotation' in dic:
        if 'xlabel' not in dic:
            dic['xlabel'] = axs.get_xlabel()
        if 'xlabelfontsize' not in dic:
            dic['xlabelfontsize'] = 10
        if 'xlabelrotation' not in dic:
            dic['xlabelrotation'] = 0
        axs.set_xlabel(dic['xlabel'],fontsize=dic['xlabelfontsize'],rotation=dic['xlabelrotation'])

    if 'ylabel' in dic or 'ylabelfontsize' in dic or 'ylabelrotation' in dic:
        if 'ylabel' not in dic:
            dic['ylabel'] = axs.get_ylabel()
        if 'ylabelfontsize' not in dic:
            dic['ylabelfontsize'] = 10
        if 'ylabelrotation' not in dic:
            dic['ylabelrotation'] = 90
        axs.set_ylabel(dic['ylabel'],fontsize=dic['ylabelfontsize'],rotation=dic['ylabelrotation'])

    if 'xscale' in dic:
        if dic['xscale'] is not None:
            axs.set_xscale(dic['xscale'])

    if 'yscale' in dic:
        if dic['yscale'] is not None:
            axs.set_yscale(dic['yscale'])

    if 'xticks' in dic or 'xtickslabels' in dic or 'xshift' in dic or 'xticksfontsize' in dic or 'xticksrotation' in dic:
        if 'xticks' not in dic:
            dic['xticks'] = axs.get_xticks()
        if 'xtickslabels' not in dic:
            dic['xtickslabels'] = axs.get_xticklabels()
        if 'xshift' in dic:
            if isinstance(dic['xshift'],(int,float,np.ndarray)):
                dic['xticks'] = dic['xticks']+dic['xshift']
            else:
                dic['xticks'] = dic['xticks'][:-1]+(dic['xticks'][1:]-dic['xticks'][:-1])/2
                dic['xticks'] = np.insert(dic['xticks'],0,2*dic['xticks'][0]-dic['xticks'][1])
                dic['xticks'] = np.append(dic['xticks'],2*dic['xticks'][-1]-dic['xticks'][-2])
                if len(dic['xticks']) != len(dic['xtickslabels']):
                    dic['xtickslabels'] = dic['xtickslabels'][:-1]
                    dic['xtickslabels'].insert(0,None)
                    dic['xtickslabels'].append(None)
        if 'xticksfontsize' not in dic:
            dic['xticksfontsize'] = 10
        if 'xticksrotation' not in dic:
            dic['xticksrotation'] = 0
        if 'xticksha' not in dic:
            dic['xticksha'] = 'center'
        if 'xticksva' not in dic:
            dic['xticksva'] = 'top'
        axs.set_xticks(ticks=dic['xticks'],labels=dic['xtickslabels'],fontsize=dic['xticksfontsize'],rotation=dic['xticksrotation'],ha=dic['xticksha'],va=dic['xticksva'])


    if 'yticks' in dic or 'ytickslabels' in dic or 'yshift' in dic or 'yticksfontsize' in dic or 'yticksrotation' in dic:
        if 'yticks' not in dic:
            dic['yticks'] = axs.get_yticks()
        if 'ytickslabels' not in dic:
            dic['ytickslabels'] = axs.get_yticklabels()
        if 'yshift' in dic:
            if isinstance(dic['yshift'],(int,float,np.ndarray)):
                dic['yticks'] = dic['yticks']+dic['yshift']
            else:
                dic['yticks'] = dic['yticks'][:-1]+(dic['yticks'][1:]-dic['yticks'][:-1])/2
                dic['yticks'] = np.insert(dic['yticks'],0,2*dic['yticks'][0]-dic['yticks'][1])
                dic['yticks'] = np.append(dic['yticks'],2*dic['yticks'][-1]-dic['yticks'][-2])
                if len(dic['yticks']) != len(dic['ytickslabels']):
                    dic['ytickslabels'] = dic['ytickslabels'][:-1]
                    dic['ytickslabels'].insert(0,None)
                    dic['ytickslabels'].append(None)
        if 'yticksfontsize' not in dic:
            dic['yticksfontsize'] = 10
        if 'yticksrotation' not in dic:
            dic['yticksrotation'] = 0
        if 'yticksha' not in dic:
            dic['yticksha'] = 'right'
        if 'yticksva' not in dic:
            dic['yticksva'] = 'center'
        axs.set_yticks(ticks=dic['yticks'],labels=dic['ytickslabels'],fontsize=dic['yticksfontsize'],rotation=dic['yticksrotation'],ha=dic['yticksha'],va=dic['yticksva'])

    if 'xaxismajorlocator' in dic:
        if dic['xaxismajorlocator'] is not None:
            axs.locator_params(axis='x',**dic['xaxismajorlocator'])

    if 'yaxismajorlocator' in dic:
        if dic['yaxismajorlocator'] is not None:
            axs.locator_params(axis='y',**dic['yaxismajorlocator'])

    if 'xaxismajorformatter' in dic:
        if dic['xaxismajorformatter'] is not None:
            axs.xaxis.set_major_formatter(mp.pyplot.FormatStrFormatter(dic['xaxismajorformatter']))

    if 'yaxismajorformatter' in dic:
        if dic['yaxismajorformatter'] is not None:
            axs.yaxis.set_major_formatter(mp.pyplot.FormatStrFormatter(dic['yaxismajorformatter']))

    if 'aspect' in dic or 'adjustable' in dic:
        if 'aspect' not in dic:
            dic['aspect'] = 'auto'
        if 'adjustable' not in dic:
            dic['adjustable'] = None
        axs.set_aspect(dic['aspect'],dic['adjustable'])

    if 'xlim' in dic:
        if dic['xlim'] is not None:
            axs.set_xlim(dic['xlim'])

    if 'ylim' in dic:
        if dic['ylim'] is not None:
            axs.set_ylim(dic['ylim'])

    if 'axvlinex' in dic:
        if dic['axvlinex'] is not None:
            axs.axvline(x=dic['axvlinex'],ls=dic['axvlinels'],lw=dic['axvlinelw'],c=dic['axvlinec'],zorder=1)

    if 'axhliney' in dic:
        if dic['axhliney'] is not None:
            axs.axhline(y=dic['axhliney'],ls=dic['axhlinels'],lw=dic['axhlinelw'],c=dic['axhlinec'],zorder=1)

    if 'xticksposition' in dic:
        axs.xaxis.set_ticks_position(dic['xticksposition'])
    if 'yticksposition' in dic:
        axs.yaxis.set_ticks_position(dic['yticksposition'])

    if 'xlabelposition' in dic:
        axs.xaxis.set_ticks_position(dic['xlabelposition'])
    if 'ylabelposition' in dic:
        axs.yaxis.set_ticks_position(dic['ylabelposition'])

    if 'legendloc' in dic or 'legendsize' in dic or 'legendhandle' in dic or 'legendlabel' in dic or 'legendanchor' in dic:
        if 'legendloc' not in dic:
            dic['legendloc'] = None
        if 'legendhandle' not in dic:
            dic['legendhandle'] = axs.get_legend_handles_labels()[0]
        if 'legendlabel' not in dic:
            dic['legendlabel'] = axs.get_legend_handles_labels()[1]
        if 'legendsize' not in dic:
            dic['legendsize'] = 10
        if 'legendscale' not in dic:
            dic['legendscale'] = 1
        if 'legendrehandle' in dic:
            for k,v in dic['legendrehandle'].items():
                dic['legendhandle'][k] = v
        if 'legendrelabel' in dic:
            for k,v in dic['legendrelabel'].items():
                dic['legendlabel'][k] = v
        if 'legendreindex' in dic:
            dic['legendhandle'] = [dic['legendhandle'][i] for i in dic['legendreindex']]
            dic['legendlabel']  = [dic['legendlabel'][i] for i in dic['legendreindex']]
        if 'legendanchor' not in dic:
            dic['legendanchor'] = None
        if 'legendcolumns' not in dic:
            dic['legendcolumns'] = 1
        elif dic['legendcolumns'] is True:
            dic['legendcolumns'] = min(len(dic['legendhandle']),len(dic['legendlabel']))
        axs.legend(loc=dic['legendloc'],handles=dic['legendhandle'],labels=dic['legendlabel'],fontsize=dic['legendsize'],markerscale=dic['legendscale'],bbox_to_anchor=dic["legendanchor"],ncols=dic['legendcolumns'])

    if 'text' in dic:
        if dic['text'] is not None:
            if 'textx' not in dic:
                dic['textx'] = -0.1
            if 'texty' not in dic:
                dic['texty'] = +1.1
            if 'legendlabel' not in dic:
                dic['legendlabel'] = axs.get_legend_handles_labels()[1]
            if 'textsize' not in dic:
                dic['textsize'] = 10
            if 'textweight' not in dic:
                dic['textweight'] = None
        axs.text(dic['textx'],dic['texty'],dic['text'],transform=axs.transAxes,size=dic['textsize'],weight=dic['textweight'])

def plot_ticks(dic,grd):
    """plot ticks
    """

    if 'i' in dic:

        for a in grd.flatten('A')[dic['i']]:
            a.set(xticklabels=[])
            a.tick_params(bottom=False)

    if 'x' in dic:

        if not isinstance(dic['x'],dict):
            dic['x'] = {}

        if 'set' not in dic['x']:
            dic['x']['set'] = [0,-1]

        for a in grd[dic['x']['set'][0]:dic['x']['set'][1]]:
            a.set(xticklabels=[])
            a.tick_params(bottom=False)

    if 'y' in dic:

        if not isinstance(dic['y'],dict):
            dic['y'] = {}

        if 'set' not in dic['y']:
            dic['y']['set'] = [1,-1]

        for a in grd[dic['y']['set'][0]:dic['y']['set'][1]]:
            a.set(yticklabels=[])
            a.tick_params(left=False)

def plot_annotate(dic,grd):
    """plot annotate
    """

    if 'columns' not in dic:
        dic['columns'] = None
    if 'rows' not in dic:
        dic['rows'] = None

    if not isinstance(dic['columns'],list):
        dic['columns'] = [dic['columns']]
    if not isinstance(dic['rows'],list):
        dic['rows'] = [dic['rows']]

    # formatting
    col = ['{}'.format(c) for c in dic['columns']]
    row = ['{}'.format(r) for r in dic['rows']]

    # points
    pad = 10

    if 'columnssize' not in dic:
        dic['columnssize'] = 'large'
    if 'rowssize' not in dic:
        dic['rowssize'] = 'large'

    if 'columnsrotation' not in dic:
        dic['columnsrotation'] = 0
    if 'rowsrotation' not in dic:
        dic['rowsrotation'] = 90

    if 'columnsweight' not in dic:
        dic['columnsweight'] = None
    if 'rowsweight' not in dic:
        dic['rowsweight'] = None

    if len(row) == 1:
        for a,c in zip(grd[:],col):
            if c != 'None':
                a.annotate(c,xy=(0.5,1),xytext=(0,2*pad),xycoords='axes fraction',textcoords='offset points',size=dic['columnssize'],weight=dic['columnsweight'],rotation=dic['columnsrotation'],ha='center',va='baseline')
        if row[0] != 'None':
            grd[0].annotate(row[0],xy=(0,0.5),xytext=(-grd[-1,0].yaxis.labelpad-pad,0),xycoords=grd[0,0].yaxis.label,textcoords='offset points',size=dic['rowssize'],weight=dic['rowsweight'],rotation=dic['rowsrotation'],ha='right',va='center')
        return

    if len(col) == 1:
        for a,r in zip(grd[:],row):
            if r != 'None':
                a.annotate(r,xy=(0,0.5),xytext=(-a.yaxis.labelpad-pad,0),xycoords=a.yaxis.label,textcoords='offset points',size=dic['rowssize'],weight=dic['rowsweight'],rotation=dic['rowsrotation'],ha='right',va='center')
        if col[0] !='None':
            grd[0].annotate(col[0],xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',size=dic['columnssize'],weight=dic['columnsweight'],rotation=dic['columnsrotation'],ha='center',va='baseline')
        return

    for a,c in zip(grd[0,:],col):
        if c != 'None':
            a.annotate(c,xy=(0.5,1),xytext=(0,2*pad),xycoords='axes fraction',textcoords='offset points',size=dic['columnssize'],weight=dic['columnsweight'],rotation=dic['columnsrotation'],ha='center',va='baseline')
    for a,r in zip(grd[:,0],row):
        if r != 'None':
            a.annotate(r,xy=(0,0.5),xytext=(-a.yaxis.labelpad-pad,0),xycoords=a.yaxis.label,textcoords='offset points',size=dic['rowssize'],weight=dic['rowsweight'],rotation=dic['rowsrotation'],ha='right',va='center')

def plot_legend(dic,fig,axs):
    """plot legend
    """

    if 'legendloc' not in dic:
        dic['legendloc'] = None

    if 'legendhandle' not in dic:
        if axs[0].get_legend_handles_labels()[0]:
            dic['legendhandle'] = axs[0].get_legend_handles_labels()[0]
        else:
            dic['legendhandle'] = axs[0].get_legend().legend_handles

    if 'legendlabel' not in dic:
        if axs[0].get_legend_handles_labels()[1]:
            dic['legendlabel'] = axs[0].get_legend_handles_labels()[1]
        else:
            dic['legendlabel'] = [i.get_text() for i in axs[0].get_legend().texts]

    if 'legendsize' not in dic:
        dic['legendsize'] = 10

    if 'legendscale' not in dic:
        dic['legendscale'] = 1

    if 'legendrehandle' in dic:
        for k,v in dic['legendrehandle'].items():
            dic['legendhandle'][k] = v

    if 'legendrelabel' in dic:
        for k,v in dic['legendrelabel'].items():
            dic['legendlabel'][k] = v

    if 'legendreindex' in dic:
        dic['legendhandle'] = [dic['legendhandle'][i] for i in dic['legendreindex']]
        dic['legendlabel']  = [dic['legendlabel'][i] for i in dic['legendreindex']]

    if "legendanchor" not in dic:
        dic['legendanchor'] = None

    if 'legendcolumns' not in dic:
        dic['legendcolumns'] = 1
    elif dic['legendcolumns'] is True:
        dic['legendcolumns'] = min(len(dic['legendhandle']),len(dic['legendlabel']))

    for a in axs:
        a.get_legend().remove()

    fig.legend(loc=dic['legendloc'],handles=dic['legendhandle'],labels=dic['legendlabel'],fontsize=dic['legendsize'],markerscale=dic['legendscale'],bbox_to_anchor=dic["legendanchor"],ncols=dic['legendcolumns'])

def plot_save(fig,nam):
    """"plot save
    """

    # folder
    fol = 'plot'

    # naming
    nam = data_name(nam)

    # path
    pth = data_path(fol,nam,'png')

    # save
    fig.savefig(pth,dpi=fig.dpi,bbox_inches='tight',transparent=False)

def round_exc(x,base=10):
    """excees rounding
    """

    r = x+(base-x)%base

    return r

def round_def(x,base=10):
    """defect rounding
    """

    r = x-(x%base)

    return r

def error_absolute(v_dat,v_fun):
    """absolute error
    """

    # array of values
    arr = (v_fun-v_dat)

    return arr

def error_relative(v_dat,v_fun):
    """relative error
    """

    # array of values
    arr = (v_fun-v_dat)/v_dat

    return arr

def error_normal(v_dat,v_fun):
    """normal error
    """

    # array of values
    arr = np.log10(1+(v_fun-v_dat)/v_dat)

    return arr

def stats_model(vfd,vfc,nam):
    """model evaluation
    """

    # distribution
    d_RP = vfd.loc[:,'RP'].to_numpy()
    d_D  = vfd.loc[:,'D'].to_numpy()
    d_F  = vfd.loc[:,'F'].to_numpy()
    d_v  = vfd.loc[:,nam].to_numpy()

    # curve
    c_RP = vfc.loc[:,'RP'].to_numpy()
    c_v  = vfc.loc[:,nam].to_numpy()

    # initialize
    sel  = np.array([])

    for i in d_RP:
        idx = np.abs(c_RP-i).argmin()
        val = c_v[idx]
        sel = np.append(sel,val)

    # dataframe
    dtf = pd.DataFrame(data=zip(d_F,d_D,d_RP,d_v,sel),columns=['F','D','RP','observations','model'])

    return dtf

def stats_errors(dtf,ttl_x,ttl_y):
    """stats errors
    """

    # distribution
    x = dtf.loc[:,ttl_x].to_numpy()
    y = dtf.loc[:,ttl_y].to_numpy()

    # errors
    err_abs = error_absolute(x,y)
    err_rel = error_relative(x,y)
    err_nor = error_normal(x,y)

    return err_abs,err_rel,err_nor

def stats_probability():
    """stats probability
    """

    # return period
    RP = fun_values([1,2,10,20,200],[0.05,0.5,2,20],None)

    # cumulative probabilities
    D = 1/RP
    F = 1-D

    # dataframe
    sts = pd.DataFrame(data=zip(RP,D,F),columns=['RP','D','F'])
    sts = sts.reset_index(drop=True)

    return sts

def stats_evaluation(wtb,ttl):
    """stats evaluation
    """

    # minimum/maximum/average
    v_min = wtb.loc[:,ttl].min()
    v_max = wtb.loc[:,ttl].max()
    v_avg = wtb.loc[:,ttl].mean()

    return v_min,v_max,v_avg

def stats_maxima(wtb,ttl):
    """stats maxima"""

    # array
    arr = wtb.dropna(subset=ttl).groupby('year')[ttl].max()

    return arr

def stats_empirical(v,nam):
    """stats empirical
    """

    # dropping nans
    v = v[~np.isnan(v)]

    # sorting values
    v = np.sort(v)

    # number of values
    n = len(v)

    # cumulative probabilities
    F  = np.arange(1,n+1)/(n+1)
    D  = np.ones(n)-F
    RP = 1/D

    # dataframe
    sts = pd.DataFrame(data=zip(v,F,D,RP),columns=[nam,'F','D','RP'])

    return sts

def stats_distribution(cdf,v,nam):
    """stats distribution
    """

    # values
    if not isinstance(v,np.ndarray):

        # if no valid input, array is set from default one
        v = fun_values([0.01,1,100],[0.01,1],None)

    # stats
    F  = cdf(v)
    D  = 1-F
    RP = 1/D

    # dataframe
    sts = pd.DataFrame(data=zip(v,F,D,RP),columns=[nam,'F','D','RP'])

    return sts

def stats_pwm(data):
    """sample probability weighted moments
    """

    # size
    siz = len(data)

    # sorting
    data = np.sort(data)

    # compute PMWs
    b0 = np.mean(data)
    b1 = 0

    for i in range(1,siz):
        b1 = b1+(i)/(siz-1)*data[i]

    # normalization
    b1 = b1/siz

    return b0,b1

def stats_lmom(b0,b1):
    """sample probability L-moments
    """

    # compute L-moments, linear combination of the PMWs
    L1 = b0
    L2 = 2*b1-b0

    return L1,L2

def dis_weibull(par,q):
    """distribution function values for WEIBULL
    """

    val = 1-np.exp(-(q/par[1])**par[0])

    return val

def dis_gamma(par,q):
    """distribution function values for GAMMA
    """

    val = sp.stats.gamma.cdf(q,par[0],scale=par[1],loc=0)

    return val

def dis_gpd(par,q):
    """distribution function values for GPD
    """

    val = 1-(1-(par[1]/par[0])*(q-par[2]))**(1/par[1])

    return val

def dis_phev(par,q,d,scl,typ):
    """distribution function values for PHEV
    """

    if len(par.shape) == 1:
        F   = phev_distribution(par,d,scl,typ,'F')
        val = F(q)

    else:

        # initialize
        val = np.ones(len(par.shape[0]))*np.nan

        for i in range(len(par.shape[0])):
            F      = phev_distribution(par[i],d,scl,typ,'F')
            val[i] = F(q)

    return val

def peaks_stationary(wtb,sen):
    """peaks stationary

    wtb -> working dataframe;
    sen -> sensitivity for the extraction.

    """

    # strength of extraction
    if not isinstance(sen,int):

        # if no valid input, sensitivity is set to default one
        sen = 10000

    # limit
    lim = None

    # size
    siz = len(wtb)

    if lim is None:
        lim = np.arange(siz)

    # streamflows
    arr = wtb.loc[:,'streamflow'].to_numpy()

    # deltas
    dlt = (wtb.loc[:,'streamflow'].max()-wtb.loc[:,'streamflow'].min())/sen

    # lengths
    len_arr = len(arr)
    len_lim = len(lim)

    if len_arr != len_lim:
        fun_print(None,'Peaks: vectors must have same length')
        raise

    if not np.isscalar(dlt):
        fun_print(None,'Peaks: delta must be a scalar')
        raise

    if dlt <= 0:
        fun_print(None,'Peaks: delta must be positive')
        raise

    # initializing
    mn,mx       = np.PINF,np.NINF
    mnpos,mxpos = np.nan,np.nan

    # lists
    lst_max = []
    lst_min = []

    # flagging
    flg  = True

    for i in np.arange(len_arr):

        # value
        val = arr[i]

        if val > mx:
            mx    = val
            mxpos = lim[i]

        if val < mn:
            mn    = val
            mnpos = lim[i]

        if flg:

            if val < mx-dlt:
                lst_max.append((mxpos,mx))
                mn    = val
                mnpos = lim[i]
                flg   = False

        else:

            if val > mn+dlt:
                lst_min.append((mnpos,mn))
                mx    = val
                mxpos = lim[i]
                flg   = True

    # arrays
    arr_max = np.asarray(lst_max,dtype=float)
    arr_min = np.asarray(lst_min,dtype=float)

    # stationary points
    wtb.loc[:,'stationary']            = None
    wtb.loc[arr_max[:,0],'stationary'] = '+'
    wtb.loc[arr_min[:,0],'stationary'] = '-'

    return wtb

def peaks_mev(wtb,lag,siz,drp):
    """peaks selection for MEV

    wtb -> working dataframe;
    lag -> value used to lag;
    siz -> size of the basin;
    drp -> drop factor for streamflow.

    """

    if not isinstance(lag,float) and not isinstance(lag,int):

        # if no valid input, lag is set to basin estimation or default value

        if isinstance(siz,float) or isinstance(siz,int):

            # lag is set to size-estimated value
            lag = round(5+np.log10(siz/2.59))

        else:

            # lag is set to default value
            lag = 0

    if not isinstance(drp,float) and not isinstance(drp,int):

        # if no valid input, drop is set to default one
        drp = 0.75

    # peaks column
    wtb.loc[:,'peak'] = False

    # temporary table
    tmp              = wtb.loc[wtb.loc[:,'stationary'] == '+'].loc[:,['streamflow','timestamp']]
    tmp              = tmp.sort_values(by='streamflow',ascending=False)
    tmp.loc[:,'idx'] = tmp.index
    tmp              = tmp.reset_index(drop=True)

    # initialize
    i   = 0
    l   = len(tmp)
    flg = None

    while i < l and not flg:

        # lag indexes
        lag_pre = np.arange(tmp.loc[i,'timestamp']-lag,tmp.loc[i,'timestamp'])
        lag_end = np.arange(tmp.loc[i,'timestamp']+1,tmp.loc[i,'timestamp']+lag+1)
        lag_int = np.concatenate((lag_pre,lag_end))

        # processing
        rem = tmp.iloc[mp.isin(tmp.loc[:,'timestamp'].to_numpy(),lag_int)].index
        tmp = tmp.drop(tmp.index[rem])
        tmp = tmp.reset_index(drop=True)

        # updating
        l = len(tmp)

        if i < l:
            flg = False

        else:
            flg = True

        # index
        i = i+1

    # peaks flagging
    wtb.loc[tmp.loc[:,'index'],'peak'] = True

    # peaks index
    idx = wtb.loc[wtb.loc[:,'peak']].index

    # select column
    wtb.loc[:,'select'] = wtb.loc[:,'peak']

    for i in range(1,len(idx)-1):

        # minimum value between two consecutive peaks and its index
        min_q1_q2 = np.min([wtb.loc[idx[i-1],'streamflow'],wtb.loc[idx[i],'streamflow']])
        ind       = np.argmin([wtb.loc[idx[i-1],'streamflow'],wtb.loc[idx[i],'streamflow']])

        # define which one is the index of the potential peak to be unflagged
        if ind == 0:
            flg = i-1
        else:
            flg = i

        # minimum flow
        qmin = np.min(wtb.loc[idx[i-1]+1:idx[i]-1,'streamflow'])

        # select unflagging
        if qmin > drp*min_q1_q2:
            wtb.loc[idx[flg],'select'] = False

    # output
    out = None

    return wtb,out

def peaks_phev(wtb,opt,win,lng,mod):
    """peaks selection for PHEV

    wtb -> working dataframe;
    opt -> computation options for recessions;
    win -> smoothing window for computing second derivative for concavity;
    lng -> recessions minimum length;
    mod -> recessions fitting mode.

    """

    # recessions computation options
    # 1 -> decreasing recessions
    # 2 -> decreasing/convex recessions
    if not isinstance(opt,int):

        # if no valid input, value is set to default one
        opt = 1

    # smoothing window for computing 2nd derivative for concavity
    # 5 -> standard value
    # 3 -> fallback value
    if not isinstance(win,int):

        # if no valid input, value is set to default one
        win = 5

    # recessions minimum length
    # 5 -> standard value
    # 3 -> fallback value
    if not isinstance(lng,float) and not isinstance(lng,int):

        # if no valid input, value is set to default one
        lng = 5

    # rfm (recessions fitting mode)
    # 1 -> linear fitting
    # 2 -> non-linear fitting
    if not isinstance(mod,int):

        # if no valid input, value is set to default one
        mod = 1

    # compute first and second derivatives of smoothed values
    wtb.loc[:,'d0_smooth']   = wtb.loc[:,'streamflow'].rolling(win).mean()
    wtb.loc[0:2,'d0_smooth'] = wtb.loc[0:2,'streamflow']
    wtb.loc[:,'d1_smooth']   = wtb.loc[:,'d0_smooth'].diff().shift(-1)
    wtb.loc[:,'d2_smooth']   = wtb.loc[:,'d1_smooth'].diff().shift(-1)

    # compute first and second derivatives of unsmoothed values
    wtb.loc[:,'d0_unsmooth'] = wtb.loc[:,'streamflow']
    wtb.loc[:,'d1_unsmooth'] = wtb.loc[:,'d0_unsmooth'].diff().shift(-1)
    wtb.loc[:,'d2_unsmooth'] = wtb.loc[:,'d1_unsmooth'].diff().shift(-1)

    # reducing
    wtb = wtb[:-2]

    # boolean vector for concavity periods
    if opt == 1:
        wtb.loc[:,'recession'] = (wtb.loc[:,'d1_unsmooth'] < 0)

    elif opt == 2:
        wtb.loc[:,'recession'] = (wtb.loc[:,'d1_unsmooth'] < 0) & ((wtb.loc[:,'d2_smooth'] >= 0)|(wtb.loc[:,'d2_unsmooth'] >= 0))

    # parameters dictionary
    dic = dic_parameters(None)

    # number of days to lag the start of any extracted recession
    lag = 0

    # adjustments
    wtb.loc[:,'streamflow'] = wtb.loc[:,'streamflow']+1e-12

    # column for selected peaks
    wtb.loc[:,'select'] = False

    # parameters
    par_a = None
    par_k = None

    # arrays
    arr_q = np.array([])
    arr_p = np.array([])
    arr_a = np.array([])
    arr_b = np.array([])
    arr_d = np.array([])
    arr_k = np.array([])

    # coefficients of determinations of recessions
    cdr_ab = np.array([])
    cdr_k  = np.array([])

    # fitting dataframe
    fit_ab = pd.DataFrame()
    fit_k  = pd.DataFrame()

    # arrays
    arr_dates = np.array([])
    arr_a_all = np.array([])
    arr_b_all = np.array([])

    # startings
    qstart_ab = []
    qstart_k  = []

    # indexers
    idx_short = []
    idx_lower = []

    # peaks/dates
    dates = wtb.loc[wtb.loc[:,'stationary'] == '+',:].index
    peaks = wtb.loc[dates,'streamflow'].to_numpy()

    # dates/peaks
    array = np.asarray([dates,peaks],dtype=float)
    array = np.transpose(array)

    # mean flow
    mean = wtb.loc[:,'streamflow'].mean()

    # recessions number
    num = 0

    for i in np.arange(len(dates)-1):

        if wtb.loc[dates[i],'streamflow'] > mean:

            # previous and next recessions
            rec_pre = dates[i]+lag
            rec_nxt = dates[i+1]

            if rec_nxt <= rec_pre:
                continue

            # end of current recession
            rec_end = wtb[rec_pre:rec_nxt][np.invert(wtb[rec_pre:rec_nxt]['recession'])].index[0]+1

            if (np.any(wtb['streamflow'][rec_pre:rec_end] < 0))|(len(wtb['streamflow'][rec_pre:rec_end]) <= lng):
                idx_short.append(i)
                continue

            # extracting
            q   = wtb['streamflow'][rec_pre:rec_end].to_numpy()
            tmp = {'i':i,'real_idx':int(array[i,0]),'q':q[0]}

            # appending
            qstart_ab.append(tmp)
            arr_q     = np.append(arr_q,tmp['q'])
            arr_dates = np.append(arr_dates,dates[i])

            try:

                if mod == 1:

                    # initial values
                    v0 = [np.log(dic[3]['reference']),np.log(dic[2]['reference'])]

                    # boundaries
                    bnd = [(-np.inf,-np.inf),(np.inf,np.inf)]

                    # increments
                    qs  = (q[:-1]+q[1:])/2
                    dqs = np.diff(q)

                    # data
                    x = np.log(qs)
                    y = np.log(-dqs)

                    if not np.any(np.isfinite(x)) and not np.any(np.isfinite(y)):
                        continue

                    # optimization
                    fun     = lambda x,a,b: b*x+a
                    opt,cov = sp.optimize.curve_fit(fun,x,y,p0=v0,bounds=bnd)
                    ab      = np.exp(opt[0]),opt[1]

                elif mod == 2:

                    # initial values
                    v0 = [dic[3]['reference'],dic[2]['reference']]

                    # boundaries
                    bnd = [(0,0),(np.inf,np.inf)]

                    # data
                    x = np.arange(len(q))
                    y = q

                    # optimization
                    fun     = lambda x,a,b: ((-1+b)*(q[0]**(1-b)/(-1+b)+a*x))**(1/(1-b))
                    opt,cov = sp.optimize.curve_fit(fun,x,y,p0=v0,bounds=bnd)
                    ab      = opt[0],opt[1]

                else:
                    fun_print(None,'Recessions: invalid recessions fitting mode')
                    raise

                # coefficient of determination of the recession
                ss_res = np.sum((y-fun(x,opt[0],opt[1]))**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                ss     = 1-(ss_res/ss_tot)

                if ab[1] < 0:
                    fun_print(None,'Recessions: physically impossible for index',i)
                    continue

                # recessions number
                num = num+1

                # function
                fun = lambda x,a,b: ((-1+b)*(q[0]**(1-b)/(-1+b)+a*x))**(1/(1-b))

                # dates
                dat_pre = wtb.loc[rec_pre,'date']
                dat_end = wtb.loc[rec_end,'date']

                # arrays
                arr_t = np.arange(dat_pre,dat_end,dt.timedelta(hours=6)).astype(dt.datetime)
                arr_f = fun(np.arange(1e-3,len(q),0.25),ab[0],ab[1])
                arr_n = num*np.ones(len(arr_t))

                # dataframe
                dtf = pd.DataFrame(data=zip(arr_t,arr_f,arr_n),columns=['date','streamflow','number'])

            except:
                fun_print(None,'Recessions: fitting (a,b) does not work for index',i)
                continue

            # appending
            arr_a_all = np.append(arr_a_all,ab[0])
            arr_b_all = np.append(arr_b_all,ab[1])
            cdr_ab    = np.append(cdr_ab,ss)
            fit_ab    = pd.concat([fit_ab,dtf],ignore_index=True)

        else:
            idx_lower.append(i)

    # eliminate any potential negative values
    ind_pos = [i for i in range(len(arr_b_all)) if arr_b_all[i] > 0]

    # recession exponents
    arr_a = np.asarray([arr_a_all[i] for i in ind_pos],dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]

    # recession exponents
    arr_b = np.asarray([arr_b_all[i] for i in ind_pos],dtype=float)
    arr_b = arr_b[~np.isnan(arr_b)]

    if num >= 5:

        # recessions number
        num = 0

        # median exponent
        b_med    = np.median(arr_b)

        # decorrelation
        ag       = sp.stats.gmean(arr_a)
        exponent = np.sum((arr_b-np.mean(arr_b))*np.log10(arr_a/ag))/np.sum((arr_b-np.mean(arr_b))**2)
        qscale   = 10**(-exponent)
        arr_d    = arr_a*qscale**(arr_b-1)

        for i in np.arange(len(dates)-1):

            if wtb['streamflow'][dates[i]] > mean:

                # previous and next recessions
                rec_pre = dates[i]+lag
                rec_nxt = dates[i+1]

                if rec_nxt <= rec_pre:
                    continue

                # end of current recession
                rec_end = wtb[rec_pre:rec_nxt][np.invert(wtb[rec_pre:rec_nxt]['recession'])].index[0]+1

                if (np.any(wtb['streamflow'][rec_pre:rec_end] < 0))|(len(wtb['streamflow'][rec_pre:rec_end]) < lng):
                    continue

                # extracting
                q   = wtb['streamflow'][rec_pre:rec_end].to_numpy()
                tmp = {'i':i,'real_idx':int(array[i,0]),'q':q[0]}

                # appending
                qstart_k.append(tmp)

                try:

                    if mod == 1:

                        # initial values
                        v0 = [np.log(dic[3]['reference'])]

                        # boundaries
                        bnd = [(-np.inf),(np.inf)]

                        # increments
                        qs  = (q[:-1]+q[1:])/2
                        dqs = np.diff(q)

                        # data
                        x = np.log(qs)
                        y = np.log(-dqs)

                        if not np.any(np.isfinite(x)) and not np.any(np.isfinite(y)):
                            continue

                        # optimization
                        fun     = lambda x,a: b_med*x+a
                        opt,cov = sp.optimize.curve_fit(fun,x,y,p0=v0,bounds=bnd)
                        k       = np.exp(opt[0])

                    elif mod == 2:

                        # initial values
                        v0 = [dic[3]['reference']]

                        # boundaries
                        bnd = [(0),(np.inf)]

                        # data
                        x = np.arange(len(q))
                        y = q

                        # optimization
                        fun     = lambda x,a: ((-1+b_med)*(q[0]**(1-b_med)/(-1+b_med)+a*x))**(1/(1-b_med))
                        opt,cov = sp.optimize.curve_fit(fun,x,y,p0=v0,bounds=bnd)
                        k       = opt[0]

                    else:
                        fun_print(None,'Recessions: invalid recessions fitting mode')
                        raise

                    # coefficient of determination of the recession
                    ss_res = np.sum((y-fun(x,opt[0]))**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    ss     = 1-(ss_res/ss_tot)

                    # recessions number
                    num = num+1

                    # function
                    fun = lambda x,a: ((-1+b_med)*(q[0]**(1-b_med)/(-1+b_med)+a*x))**(1/(1-b_med))

                    # dates
                    dat_pre = wtb.loc[rec_pre,'date']
                    dat_end = wtb.loc[rec_end,'date']

                    # arrays
                    arr_t = np.arange(dat_pre,dat_end,dt.timedelta(hours=6)).astype(dt.datetime)
                    arr_f = fun(np.arange(1e-3,len(q),0.25),k)
                    arr_n = num*np.ones(len(arr_t))

                    # dataframe
                    dtf = pd.DataFrame(data=zip(arr_t,arr_f,arr_n),columns=['date','streamflow','number'])

                except:
                    fun_print(None,'Recessions: fitting (k) does not work for index',i)
                    continue

                # appending
                arr_p = np.append(arr_p,q[0])
                arr_k = np.append(arr_k,k)
                cdr_k = np.append(cdr_k,ss)
                fit_k = pd.concat([fit_k,dtf],ignore_index=True)

                # selecting
                wtb.loc[rec_pre,'select'] = True

        # recession values
        arr_k = arr_k[~np.isnan(arr_k)]

        # recession exponent
        par_a = np.median(arr_b)

        # recession coefficient
        par_k = np.median(arr_k)

        # flagging
        flg = True

    else:
        fun_print(None,'Recessions: minimum length not respected')
        flg = False

    # recessions parameters
    par = par_a,par_k

    # recessions arrays
    arr = arr_a,arr_b,arr_d,arr_k

    # coefficients of determination of recessions
    cdr = cdr_ab,cdr_k

    # recessions fits
    fit = fit_ab,fit_k

    # peaks slope
    fun     = lambda x,a,b: a*x+b
    opt,cov = sp.optimize.curve_fit(fun,np.log10(arr_p),np.log10(arr_k))
    slp     = opt,cov

    # output
    out = par,arr,cdr,fit,slp,flg

    return wtb,out

def peaks_percentile(wtb,prc):
    """peaks percentile

    wtb -> working dataframe;
    prc ->  percentile for peaks threshold.

    """

    # peaks percentile value
    if not isinstance(prc,float) and not isinstance(prc,int):

        # if no valid input, percentile is set from default one
        prc = 10

    # peaks
    pks = wtb.loc[wtb.loc[:,'select']].loc[:,'streamflow'].to_numpy()

    # threshold
    thr = np.percentile(pks,prc)

    # unflagging
    wtb.loc[wtb.loc[:,'select']*(wtb.loc[:,'streamflow'] < thr),'select'] = False

    return wtb,thr

def peaks_results(wtb):
    """peaks results

    wtb -> working dataframe.

    """

    # array of peaks
    pks = wtb.loc[wtb.loc[:,'select']].loc[:,'streamflow'].to_numpy()

    # series of exceeds
    exc = wtb.loc[wtb.loc[:,'select']].groupby(by='year').size()
    exc = exc.to_frame(0)
    exc = table_rename(exc,0,'exceedances')

    return pks,exc

def days_period(prd):
    """period days"""

    # dictionary
    dic = dic_period(prd)

    # days
    day = dic['day']

    return day

def days_precipitation(wtb):
    """period precipitation days"""

    # size
    siz = wtb.shape[0]-1

    # drop dry days (precipitation = 0)
    dat = wtb.loc[wtb.loc[:,'precipitation'] != 0]

    # number of wet days per reference period
    day = dat.groupby('year')['precipitation'].count()

    # counting observations
    cnt = dat.precipitation.count()

    # frequency
    frq = cnt/siz

    return day,frq

def par_gamma(data):
    """parameters of the gamma distribution estimated via L-moments
    """

    # size
    n = len(data)

    # stats
    b0,b1 = stats_pwm(data)
    L1,L2 = stats_lmom(b0,b1)
    t     = L2/L1

    if n < 5:

        # too few values
        shp = np.nan
        scl = np.nan

    else:

        if t > 0 and t < 0.5:
            z   = np.pi*t**2
            shp = (1-0.3080*z)/(z-0.05812*z**2+0.01765*z**3)

        elif t >= 0.5 and t < 1:
            z   = 1-t
            shp = (0.7213*z-0.5947*z**2)/(1-2.1817*z+1.2113*z**2)

        scl = L1/shp

    return shp,scl

def par_weibull(data):
    """parameters of the weibull distribution estimated via PWM
    """

    # size
    n = len(data)

    if n < 5:

        # too few values
        scl = np.nan
        shp = np.nan

    else:

        b0 = np.mean(data)
        b1 = 0
        F  = np.zeros(n)

        for i in range(n):
            j    = i+1
            F[i] = j/(n+1)
            b1   = b1+data[i]*(1-F[i])

        b1  = b1/n
        scl = b0/sp.special.gamma(np.log(b0/b1)/np.log(2))
        shp = np.log(2)/np.log(b0/(2*b1))

    return shp,scl

def par_gpd(data):
    """parameters of the GPD estimated via LMOM
    """

    # sorting
    data = np.sort(data)

    # stats
    b0,b1 = stats_pwm(data)
    L1,L2 = stats_lmom(b0,b1)

    # parameters
    k   = L1/L2-2
    alp = (1+k)*L1

    return k,alp

def par_alpha(data):
    """alpha parameter - average amount of precipitation on rainy days
    """

    # precipitation
    data_alp = data.loc[:,'precipitation']

    # drop dry days (rain = 0)
    data_alp = data_alp.where(data_alp != 0)

    # drop missing data
    data_wet = data_alp.dropna(axis=0,how='any')

    # calculating the mean
    alp = data_wet.mean()

    # compute
    arr     = (data_wet.to_numpy()-alp)**2
    val     = np.sum(arr)
    std_alp = np.sqrt(((1/(float(len(data_wet))-1))*float(val))/float(len(data_wet)))

    return alp,std_alp,data_alp

def par_labda(data):
    """labda parameter -ratio between long term average streamflow and average amount of precipitations on rainy days (alpha)
    """

    # streamflow
    data_lab = data.loc[:,'streamflow']

    # drop missing values
    data_lab = data_lab.dropna(axis=0,how='any')

    # calculating the mean
    mean_q = data_lab.mean()

    # compute
    lab     = mean_q/par_alpha(data)[0]
    std_q   = data_lab.to_numpy().std()
    std_lab = np.sqrt((par_alpha(data)[1]**2)*(mean_q/(par_alpha(data)[0]**2)**2)+(std_q**2)*(1/(par_alpha(data)[0]**2)))

    return lab,std_lab,data_lab

def mev_values(dis,par,q,n,*args):
    """MEV distribution values

    dis  -> designed distribution function;
    par  -> parameter set of the distribution;
    q    -> streamflow values;
    n    -> exponent number;
    args -> specific parameters for the distribution function.

    """

    if dis == 'phev':
        val = dis_phev(par,q,*args)**n

    elif dis == 'weibull':
        val = dis_weibull(par,q,*args)**n

    elif dis == 'gamma':
        val = dis_gamma(par,q,*args)**n

    elif dis == 'gpd':
        val = dis_gpd(par,q,*args)**n

    return val

def mev_function(dis,par,q,n,prb,*args):
    """MEV distribution function

    dis  -> designed distribution function;
    par  -> parameter set of the distribution;
    q    -> streamflow values;
    n    -> number of exceed days per reference period;
    prb  -> probability values;
    args -> specific parameters for the distribution function.

    """

    if isinstance(n,pd.Series):
        n = n.to_numpy()

    # getting the size
    M = n.size

    # computing the values
    val = mev_values(dis,par,q,n,*args)

    # computing the function
    fun = np.sum(val)-M*prb

    return fun

def mev_discrete(dis,par,q,n,*args):
    """MEV discrete

    dis  -> designed distribution function;
    par  -> parameter set of the distribution;
    q    -> streamflow values;
    n    -> number of exceed days per reference period;
    args -> specific parameters for the distribution function.

    """

    if isinstance(n,pd.Series):
        n = n.to_numpy()

    # getting the size
    M = n.size

    # computing the values
    val = mev_values(dis,par,q,n,*args)

    # computing the cdf
    fun = (1/M)*np.sum(val)

    return fun

def mev_ffc_simple(dis,par,n,*args):
    """MEV flood frequency curve simple

    dis  -> designed distribution function;
    par  -> parameter set of the distribution;
    n    -> number of exceed days per reference period;
    args -> specific parameters for the distribution function.

    """

    # cumulative distribution function
    cdf = lambda x: mev_discrete(dis,par,x,n,*args)

    # flood frequency curve
    ffc = vfc_simple(cdf,None,'q')

    return ffc

def mev_ffc_quantile(ffd,thr,v_0,dis,par,n,*args):
    """MEV flood frequency curve quantile

    ffd  -> flood freqency distribution;
    thr  -> threshold for exceedances;
    v_0  -> first value to be given to fsolve;
    dis  -> designed distribution function;
    par  -> parameter set of the distribution;
    n    -> number of exceed days per reference period;
    args -> specific parameters for the distribution function.

    """

    if not isinstance(ffd,pd.DataFrame):
        add = None

    else:
        add = ffd.loc[:,'F'].to_numpy()

    # non-exceedance probability
    prb = fun_values([0.005,0.050,0.950,0.995],[0.001,0.010,0.001],add)

    # objective function
    obj = lambda x,prb: mev_function(dis,par,x,n,prb,*args)

    # flood frequency curve
    ffc = vfc_quantile(prb,thr,v_0,obj)

    return ffc

def phev_parameters(clb,est,flg):
    """PHEV parameters
    """

    # initialize
    par = np.ones(4)*np.nan

    # parameters indexes
    i_clb = [i for i,x in enumerate(flg) if x]
    i_est = [i for i,x in enumerate(flg) if not x]

    # parameters array
    par[i_clb] = clb
    par[i_est] = est

    return par

def phev_pj(par,scl=False):
    """PHEV probability distribution of peakflows
    """

    # distribution check
    chk = 2e-2

    if chk > abs(par[2]-2):
        return phev_pj_a2(par,scl)
    else:
        return phev_pj_np(par,scl)

def phev_pj_np(par,scl=False):
    """PHEV probability distribution of peakflows - non-linear storage-discharge relationship with a != 2
    """

    if scl:
        s = par[0]*par[1]
    else:
        s = 1

    # function
    f = lambda x: ((s*x)**(1-par[2]))*sp.exp(par[1]*((s*x)**(1-par[2]))/(par[3]*(1-par[2]))-((s*x)**(2-par[2]))/(par[0]*par[3]*(2-par[2])))

    # normalization constant
    c = 1/sp.integrate.quad(f,0,sp.inf,full_output=0)[0]

    # probability distribution equation
    p = lambda x: c*f(x)

    return p

def phev_pj_a2(par,scl=False):
    """PHEV probability distribution of peakflows - non-linear storage-discharge relationship with a == 2
    """

    if scl:
        s = par[0]*par[1]
    else:
        s = 1

    # probability distribution equation
    p = lambda x: s*sp.stats.invgamma.pdf(s*x,1/(par[0]*par[3]),0,par[1]/par[3])

    return p

def phev_distribution(par,d,scl,typ,fun):
    """PHEV distribution
    """

    # pdf of ordinary
    pj = phev_pj(par,scl)

    if typ == 'ordinary' and fun == 'p':
        return pj
    else:
        Fj = lambda x: sp.integrate.quad(pj,0,x,full_output=0)[0]
        Dj = lambda x: 1-Fj(x)

    if typ == 'ordinary' and fun == 'F':
        return Fj

    if typ == 'ordinary' and fun == 'D':
        return Dj

    # pdf of maxima
    pm = lambda x: par[1]*d*np.exp(-par[1]*d*Dj(x))*pj(x)

    if typ == 'maxima' and fun == 'p':
        return pm
    else:
        Fm = lambda x: np.exp(-par[1]*d*Dj(x))
        Dm = lambda x: 1-Fm(x)

    if typ == 'maxima' and fun == 'F':
        return Fm

    if typ == 'maxima' and fun == 'D':
        return Dm

def phev_nll(clb,est,flg,d,scl,typ,data):
    """PHEV distribution negative log likelihood
    """

    # parameters
    par = phev_parameters(clb,est,flg)

    # probability distribution function
    pdf = phev_distribution(par,d,scl,typ,'p')

    # values
    val = -np.log(pdf(data))

    # negative log likelihood
    nll = np.sum(val)

    return nll

def phev_error(clb,est,flg,d,scl,typ,data):
    """PHEV distribution error
    """

    # parameters
    par = phev_parameters(clb,est,flg)

    # emprirical statistics
    sts = stats_empirical(data,'q')

    # cumulative distribution function
    cdf = phev_distribution(par,d,scl,typ,'F')

    # size
    siz = len(data)

    # initialize
    val = np.ones(siz)*np.nan

    for i in range(siz):
        val[i] = np.abs(cdf(sts.loc[i,'q'])-sts.loc[i,'F'])

    # error
    err = np.sum(val)

    return err

def phev_ffc_simple(par,d,scl,typ):
    """PHEV flood frequency curve simple
    """

    # cumulative distribution function
    cdf = phev_distribution(par,d,scl,typ,'F')

    # flood frequency curve
    ffc = vfc_simple(cdf,None,'q')

    return ffc

def phev_ffc_quantile(ffd,thr,v_0,par,*args):
    """PHEV flood frequency curve quantile
    """

    if not isinstance(ffd,pd.DataFrame):
        add = None

    else:
        add = ffd.loc[:,'F'].to_numpy()

    # non-exceedance probability
    prb = fun_values([0.005,0.050,0.950,0.995],[0.001,0.010,0.001],add)

    # setting distribution
    args = (*args,'F')

    # cumulative distribution
    cdf = phev_distribution(par,*args)

    # objective function
    obj = lambda x,prb: cdf(x)-prb

    # flood frequency curve
    ffc = vfc_quantile(prb,thr,v_0,obj)

    return ffc

def fun_print(lvl,*args):
    """printing function

    lvl  -> logging level;
    args -> function arguments.

    """

    # verbousity setting
    # 0 -> vital
    # 1 -> brief
    # 2 -> standard
    # 3 -> extensive
    vrb = 1

    if not isinstance(lvl,int):
        lvl = 3

    if lvl <= vrb:
        print(*args)

def fun_plot(frm,plo,ext,dtf,lbl,hue,nam):
    """plotting function

    frm -> frame dictionary;
    plo -> plot type;
    ext -> extra dictionary;
    dtf -> values dataframe(s);
    lbl -> data labels;
    hue -> data hue(s);
    nam -> figure naming.

    """

    if frm is None:
        frm = {'figure':None,'annotate':None,'legend':None}
    else:
        frm = frm.copy()

    # figure
    fig,axs = plot_figure(frm['figure'])

    # grid
    grd = axs

    if not isinstance(axs,np.ndarray):
        axs = np.asarray([axs],dtype=object)
    elif axs.ndim > 1:
        axs = grd.flatten('A')

    if any(not isinstance(i,list) for i in [plo,ext,dtf,lbl,hue]):
        plo,ext,dtf,lbl,hue = [plo],[ext],[dtf],[lbl],[hue]

    for s_axs,s_plo,s_ext,s_dtf,s_lbl,s_hue in zip(axs,plo,ext,dtf,lbl,hue):

        if isinstance(s_plo,dict):
            s_dic = s_plo
        else:
            s_dic = dic_plot(s_plo)

        if s_ext is not None:
            for k,v in s_ext.items():
                if k in s_dic:
                    if isinstance(s_dic[k],dict):
                        s_dic[k].update(v)
                else:
                    s_dic[k] = v

        if s_dic['type'] == None:
            s_axs.set_visible(False)

        if isinstance(s_dic['type'],list):

            for i in range(len(s_dic['type'])):
                if 'autoscale' in s_dic:
                    if s_dic['autoscale'] is not None:
                        s_axs.autoscale(s_dic['autoscale'][i])

                if isinstance(s_dic['type'][i],dict):
                    s_m_dic = s_dic['type'][i]
                else:
                    s_m_dic = dic_plot(s_dic['type'][i])

                plot_function(s_m_dic,s_axs,s_dtf[i],s_lbl,s_hue)

        else:
            plot_function(s_dic,s_axs,s_dtf,s_lbl,s_hue)

        if 'line' in s_dic:
            if s_dic['line'] is not None:
                plot_line(s_dic['line'],s_axs,s_dtf,s_lbl)

        if 'distribution' in s_dic:
            if s_dic['fitting'] is not None:
                plot_distribution(s_dic['distribution'],s_axs,s_dtf,s_lbl)

        if 'format' in s_dic:
            if s_dic['format'] is not None:
                plot_format(s_dic['format'],s_axs)

    if 'ticks' in frm:
        if frm['ticks'] is not None:
            plot_ticks(frm['ticks'],grd)

    if 'annotate' in frm:
        if frm['annotate'] is not None:
            plot_annotate(frm['annotate'],grd)

    if 'legend' in frm:
        if frm['legend'] is not None:
            plot_legend(frm['legend'],fig,axs)

    if nam is not None:
        plot_save(fig,nam)

    # clearing
    fig.clf()

def fun_values(val,res,add):
    """values generator function

    val -> pivot values for intervals;
    res -> resolution of each interval;
    add -> optional values to be added to list.

    """

    # values
    v = np.arange(val[0],val[1],res[0],dtype=float)

    # index
    i = len(val)-1

    if i > 1:
        for j in range(1,i):
            v = np.append(v,np.arange(val[j],val[j+1],res[j]))

    if add is not None:
        v = np.append(v,add)
        v = np.sort(v)

    return v

def fun_list(idn,prd,cod):
    """list initialization function

    idn -> selected idn(s);
    prd -> selected period(s);
    cod -> selected codification(s).

    """

    if idn is None:
        idn = list_idn(None,None)
    elif not isinstance(idn,list):
        idn = [idn]

    if prd is None:
        prd = list_prd(None)
    elif not isinstance(prd,list):
        prd = [prd]

    if cod is None:
        cod = list_cod(None)
    elif not isinstance(cod,list):
        cod = [cod]

    return idn,prd,cod

def fun_stats(arr,fun,axs):
    """stats function

    arr -> designed array;
    fun -> selected function;
    axs -> number of the axis.

    """

    if fun == 'score':
        arr = np.sum(np.abs(arr))/arr.shape[axs]

    if fun == 'sum':
        arr = np.sum(arr,axis=axs)

    elif fun == 'size':
        arr = arr.shape[axs]

    elif fun == 'iqr':
        arr = sp.stats.iqr(arr,axis=axs)

    elif fun == 'min':
        arr = np.nanmin(arr,axis=axs)

    elif fun == 'max':
        arr = np.nanmax(arr,axis=axs)

    elif fun == 'mean':
        arr = np.nanmean(arr,axis=axs)

    elif fun == 'median':
        arr = np.nanmedian(arr,axis=axs)

    return arr

def fun_nearest(arr,bnd,ass):
    """nearest value function

    arr -> array of values;
    bnd -> optional boundaries;
    ass -> optional assignments.

    """

    if not isinstance(arr,np.ndarray):
        arr = np.asarray(arr,dtype=float)

    if bnd is None or ass is None:

        # interval
        v_min = min(arr)
        v_max = max(arr)

        if bnd is None:

            if ass is None:
                l_ass = 6
                l_bnd = 5

            else:
                l_ass = len(ass)
                l_bnd = l_ass-1

            # delta
            dlt = abs(v_min-v_max)/l_bnd

            # initialize
            bnd = [np.nan]*l_bnd

            for i in range(l_bnd):
                bnd[i] = v_min+dlt*(0/2+i)

        if ass is None:

            # length
            l_bnd = len(bnd)
            l_ass = l_bnd+1

            # initialize
            ass = [np.nan]*l_ass

            # assignments
            ass[0]  = None
            ass[-1] = None

            for i in range(1,l_bnd):
                ass[i] = (bnd[i-1]+bnd[i])/2

    # size
    siz = len(arr)

    # initialize
    val = [np.nan]*siz

    # boundaries
    bnd_lw = 0
    bnd_up = len(bnd)-1

    for i in range(siz):

        if arr[i] <= bnd[0]:
            val[i] = ass[0]

        for j in range(bnd_lw,bnd_up):

            if arr[i] >= bnd[j] and arr[i] <= bnd[j+1]:
                val[i] = ass[j+1]
                break

        if arr[i] >= bnd[-1]:
            val[i] = ass[-1]

    if siz == 1:
        val = val[0]

    return val

def fun_metrics(vfd,vfc,nam):
    """metrics evaluation function

    vfd -> variable freqency distribution;
    vfc -> variable freqency curve;
    nam -> variable name.

    """

    # model
    mod = stats_model(vfd,vfc,nam)

    # errors
    err = stats_errors(mod,'observations','model')

    # metrics
    mtr = mod.copy()

    # columns
    mtr.loc[:,'absolute'] = err[0]
    mtr.loc[:,'relative'] = err[1]
    mtr.loc[:,'normal']   = err[2]

    return mtr

def fun_fit(dtf,ttl,fun):
    """fitting function

    dtf -> values dataframe;
    ttl -> data titles;
    fun -> fitting function.

    """

    # data values
    x = dtf.loc[:,ttl[0]].to_numpy()
    y = dtf.loc[:,ttl[1]].to_numpy()

    # fitting the values
    opt,cov = sp.optimize.curve_fit(fun,x,y)

    # fitted values
    f = fun(x,*opt)

    # generating dataframe
    fit = pd.DataFrame(data=zip(x,y,f),columns=[ttl[0],ttl[1],'fit'])

    return fit,opt,cov

def fun_peaks(wtb,sen,prc,ext,*args):
    """peaks function

    wtb  -> working dataframe;
    sen  -> sensitivity for the extraction;
    prc  -> percentile for peaks threshold;
    ext  -> extraction mode;
    args -> specific parameters for the framework.

    """

    # max/min analyses
    wtb = peaks_stationary(wtb,sen)

    if ext == 'mev':
        wtb,out = peaks_mev(wtb,*args)
    elif ext == 'phev':
        wtb,out = peaks_phev(wtb,*args)

    if prc is None:
        thr = None
    else:
        wtb,thr = peaks_percentile(wtb,prc)

    # peaks/exceeeds
    pks,exc = peaks_results(wtb)

    return wtb,thr,pks,exc,out

def fun_timeout(sec):
    """timeout function

    sec -> timeout seconds.

    """

    # library
    import multiprocessing.pool

    def decorator(item):
        """wrap the original function
        """

        @functools.wraps(item)
        def wrapper(*args,**kwargs):
            """closure for function
            """

            # threading
            thr = multiprocessing.pool.ThreadPool(processes=1)
            res = thr.apply_async(item,args,kwargs)

            return res.get(sec)

        return wrapper

    return decorator

def vfc_simple(cdf,val,nam):
    """variable frequency curve simple

    cdf -> cumulative density function;
    val -> variable values;
    nam -> variable name.

    """

    if not isinstance(val,np.ndarray):

        # if no valid input, array is set from default one
        val = fun_values([0.01,1,100],[0.001,0.1],None)

    # frequency distribution
    vfc = stats_distribution(cdf,val,nam)

    return vfc

def vfc_quantile(prb,thr,val,obj):
    """variable frequency curve quantile

    prb -> non-exceedance probability;
    thr -> threshold for exceedances;
    val -> first value to be given to fsolve;
    obj -> objective function function.

    """

    # epsilon
    eps = 1e-7

    # delta
    dlt = 1e-1

    # limits
    x_lw = 1e-5
    x_up = 1e+5

    # non-exceedance probability
    if not isinstance(prb,np.ndarray):

        # if no valid input, array is set from default one
        prb = fun_values([0.001,0.025,0.975,0.999],[0.001,0.01,0.001],None)

    # threshold for exceedances
    if not isinstance(thr,int) and not isinstance(thr,float):

        # if no valid input, threshold is set from default one
        thr = 0

    # size
    siz = len(prb)

    # stats
    D  = 1-prb
    RP = 1/D

    # initialize
    err = np.ones(siz)*np.nan
    dff = np.ones(siz)*np.nan
    qnt = np.ones(siz)*np.nan
    flg = np.ones(siz)*np.nan

    # progress
    qnt_prg = 1
    qnt_len = len(qnt)

    # initial value
    x_0 = val

    if np.isnan(x_0) or x_0 < 0:
        x_0 = x_lw

    while True:

        try:
            fun     = lambda x: obj(x,prb[-1])
            res     = sp.optimize.fsolve(fun,x_0,fprime=None,maxfev=0,full_output=1)
            sol,inf = res[0],res[1]
            err[-1] = inf['fvec']
            dff[-1] = x_0-sol

        except:
            fun_print(None,r'Quantile: %3s of %s encountered value overflow'%(qnt_prg,qnt_len))
            err[-1] = np.inf
            pass

        if err[-1] < eps and abs(dff[-1]) > eps and sol > 0:
            fun_print(None,r'Quantile: %3s of %s optimization completed'%(qnt_prg,qnt_len))
            qnt_prg = qnt_prg+1
            qnt[-1] = sol
            flg[-1] = 1
            break

        else:

            if x_0 <= x_up:
                fun_print(None,r'Quantile: %3s of %s changing initial value'%(qnt_prg,qnt_len))
                x_0     = x_0*(1+dlt)
                flg[-1] = 0
                pass

            else:
                fun_print(None,r'Quantile: %3s of %s quitting without solution'%(qnt_prg,qnt_len))
                flg[-1] = -1
                raise

    for i in range(siz-2,-1,-1):

        # initial value
        x_0 = qnt[i+1]

        if np.isnan(x_0) or x_0 < 0:
            x_0 = x_up

        while True:

            try:
                fun     = lambda x: obj(x,prb[i])
                res     = sp.optimize.fsolve(fun,x_0,fprime=None,maxfev=0,full_output=1)
                sol,inf = res[0],res[1]
                err[i]  = inf['fvec']
                dff[i]  = x_0-sol

            except:
                fun_print(None,r'Quantile: %3s of %s encountered value overflow'%(qnt_prg,qnt_len))
                err[i] = np.inf
                pass

            if err[i] < eps and abs(dff[i]) > eps and sol > 0:
                fun_print(None,r'Quantile: %3s of %s optimization completed'%(qnt_prg,qnt_len))
                qnt_prg = qnt_prg+1
                qnt[i]  = sol
                flg[i]  = 1
                break

            else:

                if x_0 >= x_lw:
                    fun_print(None,r'Quantile: %3s of %s changing initial value'%(qnt_prg,qnt_len))
                    x_0    = x_0*(1/(1+dlt))
                    flg[i] = 0
                    pass

                else:
                    fun_print(None,r'Quantile: %3s of %s quitting without solution'%(qnt_prg,qnt_len))
                    flg[i] = -1
                    break

    if siz == sum(flg):
        fun_print(None,'Quantile: curve success')
        pass

    else:
        fun_print(None,'Quantile: curve fail')
        raise

    # adjustments
    qnt = qnt+thr

    # frequency curve
    vfc = pd.DataFrame(data=zip(qnt,prb,D,RP),columns=['q','F','D','RP'])

    return vfc

def optimization_model(data,est,cmp,use,d,typ):
    """model optimization of parameter(s)
    """

    # library
    from statsmodels.base import model

    # dictionaries
    dic_par = dic_parameters(None)
    dic_sol = dic_solvers(None)

    # scaling values
    scl = False

    # parameters number
    num = len(dic_par)

    # parameters initialize
    par = np.ones(num)*np.nan

    # parameters check
    chk = 1e-2

    # solver(s)
    sol = ['Nelder-Mead','TNC','SLSQP']

    # deep domain digging
    ddd = True

    # negative log likelihood figure
    fig = False

    # solver(s) complete output
    out = True

    # solver(s) convergence display
    dsp = False

    # solver(s) skip hessian
    hss = True

    # optimization rewinds
    rwn = 5

    # optimization iterations
    itr = 2500

    # optimization tolerance
    tol = 1e-7

    class pdfq(model.GenericLikelihoodModel):

        def __init__(self,endog,exog=None,**kwds):

            # endog -> dependent variable
            # exog  -> independent variable

            if exog is None:
                exog = np.zeros_like(endog)

            if fig:
                self.nll = pd.DataFrame(columns=[dic_par[0]['symbol'],dic_par[1]['symbol'],dic_par[2]['symbol'],dic_par[3]['symbol'],'NLL'])

            super(pdfq,self).__init__(endog,exog,**kwds)

        def nloglikeobs(self,params):

            # degrees of freedom
            self.df_model = cmp.count(True)
            self.df_resid = num-self.df_model

            for j,i in enumerate(range(num)):
                if cmp[i]:
                    par[i] = params[j]

            # distribution
            pdf = phev_distribution(par,d,scl,typ,'p')

            # values
            val = -np.log(pdf(self.endog))

            # negative log likelihood
            nll = np.sum(val)

            if fig:
                self.nll.loc[len(self.nll)] = [par[0],par[1],par[2],par[3],nll]

            return nll

        def fit(self,start_params,method,maxiter,full_output,disp,skip_hessian,**kwds):

            # fitting
            fun = super(pdfq,self).fit(start_params=start_params,method=method,maxiter=maxiter,full_output=full_output,disp=disp,skip_hessian=skip_hessian,**kwds)

            return fun

    # codification
    cmp = cod_flag(cmp)
    use = cod_flag(use)

    # initialize
    clb = []
    frs = []
    bnd = []

    for i in range(num):

        if not cmp[i]:
            par[i] = est[i]
            continue

        elif use[i]:
            par[i] = round(est[i],2)

        else:
            par[i] = dic_par[i]['reference']

        if par[i] <= dic_par[i]['min']+chk:
            par[i] = dic_par[i]['reference']

        if par[i] >= dic_par[i]['max']-chk:
            par[i] = dic_par[i]['reference']

        # calibration
        clb.append(dic_par[i]['symbol'])

        # input
        frs.append(par[i])

        # boundaries
        bnd.append((dic_par[i]['min'],dic_par[i]['max']))

    # model
    mdl = pdfq(data)

    # lists
    l_par = []
    l_nll = []

    if ddd:
        rwn_exp = int(round_exc(rwn*len(clb)*2,5))
        rwn_att = int(round_exc(rwn*(1+len(clb)/2),5))
    else:
        rwn_exp = 1
        rwn_att = 1

    for mth in sol:

        # rejected
        rjc = None

        if dic_sol[mth]:

            if dic_sol[mth]['iterations'] < itr:
                fun_print(None,'Solver: not accomplishing iterations')
                rjc = True

            if dic_sol[mth]['tolerance'] > tol:
                fun_print(None,'Solver: not accomplishing tolerance')
                rjc = True

            if rjc:
                fun_print(None,'Solver: rejecting method')
                continue

            else:
                fun_print(None,'Solver: selecting method')
                rjc = False

        else:
            fun_print(None,'Solver: unknonwn method')
            continue

        # input
        inp = frs.copy()

        # skipping
        skp = None

        for exp in range(1,rwn_exp+1):

            if skp:
                fun_print(None,'Solver: skipping method')
                break

            for att in range(1,rwn_att+1):

                if exp == 1 and att == 1:
                    fun_print(None,'Solver: parameters defaults')
                    flg = None
                    evl = [flg]*sum(cmp)

                else:

                    # previous parameter(s)
                    pre = inp.copy()

                    for j,i in enumerate(range(num)):

                        if not cmp[i]:
                            continue

                        if not evl[j]:

                            while True:

                                # delta
                                dlt = 0

                                while dlt == 0:
                                    dlt = random.uniform(-dic_par[i]['delta'],+dic_par[i]['delta'])

                                # input
                                inp[j] = dic_par[i]['reference']+dlt
                                inp[j] = round(inp[j],2)

                                if abs(frs[j]-inp[j]) > chk and abs(pre[j]-inp[j]) > chk and dic_par[i]['min']+chk <= inp[j] <= dic_par[i]['max']-chk:
                                    fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'changing input')
                                    break

                # printing
                fun_print(None,'----------------------------------------------------------------------------------------------------')
                fun_print(None,'OPTIMIZATION')
                fun_print(None,'parameters:  ',*clb)
                fun_print(None,'input:       ',*inp)
                fun_print(None,'method:      ',mth)
                fun_print(None,'exploration: ',exp)
                fun_print(None,'attempt:     ',att)
                fun_print(None,'----------------------------------------------------------------------------------------------------')

                try:
                    fun_print(None,'Solver: minimization started')
                    mdl.fit(inp,'minimize',itr,out,dsp,hss,min_method=mth,bounds=bnd,tol=tol)

                except:
                    fun_print(None,'Solver: minimization interrupted')
                    flg = False

                    if att > int(rwn_att/2):
                        fun_print(None,'Solver: minimization stalled')
                        skp = True
                        break

                else:
                    fun_print(None,'Solver: minimization finished')
                    flg = None
                    evl = [flg]*sum(cmp)

                    # parameter index
                    j = 0

                    for i in range(num):

                        if not cmp[i]:
                            continue

                        elif abs(par[i]-inp[j]) < chk:
                            fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'overlaps input value')
                            evl[j] = False

                        elif par[i] <= dic_par[i]['min']+chk:
                            fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'exceeds minimum limit')
                            evl[j] = False

                        elif par[i] >= dic_par[i]['max']-chk:
                            fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'exceeds maximum limit')
                            evl[j] = False

                        else:
                            fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'respects conditions')
                            evl[j] = True

                        # parameter index
                        j = j+1

                    # flagging
                    flg = all(evl)

                if flg:

                    # distribution
                    pdf = phev_distribution(par.copy(),d,scl,typ,'p')

                    # values
                    val = -np.log(pdf(data))

                    # summatory
                    nll = np.sum(val)

                    # appending
                    l_par.append(par.copy())
                    l_nll.append(nll.copy())

                    # resetting
                    flg = None
                    evl = [flg]*sum(cmp)

                    fun_print(None,'Solver: parameters accepted')
                    break

                if att == rwn_att:

                    # resetting
                    flg = None
                    evl = [flg]*sum(cmp)

                    fun_print(None,'Solver: attempts limit')
                    break

                else:
                    fun_print(None,'Solver: next attempt')
                    continue

            if exp == rwn_exp:
                fun_print(None,'Solver: explorations limit')
                break

            else:

                # resetting
                flg = None
                evl = [flg]*sum(cmp)

                fun_print(None,'Solver: next exploration')
                continue

        if mth != sol[-1]:
            fun_print(None,'Solver: next method')
            continue

    try:

        # index
        idx = np.argmin(l_nll)

        # parameters
        par = l_par[idx]

        # dataframe
        dtf                   = pd.DataFrame(data=l_par,columns=[dic_par[key]['symbol'] for key in dic_par.keys()])
        dtf.loc[:,'function'] = l_nll

    except:
        fun_print(None,'Solver: calibration failure')
        raise

    else:
        fun_print(None,'Solver: calibration success')
        pass

    return par,dtf

def optimization_roadrunner(data,est,cmp,use,d,typ):
    """roadrunner optimization of parameter(s)
    """

    # dictionaries
    dic_par = dic_parameters(None)
    dic_sol = dic_solvers(None)

    # objective function
    fun = phev_nll

    # scaling values
    scl = False

    # parameters check
    chk = 5e-3

    # solver(s)
    sol = ['Nelder-Mead','TNC','SLSQP']

    # deep domain digging
    ddd = True

    # solver convergence display
    dsp = False

    # optimization rewinds
    rwn_exp = 10
    rwn_att = 10

    # optimization iterations
    itr_lw = 2000
    itr_up = 3000

    # optimization tolerance
    tol_lw = 1e-7
    tol_up = 1e-9

    # codification
    cmp = cod_flag(cmp)
    use = cod_flag(use)

    # parameters indexes
    i_opt = [i for i,x in enumerate(cmp) if x]
    i_cst = [i for i,x in enumerate(cmp) if not x]

    # symbols
    s_opt = [dic_par[i]['symbol'] for i in i_opt]
    s_cst = [dic_par[i]['symbol'] for i in i_cst]

    # parameters number
    n_par = len(dic_par)
    n_opt = len(s_opt)
    n_cst = len(s_cst)

    # parameters array
    par   = np.ones(n_par)*np.nan
    a_frs = np.ones(n_opt)*np.nan
    a_cst = np.ones(n_cst)*np.nan

    if n_opt > 0:
        fun_print(None,'Solver:',n_opt,'parameter(s) to optimize')
        pass
    else:
        fun_print(None,'Solver: no parameter(s) to optimize')
        raise

    if n_cst > 0:
        par[i_cst] = est[i_cst]
        pass

    # boundaries
    bnd = []

    # columns
    col = {'method':str,'function':float,'evaluations':int,'iterations':int,'flag':bool,'difficulties':bool,'exploration':int,'attempt':int}

    # dataframe
    dtf              = pd.DataFrame({c:pd.Series(dtype=t) for c,t in col.items()})
    dtf.loc[:,s_opt] = np.nan

    for n,i in zip(range(n_opt),i_opt):

        if use[i]:
            a_frs[n] = round(est[i],2)

        else:
            a_frs[n] = dic_par[i]['reference']

        if a_frs[n] <= dic_par[i]['min']+chk:
            a_frs[n] = dic_par[i]['reference']

        if a_frs[n] >= dic_par[i]['max']-chk:
            a_frs[n] = dic_par[i]['reference']

        # boundaries
        bnd.append((dic_par[i]['min'],dic_par[i]['max']))

    # costants
    a_cst = est[i_cst]

    if ddd:
        rwn_exp = int(rwn_exp*n_opt)
        rwn_att = int(rwn_att*n_opt)
    else:
        rwn_exp = int(n_opt)
        rwn_att = int(n_opt)

    # dataframe
    dtf.loc[0,col.keys()] = 'preliminary',fun(a_frs,a_cst,cmp,d,scl,typ,data),0,0,False,False,0,0
    dtf.loc[0,s_opt]      = a_frs

    for mth in sol:

        # rejected
        rjc = None

        if not dic_sol[mth]:
            fun_print(None,'Solver: unknonwn method')
            continue

        # input
        a_inp = a_frs.copy()

        # skipping
        skp = None

        for exp in range(1,rwn_exp+1):

            # settings
            itr = itr_lw
            tol = tol_up

            if skp:
                fun_print(None,'Solver: skipping method')
                break

            if dic_sol[mth]['iterations'] < itr:
                fun_print(None,'Solver: not accomplishing iterations')
                rjc = True

            if dic_sol[mth]['tolerance'] > tol:
                fun_print(None,'Solver: not accomplishing tolerance')
                rjc = True

            if rjc:
                fun_print(None,'Solver: rejecting method')
                continue

            else:
                fun_print(None,'Solver: selecting method')
                rjc = False

            for att in range(1,rwn_att+1):

                if att == 1:
                    exc = 0
                    dif = False

                if att == 1 and exp == 1:
                    fun_print(None,'Solver: parameters defaults')
                    flg = None
                    evl = [flg]*n_opt

                else:

                    # previous parameter(s)
                    a_pre = a_inp.copy()

                    for n,i in zip(range(n_opt),i_opt):

                        if not evl[n]:

                            while True:

                                # delta
                                dlt = 0

                                while dlt == 0:
                                    dlt = random.uniform(-dic_par[i]['delta'],+dic_par[i]['delta'])

                                # input
                                a_inp[n] = dic_par[i]['reference']+dlt
                                a_inp[n] = round(a_inp[n],2)

                                if abs(a_frs[n]-a_inp[n]) > chk and abs(a_pre[n]-a_inp[n]) > chk and dic_par[i]['min']+chk <= a_inp[n] <= dic_par[i]['max']-chk:
                                    fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'changing input')
                                    break

                if dif:

                    # flagging
                    chn = None

                    # changing settings
                    itr_nxt = itr+int(5e+2)
                    tol_nxt = tol*float(1e+1)

                    if not chn:
                        if itr_nxt <= itr_up:
                            fun_print(None,'Solver: iterations value changing')
                            chn = True
                            itr = itr_nxt
                        else:
                            fun_print(None,'Solver: iterations limit reached')
                            chn = False

                    if not chn:
                        if tol_nxt <= tol_lw:
                            fun_print(None,'Solver: tolerance value changing')
                            chn = True
                            tol = tol_nxt
                        else:
                            fun_print(None,'Solver: tolerance limit reached')
                            chn = False

                    if chn:
                        fun_print(None,'Solver: settings altered')
                    else:
                        fun_print(None,'Solver: settings unalterables')

                # initialize
                val     = np.nan
                n_e,n_i = np.nan,np.nan
                opt     = np.ones(n_opt)

                # printing
                fun_print(None,'----------------------------------------------------------------------------------------------------')
                fun_print(None,'OPTIMIZATION')
                fun_print(None,'symbols:     ',*s_opt)
                fun_print(None,'input:       ',*a_inp)
                fun_print(None,'method:      ',mth)
                fun_print(None,'iterations:  ',itr)
                fun_print(None,'tolerance:   ',tol)
                fun_print(None,'exploration: ',exp)
                fun_print(None,'attempt:     ',att)
                fun_print(None,'----------------------------------------------------------------------------------------------------')

                try:
                    fun_print(None,'Solver: minimization started')
                    res = sp.optimize.minimize(fun,a_inp,args=(a_cst,cmp,d,scl,typ,data),method=mth,bounds=bnd,tol=tol,options={'maxiter':itr,'disp':dsp})

                except:
                    fun_print(None,'Solver: minimization interrupted')
                    flg = False
                    exc = exc+1

                    if exc > int(rwn_att/2):
                        fun_print(None,'Solver: minimization stalled')
                        skp = True
                        break

                    if exc > int(rwn_att/8) and not dif:
                        fun_print(None,'Solver: minimization difficulties')
                        dif = True
                        pass

                else:

                    if exc != 0:
                        exc = 0

                    if res['success']:

                        fun_print(None,'Solver: minimization finished')
                        flg     = None
                        evl     = [flg]*sum(cmp)
                        val     = res['fun']
                        n_e,n_i = res['nfev'],res['nit']
                        opt     = res['x']

                        if abs(val) == np.inf:
                            fun_print(None,'Solver: function value error')
                            flg = False

                        else:

                            for n,i in zip(range(n_opt),i_opt):

                                if abs(res['x'][n]-a_inp[n]) < chk:
                                    fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'overlaps input value')
                                    evl[n] = False

                                elif res['x'][n] <= bnd[n][0]+chk:
                                    fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'exceeds minimum limit')
                                    evl[n] = False

                                elif res['x'][n] >= bnd[n][1]-chk:
                                    fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'exceeds maximum limit')
                                    evl[n] = False

                                else:
                                    fun_print(None,'Solver: parameter',dic_par[i]['symbol'],'respects conditions')
                                    evl[n] = True

                            # flagging
                            flg = all(evl)

                    else:

                        fun_print(None,'Solver: minimization failed')
                        flg = False

                    # indexing
                    idx = len(dtf)

                    # dataframe
                    dtf.loc[idx,col.keys()] = mth,val,n_e,n_i,flg,dif,exp,att
                    dtf.loc[idx,s_opt]      = opt

                if flg:

                    # resetting
                    flg = None
                    evl = [flg]*n_opt

                    fun_print(None,'Solver: parameters accepted')
                    break

                if att > int(rwn_att/4) and not dif:
                    fun_print(None,'Solver: minimization difficulties')
                    dif = True

                if att == rwn_att:

                    # resetting
                    flg = None
                    evl = [flg]*n_opt

                    fun_print(None,'Solver: attempts limit')
                    break

                else:
                    fun_print(None,'Solver: next attempt')
                    continue

            if exp == rwn_exp:
                fun_print(None,'Solver: explorations limit')
                break

            else:

                # resetting
                flg = None
                evl = [flg]*n_opt

                fun_print(None,'Solver: next exploration')
                continue

        if mth != sol[-1]:
            fun_print(None,'Solver: next method')
            continue

    try:

        # selecting
        sel = dtf[dtf.loc[:,'flag'].to_numpy().astype(bool)*(dtf.loc[:,'function'].to_numpy() < dtf.loc[:,'function'].mean(numeric_only=True))]

        # indexing
        idx = sel.loc[:,'function'].idxmin()

        # parameters
        par[i_opt] = dtf.loc[idx,s_opt].to_numpy()

    except:
        fun_print(None,'Solver: calibration failure')
        raise

    else:
        fun_print(None,'Solver: calibration success')
        pass

    return par,dtf

def framework_mev(dtb,idn,prd,cod,yea,scl,dis):
    """framework MEV
    """

    # parameters initialize
    # par[0]: shape -> shape parameter of the distribution
    # par[1]: scale -> scale parameter of the distribution
    par = np.ones(2)*np.nan

    # work table
    wtb = table_work(dtb,idn)
    wtb = table_clean(wtb,'streamflow')
    wtb = table_adjust(wtb,'streamflow',0.1)
    wtb = table_select(wtb,yea,prd)

    # streamflow maximas
    q_yea = stats_maxima(wtb,'streamflow')

    # streamflow stats
    q_min,q_max,q_avg = stats_evaluation(wtb,'streamflow')

    # analyses
    wtb,q_thr,q_pks,exc,out = fun_peaks(wtb,None,None,'mev',None,None,None)

    if scl:
        q_pks = q_pks/q_avg

    if dis == 'weibull':

        # fitting weibull distribution via PWMs
        par[0],par[1] = par_weibull(q_pks)

    if dis == 'gamma':

        # fitting gamma distribution via L-moments
        par[0],par[1] = par_gamma(q_pks)

    if scl:
        ffd = stats_empirical(q_yea/q_avg,'q')
    else:
        ffd = stats_empirical(q_yea,'q')

    # flood frequency curve
    ffc = mev_ffc_quantile(ffd,q_thr,q_avg,dis,par,exc)

    # projection evaluations
    mtr = fun_metrics(ffd,ffc,'q')

    # table titles
    ttl = ['parameters','distributions','curves','metrics']

    # output
    out = par,ffd,ffc,mtr

    # table creation
    dtf = table_multi(idn,prd,cod,ttl)

    # packing
    dtf.loc[(idn,prd)] = np.asarray(out,dtype=object)

    return dtf

def framework_phev(dtb,idn,prd,cod,yea,scl,ext,typ,cmp,use):
    """framework PHEV
    """

    if isinstance(cmp,bool) and cmp or cmp is None:
        cmp = cod

    if isinstance(use,bool) and use or use is None:
        use = cod

    # dictionary
    dic = dic_codification(None)

    # peaks initialize
    pks_phev = np.nan
    pks_mev  = np.nan

    # recessions initialize
    rec_arr = np.nan
    rec_cdr = np.nan
    rec_fit = np.nan
    rec_slp = np.nan

    # flood frequency distributions initialize
    ffd_pks = np.nan
    ffd_yea = np.nan

    # parameters initialize
    # par[0]: alpha -> average amount of precipitation on rainy days
    # par[1]: labda -> ratio between long term average streamflow and alpha
    # par[2]: a     -> recession distribution power law exponent
    # par[3]: k     -> recession distribution power law coefficient
    par_est = np.ones(4)*np.nan
    par_clb = np.ones(4)*np.nan

    # flood frequency curves initialize
    ffc_typ  = np.nan
    ffc_phev = np.nan
    ffc_mev  = np.nan

    # metrics initialize
    mtr_typ  = np.nan
    mtr_phev = np.nan
    mtr_mev  = np.nan

    # flags initialize
    flg_rec = np.nan
    flg_est = np.nan
    flg_clb = np.nan
    flg_mod = np.nan
    flg_ffc = np.nan

    try:
        siz = dtb.loc[idn,'META'].loc['area']
    except:
        siz = None

    # work table
    wtb = table_work(dtb,idn)
    wtb = table_clean(wtb,['precipitation','streamflow'])
    wtb = table_adjust(wtb,['precipitation','streamflow'],[0.1,0.1])
    wtb = table_select(wtb,yea,prd)

    # evaluating stats
    q_min,q_max,q_avg = stats_evaluation(wtb,'streamflow')

    # streamflow maximas
    q_yea = stats_maxima(wtb,'streamflow')

    # days of period
    d = days_period(prd)

    # sensitivity for the extraction
    p_sen = None
    # percentile for peaks threshold
    p_prc = None

    # PHEV peaks analyses
    # opt -> computation options for recessions
    # win -> smoothing window for computing second derivative for concavity
    # lng -> recessions minimum length
    # mod -> recessions fitting mode
    wtb_phev,thr_phev,q_phev,exc_phev,out_phev = fun_peaks(wtb,p_sen,p_prc,'phev',1,5,5,1)
    pks_phev                                   = wtb_phev,thr_phev,q_phev,exc_phev

    if not out_phev[-1]:
        fun_print(None,'Phev: limited recessions number')
        wtb_phev,thr_phev,q_phev,exc_phev,out_phev = fun_peaks(wtb,p_sen,p_prc,'phev',1,5,3,1)
        pks_phev                                   = wtb_phev,thr_phev,q_phev,exc_phev

    # output unpacking
    rec_par,rec_arr,rec_cdr,rec_fit,rec_slp,flg_rec = out_phev

    # MEV peaks analyses
    # lag -> value used to lag
    # siz -> size of the basin
    # drp -> drop factor for streamflow
    wtb_mev,thr_mev,q_mev,exc_mev,out_mev = fun_peaks(wtb,p_sen,p_prc,'mev',None,siz,0.75)
    pks_mev                               = wtb_mev,thr_mev,q_mev,exc_mev

    if ext == 'phev':
        q_thr,q_pks,exc = None,q_phev,exc_phev
    elif ext == 'mev':
        q_thr,q_pks,exc = None,q_mev,exc_mev

    if scl:
        ffd_pks = stats_empirical(q_pks/q_avg,'q')
        ffd_yea = stats_empirical(q_yea/q_avg,'q')
    else:
        ffd_pks = stats_empirical(q_pks,'q')
        ffd_yea = stats_empirical(q_yea,'q')

    if cmp in dic['analyses']:

        # results
        pks = pks_phev,pks_mev
        rec = rec_arr,rec_cdr,rec_fit,rec_slp
        ffd = ffd_pks,ffd_yea
        flg = flg_rec,flg_est,flg_clb,flg_mod,flg_ffc

        fun_print(None,'Phev: returning analyses')
        ttl = ['peaks','recessions','distributions','flags']

        # output
        out = pks,rec,ffd,flg

    if cmp in dic['model']:

        if typ == 'ordinary':
            q_typ,ffd_typ = q_pks,ffd_pks
        elif typ == 'maxima':
            q_typ,ffd_typ = q_yea,ffd_yea

        if cmp in dic['0p'] or use in dic['calibration']:

            try:

                # mean rainfall depth
                par_est[0],std_alp,data_alp = par_alpha(wtb)

                # effective rainfall frequency
                par_est[1],std_lam,data_lam = par_labda(wtb)

                # recessions exponent
                par_est[2] = rec_par[0].copy()

                # recessions coefficient
                par_est[3] = rec_par[1].copy()

                # printing
                fun_print(2,'----------------------------------------------------------------------------------------------------')
                fun_print(2,'ESTIMATION')
                fun_print(2,r'alpha: %1s'%(par_est[0]))
                fun_print(2,r'labda: %1s'%(par_est[1]))
                fun_print(2,r'a:     %1s'%(par_est[2]))
                fun_print(2,r'k:     %1s'%(par_est[3]))
                fun_print(2,'----------------------------------------------------------------------------------------------------')

            except:
                fun_print(None,'Phev: estimation fail')
                flg_est = False
                flg_mod = False

            else:
                fun_print(None,'Phev: estimation success')
                flg_est = None
                flg_mod = True

                # days of precipitation
                prc_d,prc_f = days_precipitation(wtb)

                if prc_f < par_est[1]:
                    fun_print(None,'Phev: rainfall frequency is physically inconsistent if there is no snow')
                    flg_est = False

                else:
                    fun_print(None,'Phev: rainfall frequency is physically consistent')
                    flg_est = True

        if cmp in dic['0p']:

            # assign
            par = par_est

        if cmp in dic['calibration']:

            try:

                # parameter(s) calibration
                par_clb,par_dtf = optimization_roadrunner(q_typ,par_est,cmp,use,d,typ)

                # printing
                fun_print(2,'----------------------------------------------------------------------------------------------------')
                fun_print(2,'CALIBRATION')
                fun_print(2,r'alpha: %1s'%(par_clb[0]))
                fun_print(2,r'labda: %1s'%(par_clb[1]))
                fun_print(2,r'a:     %1s'%(par_clb[2]))
                fun_print(2,r'k:     %1s'%(par_clb[3]))
                fun_print(2,'----------------------------------------------------------------------------------------------------')

            except:
                fun_print(None,'Phev: calibration fail')
                flg_clb = False
                flg_mod = False

            else:
                fun_print(None,'Phev: calibration success')
                flg_clb = True
                flg_mod = True

            # assign
            par = par_clb

        if flg_mod:

            try:

                # figures
                fig = False

                # PHI
                mtr_phi = par[1]/(par[3]*(par[0]*par[1])**(par[2]-1))

                # printing
                fun_print(2,'----------------------------------------------------------------------------------------------------')
                fun_print(2,'MODEL: phi')
                fun_print(2,r'value: %1s'%(mtr_phi))
                fun_print(2,'----------------------------------------------------------------------------------------------------')

                # TYPE
                ffc_typ = phev_ffc_quantile(ffd_typ,q_thr,q_avg,par,d,scl,typ)
                mtr_typ = fun_metrics(ffd_typ,ffc_typ,'q')

                # printing
                fun_print(2,'----------------------------------------------------------------------------------------------------')
                fun_print(2,'MODEL: type')
                fun_print(2,r'absolute: %1s'%(mtr_typ['absolute'].mean()))
                fun_print(2,r'relative: %1s'%(mtr_typ['relative'].mean()))
                fun_print(2,r'normal:   %1s'%(mtr_typ['normal'].mean()))
                fun_print(2,'----------------------------------------------------------------------------------------------------')

                # PHEV
                ffc_phev = phev_ffc_quantile(ffd_yea,q_thr,q_avg,par,d,scl,'maxima')
                mtr_phev = fun_metrics(ffd_yea,ffc_phev,'q')

                # printing
                fun_print(2,'----------------------------------------------------------------------------------------------------')
                fun_print(2,'MODEL: phev')
                fun_print(2,r'absolute: %1s'%(mtr_phev['absolute'].mean()))
                fun_print(2,r'relative: %1s'%(mtr_phev['relative'].mean()))
                fun_print(2,r'normal:   %1s'%(mtr_phev['normal'].mean()))
                fun_print(2,'----------------------------------------------------------------------------------------------------')

                # MEV
                ffc_mev = mev_ffc_quantile(ffd_yea,q_thr,q_avg,'phev',par,exc,d,scl,'ordinary')
                mtr_mev = fun_metrics(ffd_yea,ffc_mev,'q')

                # printing
                fun_print(2,'----------------------------------------------------------------------------------------------------')
                fun_print(2,'MODEL: mev')
                fun_print(2,r'absolute: %1s'%(mtr_mev['absolute'].mean()))
                fun_print(2,r'relative: %1s'%(mtr_mev['relative'].mean()))
                fun_print(2,r'normal:   %1s'%(mtr_mev['normal'].mean()))
                fun_print(2,'----------------------------------------------------------------------------------------------------')

                if fig:
                    hnd = []
                    hnd.append(mp.pyplot.Line2D([0],[0],markersize=5,marker='o',label='observed',linestyle='None'))
                    hnd.append(mp.pyplot.Line2D([0],[0],markersize=0,marker='o',label='model'))
                    ext = {'fomat':{'legendloc':None,'legendhandle':hnd,'xlabel':'RP [events]','ylabel':'q [-]'}}
                    post_plot(None,'ffc_type_%s_%s_%s'%(idn,prd,cmp),'mlt_ffc',ext,[ffd_typ,ffc_typ],None,['RP','q'],None)
                    ext = {'format':{'legendloc':None,'legendhandle':hnd,'xlabel':'RP [years]','ylabel':'q [-]'}}
                    post_plot(None,'ffc_maxima_phev_%s_%s_%s'%(idn,prd,cmp),'mlt_ffc',ext,[ffd_yea,ffc_phev],None,['RP','q'],None)
                    ext = {'format':{'legendloc':None,'legendhandle':hnd,'xlabel':'RP [years]','ylabel':'q [-]'}}
                    post_plot(None,'ffc_maxima_mev_%s_%s_%s'%(idn,prd,cmp),'mlt_ffc',ext,[ffd_yea,ffc_mev],None,['RP','q'],None)
                    del hnd

            except:
                fun_print(None,'Phev: flood frequency curves fail')
                flg_ffc = False

            else:
                fun_print(None,'Phev: flood frequency curves success')
                flg_ffc = True

        # results
        ffc = ffc_typ,ffc_phev,ffc_mev
        mtr = mtr_typ,mtr_phev,mtr_mev
        flg = flg_rec,flg_est,flg_clb,flg_mod,flg_ffc

        fun_print(None,'Phev: returning model')
        ttl = ['parameters','curves','metrics','flags']

        # output
        out = par,ffc,mtr,flg

    # table creation
    dtf = table_multi(idn,prd,cod,ttl)

    # packing
    dtf.loc[(idn,prd,cod)] = np.asarray(out,dtype=object)

    return dtf

def proc_framework(frm,skp,dtb,idn,prd,cod,yea,*args):
    """framework processing

    frm  -> designed framework;
    skp  -> skip existing entry(es);
    dtb  -> data dataframe for the analyses;
    idn  -> selected idn(s);
    prd  -> selected period(s);
    cod  -> selected codification(s);
    yea  -> selected year(s);
    args -> specific parameters for the framework.

    """

    # subfolder
    sub = 'proc'

    if frm == 'mev':
        fun = framework_mev
    elif frm == 'phev':
        fun = framework_phev

    # lists
    idn,prd,cod = fun_list(idn,prd,cod)

    # lengths
    idn_len,prd_len,cod_len = len(idn),len(prd),len(cod)

    # id progress
    idn_prg = 0

    for i in idn:

        try:
            fil = data_name([frm,i])
            ftb = data_load(sub,fil)

        except:
            fun_print(None,'Framework: loading fail')
            ftb = table_multi(None,None,None,None)
            pass

        else:
            fun_print(None,'Framework: loading success')
            pass

        # index
        idx = ftb.index

        # id progress
        idn_prg = idn_prg+1

        # period progress
        prd_prg = 0

        for p in prd:

            # period progress
            prd_prg = prd_prg+1

            # codification progress
            cod_prg = 0

            for c in cod:

                # codification progress
                cod_prg = cod_prg+1

                # flagging
                flg = None

                # printing
                fun_print(1,'----------------------------------------------------------------------------------------------------')
                fun_print(1,'PROCESSING')
                fun_print(1,'id:           %10s (%1s of %1s)'%(i,idn_prg,idn_len))
                fun_print(1,'period:       %10s (%1s of %1s)'%(p,prd_prg,prd_len))
                fun_print(1,'codification: %10s (%1s of %1s)'%(c,cod_prg,cod_len))
                fun_print(1,'----------------------------------------------------------------------------------------------------')

                if (i,p,c) in idx and skp:
                    fun_print(None,'Framework: existing entry skipping')
                    continue

                elif (i,p,c) in idx and not skp:
                    fun_print(None,'Framework: existing entry updating')
                    ftb.drop((i,p,c),axis=0,inplace=True)

                try:
                    fun_print(None,'Framework: entry processing')
                    dtf = fun(dtb,i,p,c,yea,*args)

                except:
                    fun_print(None,'Framework: entry error')
                    flg = False

                else:
                    fun_print(None,'Framework: entry adding')
                    ftb = pd.concat([ftb,dtf])
                    flg = True

                if flg:
                    fun_print(None,'Framework: processing success')

                    try:
                        fun_print(None,'Framework: saving dataframe')
                        data_save(sub,fil,ftb)

                    except:
                        fun_print(None,'Framework: saving fail')

                else:
                    fun_print(None,'Framework: processing error')

def proc_pack(frm,skp,rem,idn,prd,cod):
    """pack processing

    frm -> designed framework;
    skp -> skip existing entry(es);
    rem -> remove processed entry(es);
    idn -> selected idn(s);
    prd -> selected period(s);
    cod -> selected codification(s).

    """

    # subfolder
    sub = 'proc'

    # lists
    idn,prd,cod = fun_list(idn,prd,cod)

    try:
        fil = data_name(frm)
        ptb = data_load(sub,fil)

    except:
        fun_print(None,'Pack: loading fail')
        ptb = table_multi(None,None,None,None)

    else:
        fun_print(None,'Pack: loading success')

    # index
    idx = ptb.index

    # flagging
    flg = None

    for i in idn:

        try:
            fil = data_name([frm,i])
            ftb = data_load(sub,fil)

        except:
            fun_print(None,'Pack: dataframe',i,'missing')
            continue

        else:

            # multiindex
            mlt = pd.MultiIndex.from_product([[i],prd,cod],names=['ID','period','codification'])

            for m in mlt:

                try:
                    dtf = ftb.loc[[m]]

                except:
                    fun_print(None,'Pack: entry',m,'missing')

                else:
                    fun_print(None,'Pack: entry',m,'found')

                    if m in idx and skp:
                        fun_print(None,'Pack: existing entry',m,'skipping')
                        continue

                    elif m in idx and not skp:
                        fun_print(None,'Pack: existing entry',m,'updating')
                        ptb.drop(m,axis=0,inplace=True)
                        flg = True

                    fun_print(None,'Pack: entry',m,'adding')
                    ptb = pd.concat([ptb,dtf])
                    flg = True

            if rem:
                fun_print(None,'Pack: dataframe',i,'removing')
                fil = data_name([frm,i])
                data_remove(sub,fil)
                continue

    if flg:

        fun_print(None,'Pack: processing altered')

        try:
            fil = data_name(frm)
            data_save(sub,fil,ptb)

        except:
            fun_print(None,'Pack: saving fail')

        else:
            fun_print(None,'Pack: saving success')

    else:
        fun_print(None,'Pack: processing unaltered')

def proc_unpack(frm,skp,rem,idn,prd,cod):
    """unpack processing

    frm -> designed framework;
    skp -> skip existing entry(es);
    rem -> remove processed entry(es);
    idn -> selected idn(s);
    prd -> selected period(s);
    cod -> selected codification(s).

    """

    # subfolder
    sub = 'proc'

    # lists
    idn,prd,cod = fun_list(idn,prd,cod)

    try:
        fil = data_name(frm)
        ptb = data_load(sub,fil)

    except:
        fun_print(None,'Unpack: dataframe',frm,'missing')

    else:

        for i in idn:

            # flagging
            flg = None

            try:
                fil = data_name([frm,i])
                ftb = data_load(sub,fil)

            except:
                fun_print(None,'Unpack: loading fail')
                ftb = table_multi(None,None,None,None)

            # index
            idx = ftb.index

            # multiindex
            mlt = pd.MultiIndex.from_product([[i],prd,cod],names=['ID','period','codification'])

            for m in mlt:

                try:
                    dtf = ptb.loc[[m]]

                except:
                    fun_print(None,'Unpack: entry',m,'missing')

                else:
                    fun_print(None,'Unpack: entry',m,'found')

                    if m in idx and skp:
                        fun_print(None,'Unpack: existing entry',m,'skipping')
                        continue

                    elif m in idx and not skp:
                        fun_print(None,'Unpack: existing entry',m,'updating')
                        ftb.drop(m,axis=0,inplace=True)
                        flg = True

                    fun_print(None,'Unpack: entry',m,'adding')
                    ftb = pd.concat([ftb,dtf])
                    flg = True

            if flg:

                fun_print(None,'Unpack: processing altered')

                try:
                    fil = data_name([frm,i])
                    data_save(sub,fil,ftb)

                except:
                    fun_print(None,'Unpack: saving fail')

                else:
                    fun_print(None,'Unpack: saving success')

            else:
                fun_print(None,'Unpack: processing unaltered')

        if rem:
            fun_print(None,'Unpack: dataframe',frm,'removing')
            fil = data_name(frm)
            data_remove(sub,fil)

def proc_extract(frm,idx,ttl,pos,col,fun):
    """extract processing

    frm -> designed framework;
    idx -> indexer(s);
    ttl -> data title;
    pos -> optional nested position;
    col -> optional column in subdataframe;
    fun -> optional data function.

    """

    if not isinstance(pos,list) and pos is not None:
        pos = [pos]

    if not isinstance(col,list) and col is not None:
        col = [col]

    if idx is None:

        try:
            sub = 'extract'
            fil = data_name([frm,ttl,pos,col,fun])
            dtf = data_load(sub,fil)

        except:
            fun_print(None,'Extract: loading fail')

        else:
            fun_print(None,'Extract: loading success')
            return dtf

    # initialize
    nam = ['ID','period','codification']
    dtf = pd.DataFrame('-',pd.MultiIndex.from_tuples([],names=nam),[])

    try:
        sub = 'proc'
        fil = data_name(frm)
        ptb = data_load(sub,fil)

    except:
        fun_print(None,'Extract: processing',frm,'fail')

    else:
        fun_print(None,'Extract: processing',frm,'started')

        # funtion
        ext = None

        # filtering
        ptb = ptb.filter(items=[ttl])

        if isinstance(idx,pd.MultiIndex):
            pass

        if idx is None:
            flg = True
            idx = ptb.index

        elif isinstance(idx,pd.MultiIndex):
            flg = False
            idx = idx.copy()

        elif isinstance(idx,list):

            for i in range(3):

                if idx[i] == None:
                    idx[i] = ptb.index.get_level_values(i).unique().to_list()

                if not isinstance(idx[i],list):
                    idx[i] = [idx[i]]

            flg = False
            idx = pd.MultiIndex.from_product(idx,names=['ID','period','codification'])

        for i in idx:

            try:

                # object
                obj = ptb.loc[i,ttl]

                if pos is None:

                    if ext is None:
                        ext = extract_data

                else:

                    for p in pos:
                        obj = obj[p]

                    if ext is None:

                        if isinstance(obj,(bool,int,float)):
                            ext = extract_value
                        elif isinstance(obj,np.ndarray):
                            ext = extract_array
                        elif isinstance(obj,pd.DataFrame):
                            ext = extract_dataframe
                        else:
                            fun_print(None,'Extract: unknown type')
                            continue

                # extraction
                dtf = ext(dtf,i,obj,col,fun,nam)

            except:
                fun_print(None,'Extract: entry',i,'missing')
                continue

            else:
                fun_print(None,'Extract: entry',i,'adding')
                if flg is None:
                    flg = True

        if flg:

            try:
                sub = 'extract'
                fil = data_name([frm,ttl,pos,col,fun])
                data_save(sub,fil,dtf)

            except:
                fun_print(None,'Extract: saving fail')

            else:
                fun_print(None,'Extract: saving success')

    return dtf

def post_select(ext,val,cnd,key,drp):
    """select post-processing

    ext -> extracted data;
    val -> score values;
    cnd -> assigned condition;
    key -> filtering key;
    drp -> drop rows.

    """

    if val is None:
        val = ext.copy()

    if drp is None:
        drp = 'any'

    if isinstance(cnd,list):
        i_x = np.invert((val.values > cnd[0])*(val.values < cnd[1]))

    if isinstance(cnd,bool):
        i_x = np.invert(val.values is cnd)

    # marking
    val.loc[i_x] = np.nan

    # indexing
    i_x = ext.index.intersection(val.loc[val.isnull().values].index.unique(),sort=False)
    i_y = ext.index.difference(i_x,sort=False).unique()

    # unstack
    sel = pd.DataFrame(1,i_y,columns=ext.columns).unstack(level=key)

    if isinstance(drp,str):
        sel = sel.dropna(how=drp)

    if isinstance(drp,list):
        sel = sel.dropna(subset=drp)

    # stack
    sel = sel.stack().reorder_levels(ext.index.names)

    # dataframe
    dtf = ext.loc[ext.index.intersection(sel.index,sort=False)]

    return dtf

def post_filter(ext,key,ttl,qnt):
    """filter post_processing

    ext -> extracted data;
    key -> filtering key;
    ttl -> data title;
    qnt -> quantile values

    """

    # initializing
    nam = list(ext.index.names)
    pos = nam.index(key)
    prd = [list(ext.index.get_level_values(l).unique()) for l in range(ext.index.nlevels)]
    fil = ext.index.get_level_values(key).unique()

    # initialize
    ext = ext.reset_index().reset_index().set_index(nam)
    drp = ext.iloc[0:0].reset_index().set_index(nam+['index']).index

    for f in fil:

        # filtering
        ind = [[f] if i == pos else prd[i] for i in range(ext.index.nlevels)]
        mlt = pd.MultiIndex.from_product(ind).intersection(ext.index)

        # quantiles
        q_lw = ext.loc[mlt].loc[:,ttl].quantile(qnt[0])
        q_up = ext.loc[mlt].loc[:,ttl].quantile(qnt[1])

        # dataframe
        dtf = ext.loc[mlt].reset_index().set_index(nam+['index'])

        # indexing
        idx = dtf.loc[(dtf.loc[:,ttl] < q_lw)|(dtf.loc[:,ttl] > q_up).values].index

        # appending
        drp = pd.MultiIndex.append(drp,idx)

    # dataframe entries
    ext            = ext.reset_index().set_index(nam+['index'])
    ext.loc[drp,:] = np.nan
    ext            = ext.droplevel('index').reset_index().set_index(nam)

    return ext

def post_sample(ext,key,crt,col,typ,num):
    """sample post-processing

    ext -> extracted data;
    key -> filtering key;
    crt -> criteria title;
    col -> optional column in subdataframe;
    typ -> sampling type;
    num -> sampling number.

    """

    # initialize
    lvl = [slice(None)]*3

    if key == 'ID':
        j = 0
    if key == 'period':
        j = 1
    if key == 'codification':
        j = 2

    if crt is None:
        crt = ext.index.get_level_values(j).unique().to_list()

    if not isinstance(crt,list):
        crt = [crt]

    # levels
    lvl[j] = crt

    # score
    sco = ext.filter(items=[col]).sort_index(ascending=True).loc[tuple(lvl)]

    if typ != 'ok':

        if typ == 'flag':
            i_x = np.invert(sco.values)
        elif typ == 'up':
            i_x = np.invert(sco.values > 0)
        elif typ == 'lw':
            i_x = np.invert(sco.values < 0)

        # marking
        sco.loc[i_x] = np.nan

        # indexing
        i_x = sco.loc[sco.isnull().values].index.unique()
        i_y = sco.index.difference(i_x).unique()

        # dataframe
        sco = sco.loc[i_y]

    # index labels
    lbl = [x for x in sco.index.names if x not in set([key])]

    # pivot table
    sam = pd.pivot_table(sco,values=col,index=lbl,aggfunc=np.mean)
    sam = table_rename(sam,col,'score')

    if typ == 'flag':
        pass

    else:

        if typ == 'ok':
            asc = False

        else:
            asc = True

        # adjustments
        sam = sam.abs()
        sam = sam.sort_values(by='score',ascending=asc)
        sam = sam.tail(num)

    # dataframe
    dtf = ext.reset_index().set_index(lbl).loc[sam.index].reset_index().set_index(ext.index.names)

    return dtf

def post_case(dtf,key,idx,ttl):
    """case post-processing

    dtf -> values dataframe;
    key -> filtering key;
    idx -> case indexer(s);
    ttl -> case title(s).

    """

    if key is None:
        key = dtf.index.names

    # cases
    cas = len(idx)

    for i in range(cas):

        try:

            # initialize
            tmp = None

            # filter
            fil = [n for n in dtf.index.names if (n in idx[i].names and n in key)]

            # dropping
            drp_dtf = [p for p in range(dtf.index.nlevels) if dtf.index.names[p] not in fil]
            drp_idx = [p for p in range(idx[i].nlevels) if idx[i].names[p] not in fil]

            # adjustments
            tmp = dtf.reset_index().reset_index().set_index(fil+['index']).droplevel(-1)

            # indexing
            mlt = pd.MultiIndex.intersection(dtf.index.droplevel(drp_dtf),idx[i].droplevel(drp_idx))

            # assignment
            tmp.loc[mlt,'case'] = ttl[i]

            # setting
            dtf = tmp.reset_index().set_index(dtf.index.names)

        except:
            fun_print(None,'Case: post-processing error')
            pass

    # dropping
    dtf = dtf.dropna(subset='case')

    return dtf

def post_side(dtf,ext,ttl,ren):
    """side post-processing

    dtf -> values dataframe;
    ext -> extracted dataframe(s);
    ttl -> column title(s);
    ren -> optional rename(s).

    """

    if not isinstance(ext,list):
        ext = [ext]

    if not isinstance(ttl,list):
        ttl = [ttl]

    # size
    siz = len(ext)

    for i in range(siz):

        try:

            # initialize
            tmp_dtf = None
            tmp_ext = None

            # filter
            fil = [n for n in dtf.index.names if n in ext[i].index.names]

            # dropping
            drp_dtf = [p for p in range(dtf.index.nlevels) if dtf.index.names[p] not in fil]
            drp_ext = [p for p in range(ext[i].index.nlevels) if ext[i].index.names[p] not in fil]

            # adjustments
            tmp_dtf = dtf.reset_index().reset_index().set_index(fil+['index']).droplevel(-1)
            tmp_ext = ext[i].reset_index().reset_index().set_index(fil+['index']).droplevel(-1)

            # indexing
            mlt = pd.MultiIndex.intersection(dtf.index.droplevel(drp_dtf),ext[i].index.droplevel(drp_ext))

            # assignment
            tmp_dtf.loc[mlt,ttl[i]] = tmp_ext.loc[mlt,ttl[i]]

            # setting
            dtf = tmp_dtf.reset_index().set_index(dtf.index.names)

        except:
            fun_print(None,'Side: post-processing error')
            pass

    if ren is not None:
        dtf = table_rename(dtf,ttl,ren)

    return dtf

def post_intervals(dtf,ttl,fun,bns,lbl,prc,rnd):
    """intervals post-processing

    dtf -> values dataframe;
    ttl -> data title;
    fun -> cutting function;
    bns -> bins number;
    lbl -> labels list
    prc -> precision value.
    rnd -> rounding function.

    """

    if prc is None:
        prc = 3

    if fun == 'cut':
        dtf['%s_intervals'%(ttl)] = pd.cut(dtf.loc[:,ttl],bns,labels=lbl,precision=prc)

    if fun == 'qnt':
        dtf['%s_intervals'%(ttl)] = pd.qcut(dtf.loc[:,ttl],bns,labels=lbl,precision=prc)

    if lbl is None and rnd is not None:
        dtf['%s_intervals'%(ttl)] = dtf['%s_intervals'%(ttl)].apply(lambda x: pd.Interval(left=rnd(x.left),right=rnd(x.right)))

    return dtf

def post_tag(dtf,col,bnd,ass):
    """tag post-processing

    dtf -> values dataframe;
    col -> optional column(s);
    bnd -> optional boundaries;
    ass -> optional assignments.

    """

    if col is None:
        col = dtf.columns.tolist()

    if not isinstance(col,list):
        col = [col]

    for c in col:

        # array
        val = dtf.loc[:,c].to_numpy()

        # nearest values
        out = fun_nearest(val,bnd,ass)

        # reassign
        dtf.loc[:,c] = out

    return dtf

def post_plot(frm,nam,*args):
    """plot post-processing

    frm   > frame dictionary;
    nam   > post-processing name;
    args -> plotting arguments.

    """

    # plotting arguments structure
    # plo -> plot type;
    # ext -> extra dictionary;
    # dtf -> values dataframe;
    # idx -> indexer(s)/list(s);
    # lbl -> data labels;
    # hue -> data hue(s).

    if isinstance(args[0],dict) and len(args) == 1:
        dic = args[0]

    else:
        dic = {}
        for k,v in zip(['plo','ext','dtf','idx','lbl','hue'],args):
            dic[k] = v

    if any(not isinstance(key,int) for key in dic.keys()):
        dic = {0:dic}

    # size
    siz = len(dic)

    # initializing
    plo = [None]*siz
    ext = [None]*siz
    dtf = [None]*siz
    lbl = [None]*siz
    hue = [None]*siz
    flg = [None]*siz

    for k,v in dic.items():

        try:
            fun_print(None,'Plot: post-processing data')

            if 'idx' in v:

                if v['idx'] is not None:

                    if isinstance(v['dtf'],pd.DataFrame):
                        v['dtf'] = table_reduce(v['dtf'],v['idx'])

                    elif isinstance(v['dtf'],list):

                        if isinstance(v['idx'],(pd.Index,pd.MultiIndex)):

                            for idx in range(len(v['dtf'])):

                                # reducing
                                v['dtf'][idx] = table_reduce(v['dtf'][idx],v['idx'])

                        if isinstance(v['idx'],list):

                            for idx in range(len(v['dtf'])):

                                for lvl in range(len(v['idx'])):

                                    if not isinstance(v['idx'][lvl],list):
                                        v['idx'][lvl] = [v['idx'][lvl]]

                                    if len(v['idx'][lvl]) != len(v['dtf']):
                                        v['idx'][lvl] = v['idx'][lvl]*len(v['dtf'])

                                # reducing
                                v['dtf'][idx] = table_reduce(v['dtf'][idx],[i[idx] for i in v['idx']])

            if isinstance(v['dtf'],pd.DataFrame):
                v['dtf'] = v['dtf'].reset_index()

            elif isinstance(v['dtf'],list):
                for idx in range(len(v['dtf'])):
                    v['dtf'][idx] = v['dtf'][idx].reset_index()

            # extraction
            plo[k],ext[k],dtf[k],lbl[k],hue[k] = v['plo'],v['ext'],v['dtf'],v['lbl'],v['hue']

            # flagging
            flg[k] = True

        except:
            fun_print(None,'Plot: post-processing error')
            flg[k] = False

    if all(flg):
        fun_print(None,'Plot: post-processing figure')
        fun_plot(frm,plo,ext,dtf,lbl,hue,nam)

def post_geo(crd,cnt,dtf,ttl,dic,nam):
    """geo post-processing

    crd -> coordinate system;
    cnt -> country codename;
    dtb -> data dataframe for the analyses;
    dtf -> values dataframe;
    ttl -> column title;
    dic -> plotting dictionary;
    nam -> post-processing name.

    """

    # geodataframe
    gdf = geo_dataframe(crd)
    gdf = geo_country(gdf[0],cnt)

    # plotting
    axs = gdf.plot(**dic['gdf'])
    axs = gdf.boundary.plot(ax=axs,**dic['gdf_boundary'])
    axs = dtf.plot(ax=axs,**dic['dtf'])
    axs = dtf.boundary.plot(ax=axs,**dic['dtf_boundary'])

    # figure
    fig = axs.figure

    if nam is not None:
        plot_save(fig,nam)

    return fig,axs

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""BREAKPOINT"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# breakpoint()
