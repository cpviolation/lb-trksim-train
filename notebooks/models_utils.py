from json import dump,load
from yaml import safe_load
import numpy as np
from os import path,environ,walk
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def GetModelVariantList(model_dir:str):
    """return list of existing model variants"""
    variants = next(walk(model_dir))[1]
    for d in ["assets","variables"]:
        variants.remove(d)
    # remove hidden folders
    variants = [d for d in variants if not d[0] == '.']
    return variants


def Stringfy(element):
    "recursively convert into string all dictionary values"
    for k,val in element.items():
        if isinstance(val,dict):
            element[k] = Stringfy(val)
        else:
            element[k] = str(val)
    return element

def GetModelProperties(model=None,history=None):
    """write the property of the model (and history) in the path"""
    
    assert (model is not None or history is not None), "either model or history must be given"
    
    info2save = dict(model={})
    
    if history is not None:
        model                = history.model
        info2save['history'] = history.history
    
    info2save['model']['layers'] = dict()      # store layer config
    info2save['model']['layers_type'] = dict() # count layer types
    for i,L in enumerate(model.layers,start=1):
        info2save['model']['layers'][f'layer {i}'] = L.get_config()
        info2save['model']['layers'][f'layer {i}'].update(
            {
                'module': L.__module__,
                'class_name': L.__class__.__name__
            })
        
        # increment counter or initialize it
        L_type = L.__class__.__name__
        if L_type in info2save['model']['layers_type'].keys():
            info2save['model']['layers_type'][L_type] = info2save['model']['layers_type'][L_type]+1
        else:
            info2save['model']['layers_type'][L_type] = 1
            
    info2save['model']['N_layers'] = len(info2save['model']['layers'])
    info2save['model']['optimizer'] = Stringfy(model.optimizer.get_config())
    return info2save

def diffDict(dictA : dict, dictB : dict):
    """Return dict with models parameters that are different in dictB w.r.t. dictA
       
       It assumes that type(dictA[key]) == type(dictB[key]) for each key
    """
    
    diff_dict = dict()
    for k in dictA.keys():
        if k in dictB.keys():
            if isinstance(dictA[k],dict) and isinstance(dictB[k],dict):
                diff_dict[k] = diffDict(dictA[k],dictB[k])
            else:
                if dictA[k] != dictB[k]:
                    diff_dict[k] = (dictA[k],dictB[k])
        else:
            #diff_dict.update({'missing_keys': diff_dict.get('missing_keys','')+k+r",\n"})
            diff_dict.update({'missing_keys': {**diff_dict.get('missing_keys',{}),**{k:''}}})
    # remove empty keys
    ddk = list(diff_dict.keys())
    for k in ddk:
        if not diff_dict[k]:
            diff_dict.pop(k)
    return diff_dict
    
def dict_depth(dic, level = 0):
    """to find depth of a dictionary"""
    if not isinstance(dic, dict) or not dic:
        return level
    return max(dict_depth(dic[key], level + 1) for key in dic.keys())
 
def IndentDictMD(dictionary,lines=[],col=1,max_col=1):
    """print dictionray content in MarkDown table 'col'th column"""
    
    #assert col < max_col, f"The table should have at least {col} columns"
    if col < max_col:
        for k,val in dictionary.items():
            if isinstance(val,dict):
                lines.append("| "*col + f"{k}:" + " | "*(max_col-col))
                IndentDictMD(val,lines=lines,col=col+1,max_col=max_col)
            else:
                lines.append("| "*col + f"{k}: {val}" + " | "*(max_col-col))
    return lines

def MakeMD(props:dict,head="Head",mdfile=None,ret=False,maxdepth=None):
    """Print MD file with model properties"""

    lines = []
    max_depth = maxdepth+1 if maxdepth is not None else dict_depth(props)+1
    lines.append(f"| {head} " + " | " * (max_depth-1) + " |"+"\n")
    lines.append("|" + " | ".join(["---"] * max_depth) + "|"+"\n")

    for line in IndentDictMD(props,lines=[],max_col=max_depth):
        lines.append("| " + line+"\n")
    
    if ret:
        return lines
    elif mdfile is None:
        for line in lines:
            print(line,end='')
    else:
        with open(mdfile,'w') as md_output:
            for line in lines:
                md_output.write(line)

def plotHistory(variants: dict, metrics=["loss"]):
    """Plot metrics of different variants"""
    
    for m in metrics:
        # print figure for each metrics
        fig,axes = plt.subplots(2,1,figsize=(7,7))
        
        for v in variants.keys():
            # plot metric for train sample
            if 'history' not in variants[v]: continue
            
            epochs = np.arange(1,len(variants[v]['history'][m])+1)
            axes[0].plot(epochs,variants[v]['history'][m],label=v)
            axes[1].plot(epochs,variants[v]['history'][f"val_{m}"],label=v)
            
            axes[0].set_title("train sample")
            axes[1].set_title("validation sample")
            
        for ax in axes:
            ax.legend()
            ax.set_xlabel("epoch")
            ax.set_ylabel(m)
            ax.grid()
            
        plt.tight_layout()
        plt.show()

def LoadModelsInfos(model_dir:str):
    """return dictionary with information on saved models and its variants"""
    
    variants = GetModelVariantList(model_dir)
    model_name = path.basename(model_dir)
    models = dict()
    
    if "model_info.json" in next(walk(model_dir))[2]:
        # properties of default model already present
        with open(path.join(model_dir,"model_info.json"),'r') as prop_file:
            models[model_name+"_default"] = load(prop_file)
    else:
        # load model and get properties
        models[model_name+'_default']=GetModelProperties(model=load_model(model_dir))

    for v in variants:
        # load description stored in json file
        prop_file = path.join(model_dir,v,"model_info.json")
        with open(prop_file,'r') as prop_file:
            models[model_name+"_"+v] = load(prop_file)
    
    return models

import pandas as pd
from io import StringIO
from IPython.display import Markdown, display

def DisplayMD(source):
    
    if isinstance(source,list):
        md_str = ''.join(source)
    else:
    # Load Markdown file as string
    
        with open(source, 'r') as f:
            md_str = f.read()

    # Parse Markdown string into a Pandas DataFrame
    df = pd.read_table(StringIO(md_str), delimiter='|')
    df = df.drop('Unnamed: 0',axis=1).drop(0,axis=0)
    df = df.rename(columns={k: '' for k in df.columns[1:]})

    # Format DataFrame with Pandas styling
    styled_df = df.style.set_table_styles([{'selector': 'th',
                                            'props': [('background', '#f2f2f2'),
                                                      ('font-weight', 'bold'),
                                                      ('text-align', 'left')]},
                                           {'selector': 'td',
                                            'props': [('text-align', 'left')]}])

    # Display styled DataFrame as Markdown
    display(Markdown(styled_df.to_html()))

def GetTrainFolder(info,key:str):
    """read config file with models training dataset information"""
    # load configuration file
    with open(info, 'r') as file:
        if "yaml" in info or 'yml' in info:
            train_conf = safe_load(file)
        elif 'json' in info or 'jsn' in info:
            train_conf = load(file)
        else:
            raise Exception("Unknown file format")
    # return path
    return train_conf['training_folder'].get(key,None)
    
def reljoin(*paths):
    """join folders ignoring intermediate absolute path"""
    folders = list(paths)
    for i in range(1,len(folders)):
        if folders[i][0] == '/' :
            folders[i] = folders[i][1:]
    return path.join(*folders)
        
