import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import plotly_express as px
import plotly.graph_objects as go
import os

def violin_plot ( rf, save_path ):
    print("violin plotting...")
    Names = [ 'nodule', 'trachea', 'strap_muscle', 'artery', 'vein', 'esophagus' ]
    sns.violinplot( data = rf[ Names ],
                    scale = 'width',
                    #bw =.1,
                    palette = 'Set2')

    if not os.path.exists(save_path+'distribution'):
        os.makedirs(save_path+'distribution')               
    plt.pyplot.savefig( save_path+'distribution/' + 'violin_distribution_area.png' )
    
def box_plot( rf, save_path ):
    print("Box plotting...")
    Names = [ 'nodule', 'trachea', 'strap_muscle', 'artery', 'vein', 'esophagus' ]
    results = []
    for i in range(len(Names)):
        results.append(rf[Names[i]])
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
    
    fig = go.Figure()
    for xd, yd, cls in zip(Names, results, colors):
        fig.add_trace(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=2,
            line_width=1)
        )

    fig.update_layout(
        title='Each class area distribution',
        yaxis=dict(autorange=True,
                   showgrid=True,
                   zeroline=True,
                   dtick=5,
                   gridcolor='rgb(255, 255, 255)',
                   gridwidth=1,
                   zerolinecolor='rgb(255, 255, 255)',
                   zerolinewidth=2,),
        margin=dict( l=40,r=30,b=80,t=100,),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False) 

    if not os.path.exists(save_path+'distribution'):
        os.makedirs(save_path+'distribution') 
    fig.write_image( save_path+'distribution/' + 'box_distribution_area.png')      

def load_txt(dir_path):
    Cls = { 'nodule': [], 'trachea': [], 'strap_muscle': [], 'artery': [], 'vein': [], 'esophagus': [] }
    files = os.listdir(dir_path)
    out_range= []
    for f in files:
        if f.endswith(".txt"):
            fidx = f.split('.',1)
            index = fidx[0]
            df = pd.read_csv( dir_path + f, skiprows = 3, names = ["info"] )
            df = pd.DataFrame(df['info'].str.split(':').tolist(), columns =['class', 'area'])
            df['area'] = df['area'].str.split('%').str.get(0)
            n = len(df)
            for i in range(n):
                if df['class'][i] == 'nodule Area':
                    Cls['nodule'].append(int(df['area'][i])) 
                    if int(df['area'][i]) >= 100:
                        out_range.append(f)
                if df['class'][i] == 'trachea Area':
                    Cls['trachea'].append(int(df['area'][i])) 
                if df['class'][i] == 'strap_muscle Area':
                    Cls['strap_muscle'].append(int(df['area'][i])) 
                if df['class'][i] == 'artery Area':
                    Cls['artery'].append(int(df['area'][i])) 
                if df['class'][i] == 'vein Area':
                    Cls['vein'].append(int(df['area'][i])) 
                if df['class'][i] == 'esophagus Area':
                    Cls['esophagus'].append(int(df['area'][i])) 
    

    rf = pd.DataFrame()
    rf['nodule'] = Cls['nodule']
    rf.loc[:,'trachea'] = pd.Series(Cls['trachea'])
    rf.loc[:,'strap_muscle'] = pd.Series(Cls['strap_muscle'])
    rf.loc[:,'artery'] = pd.Series(Cls['artery'])
    rf.loc[:,'vein'] = pd.Series(Cls['vein'])
    rf.loc[:,'esophagus'] = pd.Series(Cls['esophagus'])
    rf.loc[:,'out_range'] = pd.Series(out_range)
    rf.to_excel(dir_path + 'df.xlsx')
    return rf
    
if __name__ == '__main__':
    dir_path = './yolov4_12a/train_img/'
    save_path = './yolov4_12a/'
    all_data = load_txt(dir_path)
    violin_plot ( all_data, save_path )
    box_plot( all_data, save_path )
