import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np
import json
import os
import argparse



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='Bscan_plotly_viewer.py',
    description='B-scan plot viewer with annotations on the web browser',
    epilog='End of help message',
    usage='python tools/Bscan_plotly_viewer.py [data_path]',
)
parser.add_argument('data_path', help='Path to the txt file of data.')
args = parser.parse_args()


#* Parameters
sample_interval = 0.312500e-9  # [s]
trace_interval = 3.6e-2 # [m], [Li et al. (2020), Sci. Adv.]

# データの読み込み
print('Loading data...')
data = np.loadtxt(args.data_path, delimiter=' ')
print('Successfully loaded data. \n Data shape:', data.shape)
print('   ')


#* Define the Dash app
app = dash.Dash(__name__)


#* Define the annotation file path
ANNOTATION_FILE = os.path.dirname(args.data_path) + '/annotations.json'

# 保存された注釈を読み込む関数
def load_annotations():
    if os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, 'r') as f:
            annotations = json.load(f)
        return annotations
    else:
        return []

# 初期の注釈を読み込む
initial_annotations = load_annotations()


# ヒートマップの作成（Heatmapglを使用）
heatmap = go.Heatmapgl(
    z=data,
    colorscale='Viridis',
    zsmooth='fast',
    # オプションで軸の範囲を設定
    x0=0,
    dx=trace_interval,
    y0=0,
    dy=sample_interval
)

# レイアウトの設定
layout = go.Layout(
    title = str(args.data_path),
    xaxis=dict(
        title='x [m]',
        # オプションで軸範囲を設定
        range=[0, data.shape[1] * trace_interval]
    ),
    yaxis=dict(
        title='Time [ns]',
        autorange='reversed',
        range=[data.shape[0] * sample_interval / 1e-9, 0]
    ),
)


# Figureオブジェクトの作成
# Figureの作成
fig = go.Figure(data=[heatmap], layout=layout)


app.layout = html.Div([
    dcc.Graph(
        id='b-scan-plot',
        figure=fig
    ),
    html.Div([
        html.Label('注釈テキスト:'),
        dcc.Input(id='annotation-text', type='text', value='メモ', style={'marginRight': '10px'}),
        html.Button('注釈を追加', id='add-annotation-button')
    ])
])

@app.callback(
    Output('b-scan-plot', 'figure'),
    [Input('add-annotation-button', 'n_clicks')],
    [State('b-scan-plot', 'figure'),
     State('annotation-text', 'value'),
     State('b-scan-plot', 'clickData')]
)
def add_annotation(n_clicks, fig, text, clickData):
    if n_clicks is None or clickData is None:
        return fig
    point = clickData['points'][0]
    x = point['x']
    y = point['y']
    new_annotation = dict(
        x=x,
        y=y,
        xref='x',
        yref='y',
        text=text,
        showarrow=True,
        arrowhead=1
    )
    if 'annotations' in fig['layout']:
        fig['layout']['annotations'].append(new_annotation)
    else:
        fig['layout']['annotations'] = [new_annotation]
    
    # 注釈をファイルに保存
    with open(ANNOTATION_FILE, 'w') as f:
        json.dump(fig['layout']['annotations'], f)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
