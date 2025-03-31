import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table, State, callback
import os
from datetime import datetime
import dash_bootstrap_components as dbc
import warnings
import numpy as np
from plotly.subplots import make_subplots
import base64
import io
from collections import defaultdict
import tempfile
import pdfkit
from io import BytesIO

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure PDFKit - Update this path to your wkhtmltopdf installation
WKHTMLTOPDF_PATH = os.getenv('WKHTMLTOPDF_PATH', '/usr/local/bin/wkhtmltopdf')
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

def load_and_preprocess(df, sample_size=50000):
    try:
        df.columns = df.columns.str.strip()
        df = df.rename(columns=lambda x: x.lower())
        
        # Auto-detect relevant columns
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower:
                col_map[col] = 'id'
            elif 'code' in col_lower:
                col_map[col] = 'code'
            elif 'value' in col_lower or 'amount' in col_lower or 'price' in col_lower:
                col_map[col] = 'value'
            elif 'date' in col_lower or 'time' in col_lower:
                col_map[col] = 'date'
        
        df = df.rename(columns=col_map)
        
        # Convert value column to numeric if exists
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
        
        # Convert date column to datetime if exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.drop_duplicates()
        
        if len(df) > sample_size:
            print(f"Sampling {sample_size} records from {len(df)}...")
            df = df.sample(n=sample_size, random_state=42)
            
        return df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def parse_uploaded_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # Read all sheets from the Excel file
            xls = pd.ExcelFile(io.BytesIO(decoded))
            sheets_dict = {}
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                processed_df = load_and_preprocess(df)
                if processed_df is not None and not processed_df.empty:
                    sheets_dict[sheet_name] = processed_df
            
            return sheets_dict
        
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            processed_df = load_and_preprocess(df)
            return {'CSV Data': processed_df} if processed_df is not None and not processed_df.empty else {}
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        return {}

def create_dashboard_app():
    # Neumorphism color palette
    colors = {
        'background': '#e0e5ec',  # Soft gray background
        'card_bg': '#e0e5ec',     # Card background same as main background
        'text': '#2c3e50',         # Dark text for contrast
        'primary': '#6c5ce7',      # Purple primary color
        'secondary': '#00cec9',    # Teal secondary color
        'highlight': '#fd79a8',    # Pink highlight color
        'shadow_dark': '#a3b1c6',  # Dark shadow color
        'shadow_light': '#ffffff', # Light shadow color
        'success': '#00b894',      # Green success color
        'warning': '#fdcb6e',      # Yellow warning color
        'danger': '#d63031'        # Red danger color
    }
    
    # Neumorphism shadow styles
    shadow_styles = {
        'regular': (
            '-5px -5px 10px ' + colors['shadow_light'] + ', '
            '5px 5px 10px ' + colors['shadow_dark']
        ),
        'inset': (
            'inset -3px -3px 7px ' + colors['shadow_light'] + ', '
            'inset 3px 3px 7px ' + colors['shadow_dark']
        ),
        'pressed': (
            'inset 3px 3px 7px ' + colors['shadow_dark'] + ', '
            'inset -3px -3px 7px ' + colors['shadow_light']
        ),
        'soft': (
            '-3px -3px 7px ' + colors['shadow_light'] + ', '
            '3px 3px 7px ' + colors['shadow_dark']
        )
    }
    
    color_sequence = px.colors.qualitative.Pastel
    
    app = Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, 
                                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css'])
    
    # Store all loaded datasets in memory
    datasets = defaultdict(dict)
    
    def create_figures(selected_dataset):
        if not selected_dataset or selected_dataset not in datasets:
            return {}
            
        df = datasets[selected_dataset]['data']
        figures = {}
        
        # 1. Top Codes by Value
        if 'code' in df.columns and 'value' in df.columns:
            agg_df = df.groupby('code', observed=True)['value'].agg(['sum', 'count', 'mean']).reset_index()
            agg_df = agg_df.sort_values('sum', ascending=False).head(20)
            
            figures['bar'] = px.bar(
                agg_df, 
                x='code', 
                y='sum',
                color='code',
                color_discrete_sequence=color_sequence,
                title='Top 20 Codes by Total Value',
                hover_data={
                    'code': True,
                    'sum': ':.2f',
                    'count': True,
                    'mean': ':.2f'
                },
                labels={
                    'sum': 'Total Value',
                    'count': 'Number of Records',
                    'mean': 'Average Value'
                },
                height=500
            )
            
            figures['bar'].update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                title_font_size=20,
                xaxis_title="Code",
                yaxis_title="Total Value",
                hovermode="closest",
                legend_title_text='Code',
                coloraxis_showscale=False,
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # 2. Value Distribution Heatmap
            value_min, value_max = df['value'].min(), df['value'].max()
            bin_edges = np.linspace(value_min, value_max, 15)
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
            
            binned_values = np.digitize(df['value'], bin_edges) - 1
            binned_values = np.clip(binned_values, 0, len(bin_labels)-1)
            df_temp = df.copy()
            df_temp['value_bin_label'] = [bin_labels[b] if 0 <= b < len(bin_labels) else bin_labels[0] for b in binned_values]
            
            top_codes = df['code'].value_counts().head(15).index.tolist()
            heatmap_df = df_temp[df_temp['code'].isin(top_codes)]
            
            cross_tab = pd.crosstab(
                index=heatmap_df['code'],
                columns=heatmap_df['value_bin_label'],
                normalize='index'
            )
            
            for label in bin_labels:
                if label not in cross_tab.columns:
                    cross_tab[label] = 0
            
            cross_tab = cross_tab[sorted(cross_tab.columns, key=lambda x: float(x.split('-')[0]))]
            
            figures['heatmap'] = px.imshow(
                cross_tab,
                labels=dict(x="Value Range", y="Code", color="Frequency %"),
                title="Value Distribution Across Top 15 Codes",
                color_continuous_scale='Viridis',
                height=500,
                aspect="auto"
            )
            
            figures['heatmap'].update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                title_font_size=20,
                xaxis_title="Value Range",
                yaxis_title="Code",
                coloraxis_colorbar=dict(
                    title="Frequency %",
                    tickformat=".1%"
                ),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # 3. Box & Violin Plot
            top10_codes = df['code'].value_counts().head(10).index.tolist()
            box_df = df[df['code'].isin(top10_codes)]
            
            figures['box_violin'] = make_subplots(rows=1, cols=2, 
                                                subplot_titles=("Box Plot", "Violin Plot"),
                                                horizontal_spacing=0.1)
            
            box = go.Box(
                x=box_df['code'],
                y=box_df['value'],
                marker_color=colors['primary'],
                name="Distribution",
                boxmean=True
            )
            
            violin = go.Violin(
                x=box_df['code'],
                y=box_df['value'],
                box_visible=True,
                line_color=colors['secondary'],
                fillcolor=colors['secondary'],
                opacity=0.6,
                name="Distribution"
            )
            
            figures['box_violin'].add_trace(box, row=1, col=1)
            figures['box_violin'].add_trace(violin, row=1, col=2)
            
            figures['box_violin'].update_layout(
                title_text="Value Distribution For Top 10 Codes",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                title_font_size=20,
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            figures['box_violin'].update_xaxes(title_text="Code")
            figures['box_violin'].update_yaxes(title_text="Value")
            
        # 4. Summary Statistics
        if 'value' in df.columns:
            stats = df['value'].describe()
            
            figures['stats'] = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=("Mean", "Standard Deviation", "Min", "Max")
            )
            
            # Mean value gauge
            figures['stats'].add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=stats['mean'],
                    number={"valueformat": ".2f"},
                    gauge={
                        'axis': {'range': [0, stats['max']]},
                        'bar': {'color': colors['primary']},
                        'steps': [
                            {'range': [0, stats['50%']], 'color': colors['shadow_light']},
                            {'range': [stats['50%'], stats['75%']], 'color': colors['secondary']},
                            {'range': [stats['75%'], stats['max']], 'color': colors['highlight']}
                        ],
                    },
                    title={"text": "Mean Value"}
                ),
                row=1, col=1
            )
            
            # Standard deviation gauge
            figures['stats'].add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=stats['std'],
                    number={"valueformat": ".2f"},
                    gauge={
                        'axis': {'range': [0, stats['max']/2]},
                        'bar': {'color': colors['secondary']},
                    },
                    title={"text": "Standard Deviation"}
                ),
                row=1, col=2
            )
            
            # Min value indicator
            figures['stats'].add_trace(
                go.Indicator(
                    mode="number",
                    value=stats['min'],
                    number={"valueformat": ".2f"},
                    title={"text": "Minimum Value"}
                ),
                row=2, col=1
            )
            
            # Max value indicator
            figures['stats'].add_trace(
                go.Indicator(
                    mode="number",
                    value=stats['max'],
                    number={"valueformat": ".2f"},
                    title={"text": "Maximum Value"}
                ),
                row=2, col=2
            )
            
            figures['stats'].update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                title_font_size=16,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # 5. Histogram with KDE overlay
            hist_fig = go.Figure()
            
            hist_values, hist_bins = np.histogram(df['value'], bins=50)
            bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
            
            hist_fig.add_trace(go.Bar(
                x=bin_centers,
                y=hist_values,
                name="Frequency",
                marker_color=colors['primary'],
                opacity=0.7
            ))
            
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(df['value'].dropna())
            x_kde = np.linspace(df['value'].min(), df['value'].max(), 1000)
            y_kde = kde(x_kde) * len(df['value']) * (df['value'].max() - df['value'].min()) / 50
            
            hist_fig.add_trace(go.Scatter(
                x=x_kde,
                y=y_kde,
                mode='lines',
                name='Density',
                line=dict(color=colors['highlight'], width=3)
            ))
            
            hist_fig.update_layout(
                title="Value Distribution with Density Curve",
                xaxis_title="Value",
                yaxis_title="Frequency",
                legend_title="Distribution",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                title_font_size=20,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            figures['histogram'] = hist_fig
        
        # 6. Correlation Matrix
        numerical_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numerical_cols) > 1:
            corr_df = df[numerical_cols].corr()
            
            figures['correlation'] = px.imshow(
                corr_df,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix",
                height=500,
                aspect="auto"
            )
            
            figures['correlation'].update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                title_font_size=20,
                margin=dict(l=20, r=20, t=60, b=20)
            )
        
        # 7. Time series
        if 'date' in df.columns:
            try:
                time_df = df.dropna(subset=['date']).groupby([pd.Grouper(key='date', freq='M')])['value'].agg(['sum', 'mean']).reset_index()
                
                figures['timeseries'] = make_subplots(specs=[[{"secondary_y": True}]])
                
                figures['timeseries'].add_trace(
                    go.Scatter(
                        x=time_df['date'], 
                        y=time_df['sum'],
                        mode='lines+markers',
                        name='Total Value',
                        line=dict(color=colors['primary'], width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=False
                )
                
                figures['timeseries'].add_trace(
                    go.Scatter(
                        x=time_df['date'], 
                        y=time_df['mean'],
                        mode='lines+markers',
                        name='Average Value',
                        line=dict(color=colors['secondary'], width=3, dash='dot'),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                figures['timeseries'].update_layout(
                    title_text="Value Trends Over Time",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=colors['text'],
                    title_font_size=20,
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                figures['timeseries'].update_xaxes(title_text="Date")
                figures['timeseries'].update_yaxes(title_text="Total Value", secondary_y=False)
                figures['timeseries'].update_yaxes(title_text="Average Value", secondary_y=True)
            except Exception as e:
                print(f"Error creating time series: {e}")
        
        return figures
    
    # Create the app layout with Neumorphism design
    app.layout = dbc.Container([
        dcc.Store(id='datasets-store', data={}),
        dcc.Store(id='current-dataset', data=None),
        dcc.Download(id="download-pdf"),
        
        dbc.Row([
            dbc.Col(html.H1([
                html.I(className="fas fa-chart-line me-2"),
                "Advanced Data Visualization Dashboard"
            ], className="text-center my-4", style={
                'color': colors['text'],
                'textShadow': f'2px 2px 4px {colors["shadow_dark"]}, -2px -2px 4px {colors["shadow_light"]}'
            }), width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Upload and Manage Datasets", className="fw-bold", style={
                    'backgroundColor': colors['card_bg'],
                    'border': 'none',
                    'boxShadow': shadow_styles['soft']
                }),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files', style={'color': colors['primary']})
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'border': 'none',
                            'borderRadius': '15px',
                            'textAlign': 'center',
                            'margin': '10px 0',
                            'cursor': 'pointer',
                            'background': colors['card_bg'],
                            'boxShadow': shadow_styles['regular'],
                            'color': colors['text'],
                            'transition': 'all 0.2s ease'
                        },
                        multiple=True
                    ),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Dataset", className="fw-bold mb-2", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='dataset-selector',
                                options=[],
                                placeholder="Select a dataset...",
                                style={
                                    'marginBottom': '10px',
                                    'background': colors['card_bg'],
                                    'border': 'none',
                                    'boxShadow': shadow_styles['inset'],
                                    'borderRadius': '15px'
                                }
                            )
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Actions", className="fw-bold mb-2", style={'color': colors['text']}),
                            dbc.ButtonGroup([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-database me-2"),
                                        "Load Dataset"
                                    ],
                                    id="load-dataset",
                                    color="primary",
                                    className="me-2",
                                    style={
                                        'background': colors['primary'],
                                        'border': 'none',
                                        'boxShadow': shadow_styles['regular'],
                                        'borderRadius': '15px'
                                    }
                                ),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-trash-alt me-2"),
                                        "Remove Dataset"
                                    ],
                                    id="remove-dataset",
                                    color="danger",
                                    style={
                                        'background': colors['danger'],
                                        'border': 'none',
                                        'boxShadow': shadow_styles['regular'],
                                        'borderRadius': '15px'
                                    }
                                ),
                            ], style={'width': '100%'})
                        ], width=6)
                    ]),
                    
                    html.Div(id='upload-status', className="mt-2")
                ], style={
                    'background': colors['card_bg'],
                    'borderRadius': '0 0 15px 15px'
                })
            ], style={
                'background': colors['card_bg'],
                'border': 'none',
                'borderRadius': '15px',
                'boxShadow': shadow_styles['regular']
            }), width=12)
        ], className="mb-4"),
        
        html.Div(id='dataset-overview'),
        
        dbc.Row([
            dbc.Col(
                dbc.Button(
                    [
                        html.I(className="fas fa-file-pdf me-2"),
                        "Export to PDF"
                    ],
                    id="export-pdf",
                    color="success",
                    className="mb-3",
                    style={
                        'background': colors['success'],
                        'border': 'none',
                        'boxShadow': shadow_styles['regular'],
                        'borderRadius': '15px'
                    }
                ),
                width=12
            )
        ]),
        
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="overview", tabClassName="fw-bold", style={
                'background': colors['card_bg'],
                'border': 'none',
                'boxShadow': shadow_styles['regular'],
                'borderRadius': '15px 15px 0 0',
                'margin': '5px',
                'color': colors['text']
            }),
            dbc.Tab(label="Detailed Analysis", tab_id="analysis", tabClassName="fw-bold", style={
                'background': colors['card_bg'],
                'border': 'none',
                'boxShadow': shadow_styles['regular'],
                'borderRadius': '15px 15px 0 0',
                'margin': '5px',
                'color': colors['text']
            }),
            dbc.Tab(label="Correlation Analysis", tab_id="correlation", tabClassName="fw-bold", style={
                'background': colors['card_bg'],
                'border': 'none',
                'boxShadow': shadow_styles['regular'],
                'borderRadius': '15px 15px 0 0',
                'margin': '5px',
                'color': colors['text']
            }),
            dbc.Tab(label="Raw Data", tab_id="raw_data", tabClassName="fw-bold", style={
                'background': colors['card_bg'],
                'border': 'none',
                'boxShadow': shadow_styles['regular'],
                'borderRadius': '15px 15px 0 0',
                'margin': '5px',
                'color': colors['text']
            }),
        ], id="main-tabs", active_tab="overview", className="mb-3"),
        
        html.Div(id='tab-content'),
        
        dbc.Row([
            dbc.Col(html.Footer([
                html.Hr(style={'borderColor': colors['shadow_dark']}),
                html.P([
                    html.I(className="fas fa-calendar-alt me-2"),
                    f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    html.Br(),
                    html.Small("Advanced Data Visualization Dashboard v1.0", className="text-muted")
                ], className="text-center mt-4", style={'color': colors['text']})
            ]), width=12)
        ])
    ], fluid=True, style={
        "backgroundColor": colors['background'],
        "minHeight": "100vh",
        "padding": "20px"
    })
    
    # Callback to handle file uploads
    @app.callback(
        [Output('datasets-store', 'data'),
         Output('dataset-selector', 'options'),
         Output('upload-status', 'children')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename'),
         State('datasets-store', 'data')],
        prevent_initial_call=True
    )
    def update_datasets(contents_list, filename_list, existing_data):
        if not contents_list:
            return existing_data, [], "No files uploaded yet."
            
        existing_data = existing_data or {}
        new_datasets = {}
        
        for contents, filename in zip(contents_list, filename_list):
            sheets_dict = parse_uploaded_file(contents, filename)
            
            for sheet_name, df in sheets_dict.items():
                dataset_name = f"{filename} - {sheet_name}" if len(sheets_dict) > 1 else filename
                new_datasets[dataset_name] = {
                    'filename': filename,
                    'sheet_name': sheet_name,
                    'data': df.to_dict('records'),
                    'columns': list(df.columns),
                    'shape': df.shape,
                    'has_code': 'code' in df.columns,
                    'has_value': 'value' in df.columns,
                    'has_date': 'date' in df.columns
                }
        
        # Merge new datasets with existing ones
        updated_data = {**existing_data, **new_datasets}
        options = [{'label': name, 'value': name} for name in updated_data.keys()]
        
        status = f"Successfully uploaded {len(new_datasets)} new dataset(s). Total datasets: {len(updated_data)}"
        
        return updated_data, options, dbc.Alert(status, color="success", dismissable=True, style={
            'background': colors['success'],
            'color': 'white',
            'border': 'none',
            'boxShadow': shadow_styles['soft'],
            'borderRadius': '15px'
        })
    
    # Callback to load a selected dataset
    @app.callback(
        [Output('current-dataset', 'data'),
         Output('dataset-overview', 'children')],
        [Input('load-dataset', 'n_clicks')],
        [State('dataset-selector', 'value'),
         State('datasets-store', 'data')],
        prevent_initial_call=True
    )
    def load_dataset(n_clicks, selected_dataset, stored_data):
        if not n_clicks or not selected_dataset or not stored_data or selected_dataset not in stored_data:
            return None, html.Div()
            
        dataset_info = stored_data[selected_dataset]
        df = pd.DataFrame.from_records(dataset_info['data'])
        
        # Store the DataFrame in memory
        datasets[selected_dataset]['data'] = df
        
        # Create overview cards with Neumorphism style
        overview_cards = []
        
        # Basic stats card
        overview_cards.append(
            dbc.Col(dbc.Card([
                dbc.CardHeader("Basic Statistics", className="fw-bold", style={
                    'background': colors['card_bg'],
                    'border': 'none',
                    'boxShadow': shadow_styles['soft']
                }),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Span("Records:", className="fw-bold", style={'color': colors['text']}),
                            html.Span(f"{len(df):,}", className="ms-2", style={'color': colors['text']})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Columns:", className="fw-bold", style={'color': colors['text']}),
                            html.Span(f"{len(df.columns):,}", className="ms-2", style={'color': colors['text']})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Size:", className="fw-bold", style={'color': colors['text']}),
                            html.Span(f"{df.memory_usage().sum()/(1024*1024):.2f} MB", className="ms-2", style={'color': colors['text']})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Missing Values:", className="fw-bold", style={'color': colors['text']}),
                            html.Span(f"{df.isnull().sum().sum():,}", className="ms-2", style={'color': colors['text']})
                        ])
                    ])
                ], style={
                    'background': colors['card_bg'],
                    'borderRadius': '0 0 15px 15px'
                })
            ], style={
                'background': colors['card_bg'],
                'border': 'none',
                'borderRadius': '15px',
                'boxShadow': shadow_styles['regular'],
                'height': '100%'
            }), width=3, className="mb-4")
        )
        
        # Value stats card (if value column exists)
        if 'value' in df.columns:
            value_stats = df['value'].describe()
            overview_cards.append(
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Value Statistics", className="fw-bold", style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'boxShadow': shadow_styles['soft']
                    }),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("Mean:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{value_stats['mean']:,.2f}", className="ms-2", style={'color': colors['text']})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Min:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{value_stats['min']:,.2f}", className="ms-2", style={'color': colors['text']})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Max:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{value_stats['max']:,.2f}", className="ms-2", style={'color': colors['text']})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Total:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{df['value'].sum():,.2f}", className="ms-2", style={'color': colors['text']})
                            ])
                        ])
                    ], style={
                        'background': colors['card_bg'],
                        'borderRadius': '0 0 15px 15px'
                    })
                ], style={
                    'background': colors['card_bg'],
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': shadow_styles['regular'],
                    'height': '100%'
                }), width=3, className="mb-4")
            )
        
        # Date stats card (if date column exists)
        if 'date' in df.columns:
            date_stats = df['date'].describe(datetime_is_numeric=True)
            overview_cards.append(
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Date Range", className="fw-bold", style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'boxShadow': shadow_styles['soft']
                    }),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("Start:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{df['date'].min().strftime('%Y-%m-%d')}", className="ms-2", style={'color': colors['text']})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("End:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{df['date'].max().strftime('%Y-%m-%d')}", className="ms-2", style={'color': colors['text']})
                            ])
                        ])
                    ], style={
                        'background': colors['card_bg'],
                        'borderRadius': '0 0 15px 15px'
                    })
                ], style={
                    'background': colors['card_bg'],
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': shadow_styles['regular'],
                    'height': '100%'
                }), width=3, className="mb-4")
            )
        
        # Code stats card (if code column exists)
        if 'code' in df.columns:
            overview_cards.append(
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Code Statistics", className="fw-bold", style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'boxShadow': shadow_styles['soft']
                    }),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("Unique Codes:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{df['code'].nunique():,}", className="ms-2", style={'color': colors['text']})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Top Code:", className="fw-bold", style={'color': colors['text']}),
                                html.Span(f"{df['code'].value_counts().index[0]}", className="ms-2", style={'color': colors['text']})
                            ])
                        ])
                    ], style={
                        'background': colors['card_bg'],
                        'borderRadius': '0 0 15px 15px'
                    })
                ], style={
                    'background': colors['card_bg'],
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': shadow_styles['regular'],
                    'height': '100%'
                }), width=3, className="mb-4")
            )
        
        overview = html.Div([
            html.H4(f"Dataset: {selected_dataset}", className="text-center mb-4", style={
                'color': colors['text'],
                'textShadow': f'2px 2px 4px {colors["shadow_dark"]}, -2px -2px 4px {colors["shadow_light"]}'
            }),
            dbc.Row(overview_cards, className="mb-4 g-4")
        ])
        
        return selected_dataset, overview
    
    # Callback to remove a dataset
    @app.callback(
        [Output('datasets-store', 'data', allow_duplicate=True),
         Output('dataset-selector', 'options', allow_duplicate=True),
         Output('upload-status', 'children', allow_duplicate=True),
         Output('current-dataset', 'data', allow_duplicate=True),
         Output('dataset-overview', 'children', allow_duplicate=True)],
        [Input('remove-dataset', 'n_clicks')],
        [State('dataset-selector', 'value'),
         State('datasets-store', 'data'),
         State('current-dataset', 'data')],
        prevent_initial_call=True
    )
    def remove_dataset(n_clicks, selected_dataset, stored_data, current_dataset):
        if not n_clicks or not selected_dataset or not stored_data or selected_dataset not in stored_data:
            return stored_data, [], "No dataset selected for removal.", current_dataset, None
            
        updated_data = {k: v for k, v in stored_data.items() if k != selected_dataset}
        options = [{'label': name, 'value': name} for name in updated_data.keys()]
        
        # Remove from memory if loaded
        if selected_dataset in datasets:
            del datasets[selected_dataset]
        
        status = f"Removed dataset: {selected_dataset}. {len(updated_data)} datasets remaining."
        
        # Clear current dataset if it was the one removed
        if current_dataset == selected_dataset:
            return updated_data, options, dbc.Alert(status, color="warning", dismissable=True, style={
                'background': colors['warning'],
                'color': 'white',
                'border': 'none',
                'boxShadow': shadow_styles['soft'],
                'borderRadius': '15px'
            }), None, None
        
        return updated_data, options, dbc.Alert(status, color="warning", dismissable=True, style={
            'background': colors['warning'],
            'color': 'white',
            'border': 'none',
            'boxShadow': shadow_styles['soft'],
            'borderRadius': '15px'
        }), current_dataset, None
    
    # Callback to update tab content
    @app.callback(
        Output('tab-content', 'children'),
        [Input('main-tabs', 'active_tab'),
         Input('current-dataset', 'data')]
    )
    def update_tab_content(active_tab, selected_dataset):
        if not selected_dataset or selected_dataset not in datasets:
            return dbc.Alert("Please load a dataset first.", color="info", className="my-4", style={
                'background': colors['primary'],
                'color': 'white',
                'border': 'none',
                'boxShadow': shadow_styles['soft'],
                'borderRadius': '15px'
            })
            
        df = datasets[selected_dataset]['data']
        figures = create_figures(selected_dataset)
        
        if active_tab == "overview":
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Key Statistics", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('stats', px.scatter(
                                title="No numerical data available for statistics"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular'],
                        'height': '100%'
                    }), width=6, className="mb-4"),
                    
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Value Distribution", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('histogram', px.scatter(
                                title="No value data available for histogram"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular'],
                        'height': '100%'
                    }), width=6, className="mb-4")
                ], className="g-4"),
                
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Top Codes by Total Value", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('bar', px.scatter(
                                title="No code/value data available"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=6, className="mb-4"),
                    
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Value Distribution Heatmap", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('heatmap', px.scatter(
                                title="No code/value data available"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=6, className="mb-4")
                ], className="g-4")
            ], fluid=True)
        
        elif active_tab == "analysis":
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Value Distribution", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('box_violin', px.scatter(
                                title="No code/value data available for distribution analysis"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=12, className="mb-4")
                ]),
                
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Advanced Filtering", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Codes", className="fw-bold mb-2", style={'color': colors['text']}),
                                    dcc.Dropdown(
                                        id='code-filter',
                                        options=[{'label': code, 'value': code} 
                                               for code in sorted(df['code'].unique())] if 'code' in df.columns else [],
                                        multi=True,
                                        placeholder="Select codes to filter...",
                                        style={
                                            'marginBottom': '15px',
                                            'background': colors['card_bg'],
                                            'border': 'none',
                                            'boxShadow': shadow_styles['inset'],
                                            'borderRadius': '15px'
                                        }
                                    )
                                ], width=6, className="mb-3"),
                                
                                dbc.Col([
                                    html.Label("Value Range", className="fw-bold mb-2", style={'color': colors['text']}),
                                    dcc.RangeSlider(
                                        id='value-range',
                                        min=df['value'].min() if 'value' in df.columns else 0,
                                        max=df['value'].max() if 'value' in df.columns else 1,
                                        value=[df['value'].min() if 'value' in df.columns else 0, 
                                              df['value'].max() if 'value' in df.columns else 1],
                                        marks=None,
                                        allowCross=False,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="p-0"
                                    )
                                ], width=6, className="mb-3")
                            ], className="mb-4"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Chart Type", className="fw-bold mb-2", style={'color': colors['text']}),
                                    dbc.RadioItems(
                                        id='chart-type',
                                        options=[
                                            {'label': ' Box Plot', 'value': 'box'},
                                            {'label': ' Violin Plot', 'value': 'violin'},
                                            {'label': ' Scatter Plot', 'value': 'scatter'},
                                            {'label': ' Bar Chart', 'value': 'bar'}
                                        ],
                                        value='box',
                                        inline=True,
                                        className="ps-2"
                                    )
                                ], width=8),
                                
                                dbc.Col([
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-filter me-2"),
                                            "Apply Filters"
                                        ],
                                        id="apply-filters",
                                        color="primary",
                                        className="w-100 mt-2",
                                        style={
                                            'background': colors['primary'],
                                            'border': 'none',
                                            'boxShadow': shadow_styles['regular'],
                                            'borderRadius': '15px'
                                        }
                                    )
                                ], width=4)
                            ])
                        ], style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=12, className="mb-4")
                ]),
                
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Filtered Results", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-filtered-plot",
                                type="circle",
                                children=dcc.Graph(
                                    id='filtered-plot',
                                    config={'responsive': True}
                                )
                            ),
                            html.Div(id="filtered-stats", className="mt-3")
                        ], style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=12, className="mb-4")
                ])
            ], fluid=True)
        
        elif active_tab == "correlation":
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Correlation Analysis", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('correlation', px.scatter(
                                title="No numerical data available for correlation analysis"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=12, className="mb-4")
                ]),
                
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Time Series Analysis", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody(dcc.Graph(
                            figure=figures.get('timeseries', px.scatter(
                                title="No time series data available"
                            ).update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color=colors['text'],
                                margin=dict(l=20, r=20, t=60, b=20)
                            )),
                            config={'responsive': True}
                        ), style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=12) if 'timeseries' in figures else None
                ])
            ], fluid=True)
        
        elif active_tab == "raw_data":
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Data Preview", className="fw-bold", style={
                            'background': colors['card_bg'],
                            'border': 'none',
                            'boxShadow': shadow_styles['soft']
                        }),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.InputGroupText(
                                    html.I(className="fas fa-search")
                                ),
                                dbc.Input(
                                    id="search-input",
                                    type="text",
                                    placeholder="Search data...",
                                    className="mb-3",
                                    style={
                                        'background': colors['card_bg'],
                                        'border': 'none',
                                        'boxShadow': shadow_styles['inset'],
                                        'borderRadius': '15px'
                                    }
                                ),
                            ], className="mb-3"),
                            
                            dash_table.DataTable(
                                id='raw-data-table',
                                columns=[{"name": i, "id": i} for i in df.columns],
                                data=df.head(100).to_dict('records'),
                                page_size=15,
                                style_table={
                                    'overflowX': 'auto',
                                    'width': '100%',
                                    'borderRadius': '15px'
                                },
                                style_header={
                                    'backgroundColor': colors['primary'],
                                    'color': 'white',
                                    'fontWeight': 'bold',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    'border': 'none'
                                },
                                style_cell={
                                    'height': 'auto',
                                    'minWidth': '100px',
                                    'width': '150px',
                                    'maxWidth': '300px',
                                    'whiteSpace': 'normal',
                                    'textAlign': 'left',
                                    'padding': '10px',
                                    'backgroundColor': colors['card_bg'],
                                    'border': 'none'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': colors['shadow_light']
                                    }
                                ],
                                filter_action="native",
                                sort_action="native",
                                sort_mode="multi",
                                export_format="csv",
                                export_headers="display",
                                tooltip_data=[
                                    {
                                        column: {'value': str(value), 'type': 'markdown'}
                                        for column, value in row.items()
                                    } for row in df.head(100).to_dict('records')
                                ],
                                tooltip_duration=None
                            )
                        ], style={
                            'background': colors['card_bg'],
                            'borderRadius': '0 0 15px 15px'
                        })
                    ], style={
                        'background': colors['card_bg'],
                        'border': 'none',
                        'borderRadius': '15px',
                        'boxShadow': shadow_styles['regular']
                    }), width=12)
                ])
            ], fluid=True)
    
    # Callback for filtered plot in analysis tab
    @app.callback(
        [Output('filtered-plot', 'figure'),
         Output('filtered-stats', 'children')],
        [Input('apply-filters', 'n_clicks')],
        [State('code-filter', 'value'),
         State('value-range', 'value'),
         State('chart-type', 'value'),
         State('current-dataset', 'data')],
        prevent_initial_call=True
    )
    def update_filtered_plot(n_clicks, selected_codes, value_range, chart_type, selected_dataset):
        if not selected_dataset or selected_dataset not in datasets:
            return px.scatter(title="No dataset loaded").update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                margin=dict(l=20, r=20, t=60, b=20)
            ), ""
            
        df = datasets[selected_dataset]['data']
        filtered_df = df
        
        # Apply value filter if value column exists
        if 'value' in df.columns:
            if value_range:
                filtered_df = filtered_df[(filtered_df['value'] >= value_range[0]) & 
                                        (filtered_df['value'] <= value_range[1])]
        
        # Apply code filter if code column exists
        if selected_codes and 'code' in df.columns:
            filtered_df = filtered_df[filtered_df['code'].isin(selected_codes)]
        
        if filtered_df.empty:
            return px.scatter(title="No data matches the selected filters").update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text'],
                margin=dict(l=20, r=20, t=60, b=20)
            ), ""
        
        # Create chart based on type and available columns
        if chart_type == 'box' and 'code' in df.columns and 'value' in df.columns:
            fig = px.box(filtered_df, x='code', y='value', 
                       color='code', 
                       color_discrete_sequence=color_sequence,
                       title=f"Value Distribution by Code ({len(filtered_df)} records)")
        
        elif chart_type == 'violin' and 'code' in df.columns and 'value' in df.columns:
            fig = px.violin(filtered_df, x='code', y='value', 
                          color='code', 
                          color_discrete_sequence=color_sequence,
                          box=True,
                          title=f"Value Distribution by Code ({len(filtered_df)} records)")
        
        elif chart_type == 'scatter' and 'code' in df.columns and 'value' in df.columns:
            fig = px.scatter(filtered_df, x='code', y='value', 
                           color='code',
                           color_discrete_sequence=color_sequence,
                           opacity=0.7,
                           title=f"Value Distribution by Code ({len(filtered_df)} records)")
        
        elif chart_type == 'bar' and 'code' in df.columns and 'value' in df.columns:
            agg_df = filtered_df.groupby('code', observed=True)['value'].agg(['sum', 'count', 'mean']).reset_index()
            fig = px.bar(agg_df, x='code', y='sum', 
                       color='code',
                       color_discrete_sequence=color_sequence,
                       hover_data=['count', 'mean'],
                       title=f"Total Value by Code ({len(filtered_df)} records)")
        else:
            fig = px.scatter(title="Selected chart type not available for this data")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'],
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Create summary stats if value column exists
        stats_div = html.Div()
        if 'value' in df.columns:
            stats = filtered_df['value'].describe().reset_index()
            stats.columns = ['Metric', 'Value']
            
            stats_table = dbc.Table.from_dataframe(
                stats, 
                striped=True, 
                bordered=True,
                hover=True,
                size="sm",
                className="mt-3",
                style={
                    'background': colors['card_bg'],
                    'borderRadius': '15px',
                    'overflow': 'hidden',
                    'boxShadow': shadow_styles['soft']
                }
            )
            
            stats_div = html.Div([
                html.H5(f"Summary Statistics for {len(filtered_df):,} records", className="mb-3", style={
                    'color': colors['text']
                }),
                stats_table
            ])
        
        return fig, stats_div
    
    # Search functionality for data table
    @app.callback(
        Output('raw-data-table', 'data'),
        [Input('search-input', 'value'),
         Input('current-dataset', 'data')],
        prevent_initial_call=True
    )
    def update_table(search_term, selected_dataset):
        if not selected_dataset or selected_dataset not in datasets:
            return []
            
        df = datasets[selected_dataset]['data']
        
        if not search_term:
            return df.head(100).to_dict('records')
        
        try:
            filtered = df[df.astype(str).apply(lambda row: row.str.contains(search_term, case=False)).any(axis=1)]
            return filtered.head(100).to_dict('records')
        except:
            return df.head(100).to_dict('records')
    
    # PDF export callback
    @app.callback(
        Output("download-pdf", "data"),
        [Input("export-pdf", "n_clicks")],
        [State("current-dataset", "data"),
         State("main-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def export_to_pdf(n_clicks, selected_dataset, active_tab):
        if n_clicks is None or not selected_dataset or selected_dataset not in datasets:
            return None
            
        try:
            # Create a temporary HTML file
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_html:
                # Basic HTML structure
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Dashboard Export - {selected_dataset}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .header {{ text-align: center; margin-bottom: 30px; }}
                        .section {{ margin-bottom: 30px; }}
                        .section-title {{ 
                            background-color: {colors['primary']}; 
                            color: white; 
                            padding: 10px; 
                            border-radius: 5px;
                            margin-bottom: 15px;
                        }}
                        .chart {{ margin-bottom: 20px; }}
                        .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #666; }}
                        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                        th {{ background-color: {colors['primary']}; color: white; padding: 8px; text-align: left; }}
                        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                        tr:nth-child(even) {{ background-color: {colors['shadow_light']}; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Data Visualization Dashboard Export</h1>
                        <h2>Dataset: {selected_dataset}</h2>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                """
                
                # Get the figures for the current dataset
                figures = create_figures(selected_dataset)
                df = datasets[selected_dataset]['data']
                
                # Add overview section
                if active_tab == "overview":
                    html_content += '<div class="section"><div class="section-title">Overview</div>'
                    
                    if 'stats' in figures:
                        img_data = figures['stats'].to_image(format="png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        html_content += f'<div class="chart"><h3>Key Statistics</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    
                    if 'histogram' in figures:
                        img_data = figures['histogram'].to_image(format="png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        html_content += f'<div class="chart"><h3>Value Distribution</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    
                    if 'bar' in figures:
                        img_data = figures['bar'].to_image(format="png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        html_content += f'<div class="chart"><h3>Top Codes by Total Value</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    
                    if 'heatmap' in figures:
                        img_data = figures['heatmap'].to_image(format="png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        html_content += f'<div class="chart"><h3>Value Distribution Heatmap</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    
                    html_content += '</div>'
                
                # Add detailed analysis section
                if active_tab == "analysis" and 'box_violin' in figures:
                    html_content += '<div class="section"><div class="section-title">Detailed Analysis</div>'
                    img_data = figures['box_violin'].to_image(format="png")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    html_content += f'<div class="chart"><h3>Value Distribution</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    html_content += '</div>'
                
                # Add correlation analysis section
                if active_tab == "correlation":
                    html_content += '<div class="section"><div class="section-title">Correlation Analysis</div>'
                    
                    if 'correlation' in figures:
                        img_data = figures['correlation'].to_image(format="png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        html_content += f'<div class="chart"><h3>Correlation Matrix</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    
                    if 'timeseries' in figures:
                        img_data = figures['timeseries'].to_image(format="png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        html_content += f'<div class="chart"><h3>Time Series Analysis</h3><img src="data:image/png;base64,{img_base64}" style="width: 100%;"></div>'
                    
                    html_content += '</div>'
                
                # Add raw data section
                if active_tab == "raw_data":
                    html_content += '<div class="section"><div class="section-title">Raw Data Preview</div>'
                    html_content += df.head(100).to_html(index=False, classes="table", border=0)
                    html_content += '</div>'
                
                # Add footer
                html_content += """
                    <div class="footer">
                        <p>Exported from Advanced Data Visualization Dashboard</p>
                    </div>
                </body>
                </html>
                """
                
                tmp_html.write(html_content.encode('utf-8'))
                tmp_html_path = tmp_html.name
            
            # Convert HTML to PDF
            pdf_output = pdfkit.from_file(tmp_html_path, False, configuration=config)
            
            # Clean up temporary file
            os.unlink(tmp_html_path)
            
            # Return the PDF for download
            return dcc.send_bytes(pdf_output, f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
    
    return app

app = create_dashboard_app()
server = app.server

if __name__ == "__main__":
    print("\nStarting the dashboard server...")
    print("\nOpen your browser at http://127.0.0.1:8050/ to view the dashboard")
    app.run(debug=False, port=8060)