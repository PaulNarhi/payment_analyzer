import numpy_financial as npf
import pandas as pd
import numpy as np
from datetime import date
from pandas.tseries.offsets import BDay, MonthBegin
import matplotlib.pyplot as plt
from scipy import interpolate
import eurostat
from joblib import Memory
import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objs as go
from dash import dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template(["cyborg", "darkly"])

memory = Memory("cachedir")

@memory.cache
def load_base_rate():
    today = date.today()
    # The ECB provides yields for maturities ranging from 3 months till 30 years
    month_maturities = range(3, 12)
    year_maturities = range(1, 31)
    maturities = ['M'+str(maturity) for maturity in month_maturities]
    maturities.extend(['Y'+str(maturity) for maturity in year_maturities])

    # Taking the last 10 business days, and converting to decimal fractions
    euro_curves = eurostat.get_data_df('irt_euryld_d', filter_pars={
        'startPeriod': (today-BDay(10)).date(), 'freq': 'D',
        'yld_curv': 'SPOT_RT',
        'maturity': maturities,
        'bonds': 'CGB_EA_AAA', 'geo': 'EA'})

    return euro_curves

def get_latest_date(data: pd.DataFrame) -> pd.Timestamp:
    """
    Find the latest date in the columns of the DataFrame, ignoring non-date columns.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing date columns.
    
    Returns:
    - pd.Timestamp: The latest valid date from the columns.
    """
    # Filter the columns that can be converted to datetime
    date_columns = pd.to_datetime(data.columns, errors='coerce', format='%Y-%m-%d')
    
    # Drop NaT (Not a Time) values (invalid dates)
    valid_dates = date_columns.dropna()
    
    latest_date = valid_dates.max()
    
    # Format the latest date as a string in 'YYYY-MM-DD' format
    return latest_date.strftime('%Y-%m-%d')


def yield_series(euro_curves):
    euro_rates = euro_curves[euro_curves['maturity'].str.contains('Y')]
    maturity = euro_rates['maturity'].str[1:].astype('int')
    rates = euro_rates[get_latest_date(euro_rates)].values
    return pd.Series(rates, index=maturity)

def apply_irrbb_shocks(yield_curve: pd.Series, shock_type: str, shock_size: float = 100, use_smoothing=False) -> pd.Series:
    """
    Apply IRRBB shocks to a yield curve according to standard regulatory scenarios.
    
    Parameters:
    - yield_curve (pd.Series): The yield curve as a pandas series where the index represents maturities and values represent interest rates.
    - shock_type (str): The type of shock to apply ('parallel', 'steepen', 'flatten').
    - shock_size (float): The size of the shock in basis points (default is 100 for a 100bps shock).
    
    Returns:
    - pd.Series: The adjusted yield curve after applying the shock.
    """
    
    # Convert shock size from basis points to decimal
    shock_decimal = shock_size / 100
    
    # Apply shock based on shock type
    if shock_type == 'parallel up':
        # Parallel shift: Apply the same shock across all maturities
        adjusted_curve = yield_curve + shock_decimal
    elif shock_type == 'parallel down':
        adjusted_curve = yield_curve - shock_decimal
    elif shock_type == 'flatten':
        # Steepening: Larger shock to the long end, smaller shock to the short end
        adjusted_curve = yield_curve.copy()
        
        # Apply a smaller shock to short maturities (let's assume up to 2 years is short)
        short_end = yield_curve.index <= 2
        adjusted_curve[short_end] -= shock_decimal
        
        # Apply a larger shock to long maturities (let's assume maturities > 10 years are long)
        long_end = yield_curve.index > 10
        adjusted_curve[long_end] += shock_decimal
        
    elif shock_type == 'steepen':
        # Flattening: Larger shock to short end, smaller shock to long end
        adjusted_curve = yield_curve.copy()
        
        # Apply a larger shock to short maturities
        short_end = yield_curve.index <= 2
        adjusted_curve[short_end] += shock_decimal
        
        # Apply a smaller shock to long maturities
        long_end = yield_curve.index > 10
        adjusted_curve[long_end] -= shock_decimal
    elif shock_type == 'short rates up':
        adjusted_curve = yield_curve.copy()
        
        # Apply a larger shock to short maturities
        short_end = yield_curve.index <= 2
        adjusted_curve[short_end] += shock_decimal
    elif shock_type == 'short rates down':
        adjusted_curve = yield_curve.copy()
        
        # Apply a larger shock to short maturities
        short_end = yield_curve.index <= 2
        adjusted_curve[short_end] -= shock_decimal
        
    else:
        raise ValueError("Unsupported shock type. Correct types 'parallel up/down', 'steepen', 'flatten' or 'short rates up/down'.")
    if use_smoothing:
        tck = interpolate.splrep(adjusted_curve.index, adjusted_curve.values, k=5, s=1000)
        x2 = adjusted_curve.index
        y2 = interpolate.splev(x2, tck)
        adjusted_curve = pd.Series(y2, index=x2)
    return adjusted_curve

def pay_debt_annuity(principal, annual_rate, years_left):
    months_per_year = 12
    # Create a payment period for one year
    per = np.arange(years_left * months_per_year) + 1

    # Calculate the monthly interest rate and the total number of periods
    monthly_rate = annual_rate / months_per_year
    total_periods = years_left * months_per_year

    # Use PMT, IPMT, and PPMT functions to calculate the monthly payments, interest, and principal repayments
    pmt = npf.pmt(monthly_rate, total_periods, principal)
    ipmt = npf.ipmt(monthly_rate, per, total_periods, principal)
    ppmt = npf.ppmt(monthly_rate, per, total_periods, principal)

    # Define the format for displaying the results
    fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'

    # Loop through the periods to print the schedule for the first year
    remaining_principal = principal
    rows = []
    for payment in per[:12]: # This is essentially the euribor 12m vs 6m etc...
        index = payment - 1
        remaining_principal += ppmt[index]
        rows.append([-ppmt[index], -ipmt[index], remaining_principal])
        #print(fmt.format(payment, ppmt[index], ipmt[index], remaining_principal))
    return pd.DataFrame(rows, columns=['principal payment', 'interest payment', 'remaining principal'])

def pay_debt_constant_amortization(principal_payment, principal_left, interest_rate): 
    # Number of months per year
    months_per_year = 12
    monthly_rate = interest_rate / months_per_year

    # Create periods for one year (12 months)
    per = np.arange(1, months_per_year + 1)

    # Prepare to store the schedule
    rows = []
    
    # Loop through each month to calculate the payments and remaining principal
    for payment in per:
        # Calculate the interest for the current month based on the remaining principal
        interest_payment = principal_left * monthly_rate
        
        # The total payment is the sum of the principal and interest payments
        total_payment = principal_payment + interest_payment
        
        # Subtract the fixed principal payment from the remaining principal
        principal_left -= principal_payment
        
        # Append the data for this period to the rows
        rows.append([principal_payment, interest_payment, principal_left])
    
    # Return a DataFrame with the results
    return pd.DataFrame(rows, columns=['principal payment', 'interest payment', 'remaining principal'])

def pay_debt(principal, interest_rates, payment_type):
    frames = []
    principal_left = principal
    maturity = len(interest_rates)
    for rate, years_left in zip(interest_rates, range(maturity, 0, -1)):
        if payment_type == 'annuity':
            df = pay_debt_annuity(principal_left, rate, years_left=years_left)
            principal_left = df['remaining principal'].values[-1]
            frames.append(df)
        elif payment_type == 'constant amortization':
            payment = principal / (12 * len(interest_rates))
            df = pay_debt_constant_amortization(payment, principal_left, rate)
            principal_left = df['remaining principal'].values[-1]
            frames.append(df)
    df = pd.concat(frames).reset_index(drop=True)
    df.index = np.arange(1, len(df)+1)
    return df

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = html.Div(
    children=[
        html.H1("Mortgage Payment Analysis", style={'textAlign': 'center'}),
        html.Hr(),
        # Row for loan amount, down payment, and tenor slider
        html.Div(
            children=[
                # Loan Amount Header and Input
                html.Div(
                    children=[
                        html.H3("Loan Amount", style={'marginBottom': '10px', 'fontSize': '20px'}),
                        dcc.Input(
                            id='loan-amount',
                            type='number',
                            value=100000,
                            placeholder="Loan Amount",
                            style={'width': '100%', 'padding': '5px'}
                        ),
                    ],
                    style={'flex': '1', 'padding': '0 10px', 'maxWidth': '300px'}
                ),
                # Down Payment Header and Input
                html.Div(
                    children=[
                        html.H3("Down Payment", style={'marginBottom': '10px', 'fontSize': '20px'}),
                        dcc.Input(
                            id='down-payment',
                            type='number',
                            value=20000,
                            placeholder="Down Payment",
                            style={'width': '100%', 'padding': '5px'}
                        ),
                    ],
                    style={'flex': '1', 'padding': '0 10px', 'maxWidth': '300px'}
                ),
                # Tenor Slider Header and Slider
                html.Div(
                    children=[
                        html.H3("Loan Tenor (Years)", style={'marginBottom': '10px', 'fontSize': '25px'}),
                        dcc.Slider(
                            5, 30, 5,
                            value=25,
                            id='tenor-slider',
                        ),
                    ],
                    style={'flex': '1', 'padding': '0 10px'}
                ),
            ],
            style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}
        ),
        # Shock Type Header and Radio Items
        html.Div(
            children=[
                html.H3("Interest Rate Shock Type", style={'marginBottom': '10px', 'fontSize': '20px'}),
                dcc.RadioItems(
                    options=[
                        {'label': 'Base Case', 'value': 'base case'},
                        {'label': 'Parallel up', 'value': 'parallel up'},
                        {'label': 'Parallel down', 'value': 'parallel down'},
                        {'label': 'Steepen', 'value': 'steepen'},
                        {'label': 'Flatten', 'value': 'flatten'},
                        {'label': 'Short rates up', 'value': 'short rates up'},
                        {'label': 'Short rates down', 'value': 'short rates down'},
                    ],
                    value='base case',
                    id='shock-type',
                    style={'display': 'block'}
                ),
            ],
            style={'marginBottom': '20px'}
        ),
        # Graph Type Dropdown Header
        html.Div(
            children=[
                html.H3("Select Graph Type", style={'marginBottom': '10px', 'fontSize': '20px'}),
                dcc.Dropdown(
                    options=[
                        {'label': 'Monthly payments', 'value': 'payment'},
                        {'label': 'Interest Rate', 'value': 'interest rate'}
                    ],
                    value='payment',
                    id='graph-selector',
                    style={
                        'width': '200px',  # Set the width of the dropdown
                        'height': '50px',
                        'margin': '0',  # Align to the left by removing auto margin
                        'color': 'black',
                        'fontSize': '16px',  # Adjust font size
                        'padding': '5px',  # Padding for better alignment
                    },
                    clearable=False
                ),
            ],
            style={'width': 'auto', 'marginBottom': '20px'}
        ),
        # Use Smoothing Checkbox
        html.Div(
            children=[
                html.H3("Use interest rate smoothing", style={'marginBottom': '10px', 'fontSize': '20px'}),
                dcc.Checklist(
                    options=[
                        {'label': 'Enable Smoothing', 'value': 'smooth'}
                    ],
                    value=[],  # Default to no smoothing
                    id='use-smoothing',
                    inline=True,
                    style={'fontSize': '14px'}
                ),
            ],
            style={'marginBottom': '20px'}
        ),
        # Graphs for displaying results
        dcc.Graph(id='payment-graph'),
        dcc.Graph(id='interest-rate-graph', style={'display': 'none'}),  # Initially hidden
    ],
    style={'padding': '20px'}
)

@app.callback(
    Output('payment-graph', 'figure'),
    Output('interest-rate-graph', 'figure'),
    Output('payment-graph', 'style'),
    Output('interest-rate-graph', 'style'),
    [
        Input('graph-selector', 'value'),
        Input('shock-type', 'value'),
        Input('tenor-slider', 'value'),
        Input('use-smoothing', 'value')  # New input for use_smoothing
    ]
)
def update_graphs(graph_type, shock_type, tenor, use_smoothing):
    # Load rate data
    euro_rates = load_base_rate()
    base_rate = yield_series(euro_rates)
    rate = base_rate
    
    if shock_type != 'base case':
        if 'smooth' in use_smoothing:
            rate = apply_irrbb_shocks(base_rate, shock_type, use_smoothing=True)
        else:
            rate = apply_irrbb_shocks(base_rate, shock_type, use_smoothing=False)
    
    # Apply smoothing if enabled

    # Payment Analysis Graph
    ca = pay_debt(100000, rate[:tenor] / 100, 'constant amortization')
    an = pay_debt(100000, rate[:tenor] / 100, 'annuity')
    payment_fig = go.Figure()
    payment_fig.add_trace(go.Scatter(
        x=ca.index,
        y=ca['principal payment'] + ca['interest payment'],
        mode='lines',
        name='Constant amortization'
    ))
    payment_fig.add_trace(go.Scatter(
        x=an.index,
        y=an['principal payment'] + an['interest payment'],
        mode='lines',
        name='Annuity'
    ))
    payment_fig.update_layout(
        title=f"Monthly payments with {shock_type} interest rates",
        xaxis={
            'tickvals': [i * 12 for i in range(1, tenor + 1)],  # Monthly indices for each year
            'ticktext': [f"Y+{i}" for i in range(1, tenor + 1)],  # Labels as "Y+1", "Y+2", ...
            'title': 'Year'
        },
        yaxis_title="Payment Amount",
        plot_bgcolor='rgb(30, 30, 30)',
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(color='white'),
        template='plotly_dark',
        margin=dict(t=40, b=40, l=40, r=40),
        height=600
    )

    # Interest Rate Graph
    rate_fig = go.Figure()
    rate_fig.add_trace(go.Scatter(
        x=rate.index[:tenor],
        y=rate[:tenor],
        mode='lines',
        name='Interest Rate'
    ))
    rate_fig.update_layout(
        title="Selected Interest Rate Over Time",
        xaxis_title="Period",
        yaxis_title="Rate (%)",
        plot_bgcolor='rgb(30, 30, 30)',
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(color='white'),
        template='plotly_dark',
        margin=dict(t=40, b=40, l=40, r=40),
        height=600
    )

    # Toggle between graphs
    if graph_type == 'payment':
        return payment_fig, rate_fig, {'display': 'block'}, {'display': 'none'}
    else:
        return payment_fig, rate_fig, {'display': 'none'}, {'display': 'block'}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


# if __name__ == '__main__':
#     euro_rates = load_base_rate()
#     base_rate = yield_series(euro_rates)
#     rate = apply_irrbb_shocks(base_rate, 'flatten', use_smoothing=True)
#     ca = pay_debt(100000, rate[:25]/100, 'constant amortization')
#     an = pay_debt(100000, rate[:25]/100, 'annuity')
#     plt.plot(list(range(12*25)), ca['principal payment'] + ca['interest payment'],label = 'Constant amortization')
#     plt.plot(list(range(12*25)), an['principal payment'] + an['interest payment'], label = 'Annuity')
#     plt.legend()
#     plt.show()