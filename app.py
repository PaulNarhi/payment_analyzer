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
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    children=[
        html.H1("Mortgage Payment Analysis", style={'textAlign': 'center', 'color': '#fff'}),
        html.Hr(),
        # RadioItems for shock type selection
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
            style={'color': '#fff'}
        ),
        # Graph for plotting payments
        dcc.Graph(id='payment-graph', style={'height': '600px'}),
    ],
    style={'backgroundColor': '#333', 'padding': '20px'}
)

# Callback to update the graph and table based on the selected shock type
@app.callback(
    Output('payment-graph', 'figure'),
    [Input('shock-type', 'value')]
)
def update_graph(shock_type):

    euro_rates = load_base_rate()
    base_rate = yield_series(euro_rates)
    rate = base_rate
    if shock_type != 'base case':
        rate = apply_irrbb_shocks(base_rate, shock_type, use_smoothing=True)
    # Calculate the payments for the two amortization methods
    ca = pay_debt(100000, rate[:25] / 100, 'constant amortization')
    an = pay_debt(100000, rate[:25] / 100, 'annuity')
    # This is just an example, you'll replace this with your actual data
    fig = go.Figure()

    # Use ca.index instead of 'period'
    fig.add_trace(go.Scatter(
        x=ca.index, y=ca['principal payment'] + ca['interest payment'],
        mode='lines', name='Constant amortization'
    ))
    
    fig.add_trace(go.Scatter(
        x=an.index, y=an['principal payment'] + an['interest payment'],
        mode='lines', name='Annuity'
    ))

    # Update the layout to have dark theme
    fig.update_layout(
        title=f"Payments with {shock_type} Shock",
        xaxis_title="Period",
        yaxis_title="Payment Amount",
        plot_bgcolor='rgb(30, 30, 30)',
        paper_bgcolor='rgb(30, 30, 30)',
        font=dict(color='white'),
        template='plotly_dark',  # Dark theme for Plotly
        margin=dict(t=20, b=40, l=40, r=40),
        height=600  # Set height of the plot
    )

    return fig

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